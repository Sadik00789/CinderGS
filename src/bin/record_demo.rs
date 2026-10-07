use cinder_gs::app::OrbitCamera;
use cinder_gs::deformation::{CageSpringSimulator, TetMesh};
use cinder_gs::render::pipeline::{ExecutionBackend, RenderPipelineOrchestrator};
use cinder_gs::render::rasterizer::CompositorUniforms;
use cinder_gs::scene::ply_loader::PlyLoader;
use cinder_gs::scene::GaussianSceneSoa;
use glam::Vec3;
use std::f32::consts::PI;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

fn generate_synthetic_scene(count: usize) -> (GaussianSceneSoa, Vec3, f32) {
    let mut scene = GaussianSceneSoa::with_capacity(count);
    let golden_ratio = (1.0 + 5.0f32.sqrt()) / 2.0;

    for i in 0..count {
        let t = (i as f32 + 0.5) / count as f32;
        let r = 1.2 * t.cbrt();
        let theta = (1.0 - 2.0 * t).clamp(-1.0, 1.0).acos();
        let phi = 2.0 * PI * (i as f32 / golden_ratio);

        let pos = [
            r * theta.sin() * phi.cos(),
            r * theta.sin() * phi.sin(),
            r * theta.cos() + 2.5,
        ];
        let cov = [0.001, 0.0, 0.0, 0.001, 0.0, 0.001];
        let opacity = 0.85;
        let mut sh = [0.0f32; 48];
        sh[0] = 0.9;
        sh[1] = 0.6;
        sh[2] = 0.3;
        scene.push(pos, cov, opacity, &sh);
    }

    (scene, Vec3::new(0.0, 0.0, 2.5), 3.2)
}

fn compute_scene_bounds(scene: &GaussianSceneSoa) -> (Vec3, Vec3, Vec3, f32) {
    if scene.count == 0 {
        return (Vec3::splat(-0.5), Vec3::splat(0.5), Vec3::ZERO, 3.0);
    }

    let mut min_pos = Vec3::splat(f32::INFINITY);
    let mut max_pos = Vec3::splat(f32::NEG_INFINITY);

    for i in 0..scene.count {
        let p = Vec3::new(
            scene.positions[i * 3],
            scene.positions[i * 3 + 1],
            scene.positions[i * 3 + 2],
        );
        min_pos = min_pos.min(p);
        max_pos = max_pos.max(p);
    }

    let raw_extent = max_pos - min_pos;
    // If the scene has distant outliers (e.g. standard 3DGS 360 captures with background points)
    if raw_extent.length() > 20.0 {
        // Collect coordinates to find central percentiles
        let step = (scene.count / 20000).max(1);
        let mut xs = Vec::with_capacity(scene.count / step + 1);
        let mut ys = Vec::with_capacity(scene.count / step + 1);
        let mut zs = Vec::with_capacity(scene.count / step + 1);

        for i in (0..scene.count).step_by(step) {
            xs.push(scene.positions[i * 3]);
            ys.push(scene.positions[i * 3 + 1]);
            zs.push(scene.positions[i * 3 + 2]);
        }

        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
        zs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = xs.len();
        let idx_lo = (n as f32 * 0.15) as usize;
        let idx_hi = ((n as f32 * 0.85) as usize).min(n - 1);

        let p_min = Vec3::new(xs[idx_lo], ys[idx_lo], zs[idx_lo]);
        let p_max = Vec3::new(xs[idx_hi], ys[idx_hi], zs[idx_hi]);
        let center = (p_min + p_max) * 0.5;
        let extent = (p_max - p_min).max(Vec3::splat(0.5));
        let cage_min = center - 0.55 * extent;
        let cage_max = center + 0.55 * extent;
        let cam_dist = extent.length().max(2.5) * 1.1;

        (cage_min, cage_max, center, cam_dist)
    } else {
        let center = (min_pos + max_pos) * 0.5;
        let extent = raw_extent.max(Vec3::splat(0.05));
        let cage_min = center - 0.55 * extent;
        let cage_max = center + 0.55 * extent;
        let cam_dist = extent.length().max(1.0) * 1.5;

        (cage_min, cage_max, center, cam_dist)
    }
}

fn main() -> anyhow::Result<()> {
    println!("============================================================");
    println!("   CinderGS Autonomous Headless Demo Recorder");
    println!("   Hardware Target: NVIDIA GeForce RTX 3050 Laptop GPU");
    println!("============================================================");

    let output_dir = PathBuf::from("target/demo_frames");
    let docs_dir = PathBuf::from("docs");
    fs::create_dir_all(&output_dir)?;
    fs::create_dir_all(&docs_dir)?;

    let width = 960u32;
    let height = 540u32;
    const TOTAL_FRAMES: usize = 180;
    const FPS: u32 = 30;

    let ply_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("assets/bonsai.ply"));

    let (scene, cage_min, cage_max, cam_target, cam_dist) = if ply_path.exists() {
        println!("Loading canonical 3DGS asset from '{}'...", ply_path.display());
        let t0 = Instant::now();
        let mut loaded = PlyLoader::load_file(&ply_path)?;
        println!(
            "Successfully ingested {} Gaussians in {:.2} ms",
            loaded.len(),
            t0.elapsed().as_secs_f64() * 1000.0
        );
        if loaded.len() > 500_000 {
            // True centroid of the bonsai tree + pot in Mip-NeRF 360 dataset.
            // Raw Inria COLMAP coordinates have the dense flower cluster at (0.38, 0.88, 1.35).
            // Translate the entire scene rigidly so the bonsai tree is centered at (0.12, -0.10, 0.05),
            // tabletop at y ≈ 0.35, tablecloth at y ≈ 0.90, and floor at y ≈ 1.50.
            let shift = Vec3::new(0.12 - 0.38, -0.10 - 0.88, 0.05 - 1.35);
            for i in 0..loaded.count {
                loaded.positions[i * 3] += shift.x;
                loaded.positions[i * 3 + 1] += shift.y;
                loaded.positions[i * 3 + 2] += shift.z;
            }

            let bonsai_center = Vec3::new(0.12, -0.10, 0.05);
            let hero_min = Vec3::new(-0.35, -0.55, -0.40);
            let hero_max = Vec3::new(0.55, 0.35, 0.50);
            let dist = 2.7;
            (loaded, hero_min, hero_max, bonsai_center, dist)
        } else {
            let (c_min, c_max, target, dist) = compute_scene_bounds(&loaded);
            (loaded, c_min, c_max, target, dist)
        }
    } else {
        println!(
            "Asset '{}' not found. Falling back to synthetic 100k scene.",
            ply_path.display()
        );
        let (syn_scene, target, dist) = generate_synthetic_scene(100_000);
        let (c_min, c_max, _, _) = compute_scene_bounds(&syn_scene);
        (syn_scene, c_min, c_max, target, dist)
    };

    println!("Scene Target: {:?}, Camera Distance: {:.2}", cam_target, cam_dist);
    println!("Constructing 5-tet localized hero cage from {:?} to {:?}", cage_min, cage_max);
    let cage = TetMesh::create_box_cage(cage_min.to_array(), cage_max.to_array());

    println!("Binding {} Gaussians (hero localized with u32::MAX sentinel)...", scene.len());
    let t_bind = Instant::now();
    let bindings = cage.bind_flat_positions_with_bounds(&scene.positions, cage_min.to_array(), cage_max.to_array());
    let bound_count = bindings.iter().filter(|b| b.is_bound()).count();
    println!(
        "Bound {} / {} Gaussians to hero cage in {:.2} ms (unbound splats remain static and opaque)",
        bound_count,
        bindings.len(),
        t_bind.elapsed().as_secs_f64() * 1000.0
    );

    // Initialize CageSpringSimulator
    let mut simulator = CageSpringSimulator::new(cage.rest_vertices.clone());
    simulator.stiffness = 160.0;
    simulator.damping = 0.90;
    simulator.mass = 1.0;

    let rest_vertices_vec3: Vec<Vec3> = cage
        .rest_vertices
        .iter()
        .map(|&v| Vec3::from(v))
        .collect();

    // Find base vertices (y > 0.0, sitting on the table) and pin them permanently:
    let base_indices: Vec<usize> = rest_vertices_vec3
        .iter()
        .enumerate()
        .filter(|(_, v)| v.y > 0.0)
        .map(|(i, _)| i)
        .collect();

    // Top foliage handle (minimum Y, top of the tree foliage):
    let top_handle = rest_vertices_vec3
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.y.partial_cmp(&b.y).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    println!(
        "Selected top foliage handle: {} at {:?}, base vertices on table: {:?}",
        top_handle, rest_vertices_vec3[top_handle], base_indices
    );

    // Set up headless WGPU device and queue
    println!("Initializing headless WGPU device & queue...");
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("Failed to acquire GPU adapter");

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("DemoRecorderDevice"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        },
        None,
    ))
    .expect("Failed to acquire GPU device and queue");

    let target_format = wgpu::TextureFormat::Rgba8Unorm;
    let mut orchestrator = RenderPipelineOrchestrator::new(&device, target_format);
    orchestrator.resize(&device, width, height);
    orchestrator.backend = ExecutionBackend::CubeClGpu;

    // Offscreen render target & readback buffer
    let render_target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("DemoRenderTarget"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: target_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let surface_view = render_target.create_view(&wgpu::TextureViewDescriptor::default());

    let bytes_per_pixel = 4u32;
    let unpadded_bytes_per_row = width * bytes_per_pixel;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
    let buffer_size = (padded_bytes_per_row * height) as u64;

    let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("DemoReadbackBuffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut camera = OrbitCamera::new(cam_target, cam_dist);
    camera.pitch = -0.32;
    camera.yaw = -0.55;

    let compositor = CompositorUniforms::new(
        width,
        height,
        [0.05, 0.05, 0.08, 1.0], // Dark sleek studio backdrop
        [0.0, 0.0, 0.0],
        1,
    );

    println!(
        "\nChoreographing and rendering {} frames ({}x{} @ {} FPS)...",
        TOTAL_FRAMES, width, height, FPS
    );
    let record_start = Instant::now();
    let mut current_cage = rest_vertices_vec3.clone();

    for frame_idx in 0..TOTAL_FRAMES {
        let frame_start = Instant::now();

        // 1. Always update camera orbit every frame (smooth continuous orbit across all 180 frames):
        let progress = frame_idx as f32 / TOTAL_FRAMES as f32;
        camera.yaw = -0.55 + progress * 0.70;
        camera.pitch = -0.32;

        let view_matrix = camera.view_matrix();
        let cam_right = Vec3::new(view_matrix.x_axis.x, view_matrix.y_axis.x, view_matrix.z_axis.x).normalize();

        // 2. Trajectory Phases:
        let show_wireframe = true;
        let selected_vertex = if frame_idx < 45 {
            // Phase 1 (0.0s - 1.5s): Pure Orbit Showcase at rest
            current_cage.copy_from_slice(&rest_vertices_vec3);
            simulator.current_positions.copy_from_slice(&rest_vertices_vec3);
            simulator.velocities.fill(Vec3::ZERO);
            None
        } else if frame_idx < 90 {
            // Phase 2 (1.5s - 3.0s): Smooth foliage displacement
            let t = (frame_idx - 45) as f32 / 45.0;
            let bend_amount = (t * std::f32::consts::PI * 0.5).sin() * 0.18;
            let displacement = cam_right * bend_amount;

            current_cage.copy_from_slice(&rest_vertices_vec3);
            current_cage[top_handle] = rest_vertices_vec3[top_handle] + displacement;

            // SYNC SIMULATOR STATE:
            simulator.current_positions.copy_from_slice(&current_cage);
            simulator.velocities.fill(Vec3::ZERO);
            Some(top_handle)
        } else {
            // Phase 3 (3.0s - 6.0s): Vertex released -> PBD Spring Oscillation & Settling
            // Pin base vertices to table:
            for &base_idx in &base_indices {
                simulator.current_positions[base_idx] = rest_vertices_vec3[base_idx];
                simulator.velocities[base_idx] = Vec3::ZERO;
            }

            // Step physics simulation:
            simulator.step(1.0 / FPS as f32);

            // Re-pin base vertices to table:
            for &base_idx in &base_indices {
                simulator.current_positions[base_idx] = rest_vertices_vec3[base_idx];
                simulator.velocities[base_idx] = Vec3::ZERO;
            }

            // CRITICAL FIX: Copy simulator positions back into current_cage!
            current_cage.copy_from_slice(&simulator.current_positions);
            None
        };

        let deformed_cage_verts: Vec<[f32; 3]> = current_cage
            .iter()
            .map(|v| [v.x, v.y, v.z])
            .collect();
        let camera_uniforms = camera.build_camera_uniforms(width as f32, height as f32);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("DemoRecordEncoder"),
        });

        let stats = orchestrator.execute_frame(
            &device,
            &queue,
            &mut encoder,
            &scene,
            Some(&cage),
            Some(&bindings),
            Some(&deformed_cage_verts),
            &camera_uniforms,
            &compositor,
            &surface_view,
            show_wireframe,
            selected_vertex,
        );

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &render_target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &readback_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        queue.submit(Some(encoder.finish()));

        // Map readback buffer
        let slice = readback_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            sender.send(res).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .unwrap()
            .expect("Failed to map readback buffer");

        let mapped_range = slice.get_mapped_range();
        let mut rgba_bytes = vec![0u8; (width * height * 4) as usize];

        if padded_bytes_per_row == unpadded_bytes_per_row {
            rgba_bytes.copy_from_slice(&mapped_range);
        } else {
            for row in 0..height {
                let src_offset = (row * padded_bytes_per_row) as usize;
                let dst_offset = (row * unpadded_bytes_per_row) as usize;
                rgba_bytes[dst_offset..dst_offset + unpadded_bytes_per_row as usize]
                    .copy_from_slice(&mapped_range[src_offset..src_offset + unpadded_bytes_per_row as usize]);
            }
        }
        drop(mapped_range);
        readback_buffer.unmap();

        // Force full opacity on all pixels to eliminate transparent canvas voids in GIF viewers
        for i in 0..(width * height) as usize {
            rgba_bytes[i * 4 + 3] = 255;
        }

        let frame_path = output_dir.join(format!("frame_{:04}.png", frame_idx));
        image::save_buffer(
            &frame_path,
            &rgba_bytes,
            width,
            height,
            image::ExtendedColorType::Rgba8,
        )?;

        let elapsed_frame = frame_start.elapsed().as_secs_f64() * 1000.0;
        if frame_idx % 10 == 0 || frame_idx == TOTAL_FRAMES - 1 {
            println!(
                "Frame [{:3}/{}] rendered in {:6.2} ms | Surviving Splats: {:7} | Active Tiles: {:4}",
                frame_idx + 1,
                TOTAL_FRAMES,
                elapsed_frame,
                stats.rendered_gaussians,
                stats.active_tiles
            );
        }
    }

    let total_render_time = record_start.elapsed();
    println!(
        "\nFinished rendering {} frames in {:.2}s ({:.1} FPS average)",
        TOTAL_FRAMES,
        total_render_time.as_secs_f64(),
        TOTAL_FRAMES as f64 / total_render_time.as_secs_f64()
    );

    // 1. Encode 60 FPS MP4 Video:
    println!("\nCompiling MP4 artifact via ffmpeg: docs/cindergs_demo.mp4...");
    let mp4_output = docs_dir.join("cindergs_demo.mp4");
    let mp4_status = Command::new("ffmpeg")
        .args([
            "-y",
            "-framerate",
            "30",
            "-i",
            output_dir.join("frame_%04d.png").to_str().unwrap(),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            mp4_output.to_str().unwrap(),
        ])
        .status()?;

    if !mp4_status.success() {
        eprintln!("Warning: ffmpeg MP4 encoding returned status {:?}", mp4_status);
    } else {
        println!("Successfully generated {}", mp4_output.display());
    }

    // 2. Encode High-Fidelity GIF (format=rgb24 to guarantee solid dark background):
    println!("\nCompiling palette-optimized GIF artifact: docs/cindergs_demo.gif...");
    let gif_output = docs_dir.join("cindergs_demo.gif");
    let gif_status = Command::new("ffmpeg")
        .args([
            "-y",
            "-framerate",
            "30",
            "-i",
            output_dir.join("frame_%04d.png").to_str().unwrap(),
            "-vf",
            "fps=20,format=rgb24,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3",
            gif_output.to_str().unwrap(),
        ])
        .status()?;

    if !gif_status.success() {
        eprintln!("Warning: ffmpeg GIF encoding returned status {:?}", gif_status);
    } else {
        println!("Successfully generated {}", gif_output.display());
    }

    if let Ok(metadata) = fs::metadata(&gif_output) {
        let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
        println!("GIF File Size: {:.2} MB", size_mb);
    }

    println!("\nAutonomous demo recording completed successfully!");
    Ok(())
}
