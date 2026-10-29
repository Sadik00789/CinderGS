//! Autonomous headless demo recorder.
//!
//! Renders an orbit / deformation / spring-release sequence to PNG frames and,
//! when `ffmpeg` is installed, encodes MP4 and GIF artifacts.
//!
//! Framing defaults to the outlier-robust auto-framer. A specific capture can be
//! framed explicitly without re-introducing scene constants into the code:
//!
//! ```text
//! record_demo --input assets/bonsai.ply \
//!     --look-at 0.38,0.88,1.35 --distance 2.7 \
//!     --bounds -0.09,0.43,0.90,0.81,1.33,1.80
//! ```

use anyhow::{Result, anyhow};
use cinder_gs::deformation::{Cage, CageScene, TetMesh};
use cinder_gs::render::camera::CameraUniforms;
use cinder_gs::render::headless::HeadlessRenderer;
use cinder_gs::render::pipeline::ExecutionBackend;
use cinder_gs::render::rasterizer::CompositorUniforms;
use cinder_gs::scene::autoframe::compute_scene_framing;
use cinder_gs::scene::fibonacci_cluster;
use cinder_gs::scene::{GaussianSceneSoa, PlyLoader};
use glam::Vec3;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

const DEFAULT_WIDTH: u32 = 960;
const DEFAULT_HEIGHT: u32 = 540;
const DEFAULT_FRAMES: u32 = 180;
const DEFAULT_FPS: u32 = 30;

struct DemoOptions {
    input: Option<PathBuf>,
    output: PathBuf,
    look_at: Option<[f32; 3]>,
    distance: Option<f32>,
    bounds: Option<([f32; 3], [f32; 3])>,
    width: u32,
    height: u32,
    frames: u32,
    fps: u32,
    no_encode: bool,
}

impl Default for DemoOptions {
    fn default() -> Self {
        Self {
            input: None,
            output: PathBuf::from("target/demo_frames"),
            look_at: None,
            distance: None,
            bounds: None,
            width: DEFAULT_WIDTH,
            height: DEFAULT_HEIGHT,
            frames: DEFAULT_FRAMES,
            fps: DEFAULT_FPS,
            no_encode: false,
        }
    }
}

fn parse_vec3(value: &str) -> Result<[f32; 3]> {
    let parts: Vec<f32> = value
        .split(',')
        .map(|part| part.trim().parse::<f32>())
        .collect::<std::result::Result<_, _>>()
        .map_err(|err| anyhow!("invalid vector '{value}': {err}"))?;
    if parts.len() != 3 {
        return Err(anyhow!("expected 3 comma-separated floats, got '{value}'"));
    }
    Ok([parts[0], parts[1], parts[2]])
}

fn parse_bounds(value: &str) -> Result<([f32; 3], [f32; 3])> {
    let parts: Vec<f32> = value
        .split(',')
        .map(|part| part.trim().parse::<f32>())
        .collect::<std::result::Result<_, _>>()
        .map_err(|err| anyhow!("invalid bounds '{value}': {err}"))?;
    if parts.len() != 6 {
        return Err(anyhow!(
            "expected 6 comma-separated floats (minx,miny,minz,maxx,maxy,maxz), got '{value}'"
        ));
    }
    Ok((
        [parts[0], parts[1], parts[2]],
        [parts[3], parts[4], parts[5]],
    ))
}

fn parse_options() -> Result<DemoOptions> {
    let mut options = DemoOptions::default();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut index = 0;

    while index < args.len() {
        let argument = args[index].as_str();
        let mut take_value = |name: &str| -> Result<String> {
            index += 1;
            args.get(index)
                .cloned()
                .ok_or_else(|| anyhow!("{name} requires a value"))
        };

        match argument {
            "--input" | "-i" => options.input = Some(PathBuf::from(take_value("--input")?)),
            "--out" | "-o" => options.output = PathBuf::from(take_value("--out")?),
            "--look-at" => options.look_at = Some(parse_vec3(&take_value("--look-at")?)?),
            "--distance" => options.distance = Some(take_value("--distance")?.parse()?),
            "--bounds" => options.bounds = Some(parse_bounds(&take_value("--bounds")?)?),
            "--width" => options.width = take_value("--width")?.parse()?,
            "--height" => options.height = take_value("--height")?.parse()?,
            "--frames" => options.frames = take_value("--frames")?.parse()?,
            "--fps" => options.fps = take_value("--fps")?.parse()?,
            "--no-encode" => options.no_encode = true,
            "--help" | "-h" => {
                println!(
                    "CinderGS demo recorder\n\
                     --input <scene.ply>       input PLY (synthetic scene when omitted)\n\
                     --out <dir>               frame directory\n\
                     --look-at x,y,z           cage/camera centre override\n\
                     --distance <f>            orbit distance override\n\
                     --bounds minx,miny,minz,maxx,maxy,maxz  cage bounds override\n\
                     --width/--height <px>     output resolution\n\
                     --frames <n> --fps <n>    timing\n\
                     --no-encode               keep PNGs only, skip MP4/GIF"
                );
                std::process::exit(0);
            }
            other if !other.starts_with("--") => options.input = Some(PathBuf::from(other)),
            other => return Err(anyhow!("unknown option '{other}'")),
        }
        index += 1;
    }

    if options.width == 0 || options.height == 0 || options.frames == 0 || options.fps == 0 {
        return Err(anyhow!("width, height, frames and fps must be non-zero"));
    }

    Ok(options)
}

fn load_scene(input: &Option<PathBuf>) -> Result<GaussianSceneSoa> {
    if let Some(path) = input {
        if path.exists() {
            println!("Loading '{}'...", path.display());
            let timer = Instant::now();
            let scene = PlyLoader::load_file(path)?;
            println!(
                "Loaded {} Gaussians in {:.2} ms",
                scene.len(),
                timer.elapsed().as_secs_f64() * 1000.0
            );
            return Ok(scene);
        }
        eprintln!(
            "warning: '{}' not found; using a synthetic scene",
            path.display()
        );
    }
    Ok(fibonacci_cluster(100_000, 1.2, [0.0, 0.0, 2.5]))
}

fn encode_with_ffmpeg(frames: &std::path::Path, docs: &std::path::Path, fps: u32) -> Result<()> {
    let available = Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);

    if !available {
        println!(
            "ffmpeg not found; PNG frames remain in {}",
            frames.display()
        );
        return Ok(());
    }

    let pattern = frames.join("frame_%04d.png");
    let pattern = pattern
        .to_str()
        .ok_or_else(|| anyhow!("frame path is not valid UTF-8"))?;
    let rate = fps.to_string();

    let mp4 = docs.join("cindergs_demo.mp4");
    let mp4_status = Command::new("ffmpeg")
        .args([
            "-y",
            "-framerate",
            &rate,
            "-i",
            pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
        ])
        .arg(&mp4)
        .status();

    match mp4_status {
        Ok(status) if status.success() => println!("Wrote {}", mp4.display()),
        Ok(status) => eprintln!("warning: ffmpeg MP4 encoding failed ({status})"),
        Err(err) => eprintln!("warning: could not run ffmpeg: {err}"),
    }

    let gif = docs.join("cindergs_demo.gif");
    let gif_status = Command::new("ffmpeg")
        .args([
            "-y",
            "-framerate",
            &rate,
            "-i",
            pattern,
            "-vf",
            "fps=20,format=rgb24,scale=800:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3",
        ])
        .arg(&gif)
        .status();

    match gif_status {
        Ok(status) if status.success() => {
            if let Ok(metadata) = fs::metadata(&gif) {
                println!(
                    "Wrote {} ({:.2} MB)",
                    gif.display(),
                    metadata.len() as f64 / (1024.0 * 1024.0)
                );
            }
        }
        Ok(status) => eprintln!("warning: ffmpeg GIF encoding failed ({status})"),
        Err(err) => eprintln!("warning: could not run ffmpeg: {err}"),
    }

    Ok(())
}

fn main() -> Result<()> {
    println!("============================================================");
    println!("   CinderGS Headless Demo Recorder");
    println!("============================================================");

    let options = parse_options()?;
    let docs_dir = PathBuf::from("docs");
    fs::create_dir_all(&options.output)?;
    fs::create_dir_all(&docs_dir)?;

    let scene = load_scene(&options.input)?;
    if scene.count == 0 {
        return Err(anyhow!("scene contains no Gaussians"));
    }

    let framing = compute_scene_framing(&scene);
    let cage_min = options.bounds.map(|b| b.0).unwrap_or(framing.hero_min);
    let cage_max = options.bounds.map(|b| b.1).unwrap_or(framing.hero_max);
    let look_at = options.look_at.unwrap_or(framing.center);
    let distance = options.distance.unwrap_or(framing.camera_distance);

    let cage_mesh = TetMesh::create_box_cage(cage_min, cage_max);
    let mut cages = CageScene::auto_bind(
        vec![Cage::new("hero", cage_mesh, [0.0, 0.85, 1.0, 0.45])],
        &scene.positions,
    );
    println!(
        "Framing: centre {look_at:?}, distance {distance:.2}, cage {cage_min:?}..{cage_max:?}"
    );
    println!("Bound {}/{} Gaussians", cages.bound_count(), scene.len());

    let rest_vertices: Vec<Vec3> = cages
        .cages
        .first()
        .map(|cage| cage.simulator.rest_positions.to_vec())
        .unwrap_or_default();
    if rest_vertices.is_empty() {
        return Err(anyhow!("cage has no vertices"));
    }

    // Choose top/base by the image-space up axis of the initial camera, so the
    // deformation always bends the crown regardless of the capture's world
    // orientation.
    let camera_up = {
        let view = cinder_gs::app::OrbitCamera::new(Vec3::from(look_at), distance).view_matrix();
        Vec3::new(view.x_axis.y, view.y_axis.y, view.z_axis.y).normalize_or_zero()
    };
    // The splat rasterizer maps camera-space +Y to increasing framebuffer rows
    // (y-down), so the visual top of the frame is the *negative* camera-up axis.
    // Select deformation handles in that image-space convention.
    let image_up = -camera_up;
    let up_scores: Vec<f32> = rest_vertices
        .iter()
        .map(|vertex| vertex.dot(image_up))
        .collect();
    let highest = up_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lowest = up_scores.iter().copied().fold(f32::INFINITY, f32::min);
    // The camera up axis is tilted, so a strict "equals the max" test would pick
    // only the back edge of the cage's top face. A fraction of the vertical span
    // captures the whole face.
    let tolerance = 0.4 * (highest - lowest).max(1e-6);
    let top_indices: Vec<usize> = up_scores
        .iter()
        .enumerate()
        .filter(|(_, score)| **score >= highest - tolerance)
        .map(|(index, _)| index)
        .collect();
    let top_handle = *top_indices
        .first()
        .ok_or_else(|| anyhow!("cage has no top vertex"))?;
    let base_indices: Vec<usize> = up_scores
        .iter()
        .enumerate()
        .filter(|(_, score)| **score <= lowest + tolerance)
        .map(|(index, _)| index)
        .collect();
    let top_center = {
        let mut center = Vec3::ZERO;
        for &index in &top_indices {
            center += rest_vertices[index];
        }
        center / top_indices.len().max(1) as f32
    };

    println!(
        "Base ring (pinned): {:?}, top ring (moving): {:?} (image up {image_up:?}, range {:.2}..{:.2})",
        base_indices, top_indices, lowest, highest
    );

    let mut renderer =
        HeadlessRenderer::new(options.width, options.height, ExecutionBackend::CubeClGpu)?;
    if let Some(description) = renderer.gpu_description() {
        println!("Compute device: {description}");
    }

    let compositor = CompositorUniforms::new(
        options.width,
        options.height,
        [0.05, 0.05, 0.08, 1.0],
        [0.0, 0.0, 0.0],
        3,
    );

    let mut camera = cinder_gs::app::OrbitCamera::new(Vec3::from(look_at), distance);
    // Gentle low-angle hero orbit, matching the original demo framing.
    camera.pitch = -0.32;

    println!(
        "Rendering {} frames at {}x{} @ {} FPS...",
        options.frames, options.width, options.height, options.fps
    );
    let record_start = Instant::now();

    for frame_index in 0..options.frames {
        let frame_start = Instant::now();
        let progress = frame_index as f32 / options.frames as f32;
        camera.yaw = -0.55 + progress * 0.70;

        let view = camera.view_matrix();
        let cam_right = Vec3::new(view.x_axis.x, view.y_axis.x, view.z_axis.x).normalize_or_zero();

        let mut selected = None;

        match frame_index {
            0..=44 => {
                // Phase 1: orbit showcase at rest.
                if let Some(cage) = cages.cages.first_mut() {
                    cage.simulator.reset();
                }
            }
            45..=89 => {
                // Phase 2: the whole crown leans and puffs slightly. Moving the
                // full top ring keeps the deformation coherent across tetrahedra,
                // so the tree bends instead of splitting at a seam.
                let t = (frame_index - 45) as f32 / 45.0;
                let lean = (t * std::f32::consts::FRAC_PI_2).sin() * 0.10;
                let bloom = (t * std::f32::consts::FRAC_PI_2).sin() * 0.03;
                if let Some(cage) = cages.cages.first_mut() {
                    cage.simulator.reset();
                    for &top in &top_indices {
                        let radial = cage.simulator.rest_positions[top] - top_center;
                        cage.simulator.current_positions[top] =
                            cage.simulator.rest_positions[top] + cam_right * lean + radial * bloom;
                    }
                }
                selected = Some((0, top_handle));
            }
            _ => {
                // Phase 3: release with a gentle outward impulse so the foliage
                // ruffles and settles instead of snapping back as a rigid block.
                if frame_index == 90
                    && let Some(cage) = cages.cages.first_mut()
                {
                    for &top in &top_indices {
                        let radial = cage.simulator.rest_positions[top] - top_center;
                        cage.simulator.current_positions[top] =
                            cage.simulator.rest_positions[top] + cam_right * 0.10 + radial * 0.03;
                        cage.simulator.velocities[top] = cam_right * 1.0 + radial * 1.5;
                    }
                    for &base in &base_indices {
                        cage.simulator.current_positions[base] =
                            cage.simulator.rest_positions[base];
                        cage.simulator.velocities[base] = Vec3::ZERO;
                    }
                }

                cages.step(1.0 / options.fps as f32);

                if let Some(cage) = cages.cages.first_mut() {
                    for &base in &base_indices {
                        cage.simulator.current_positions[base] =
                            cage.simulator.rest_positions[base];
                        cage.simulator.velocities[base] = Vec3::ZERO;
                    }
                }
            }
        }

        let camera_uniforms = CameraUniforms::from_look_at(
            camera.eye_position(),
            camera.target,
            Vec3::Y,
            camera.fov_y,
            options.width as f32,
            options.height as f32,
            camera.near_plane,
        );

        let result = renderer.render(
            &scene,
            Some(&cages),
            &camera_uniforms,
            &compositor,
            true,
            selected,
        )?;

        let frame_path = options.output.join(format!("frame_{frame_index:04}.png"));
        image::save_buffer(
            &frame_path,
            &result.image.rgba8,
            result.image.width,
            result.image.height,
            image::ExtendedColorType::Rgba8,
        )?;

        if frame_index % 10 == 0 || frame_index + 1 == options.frames {
            println!(
                "Frame [{:3}/{}] {:6.2} ms | splats {} | tiles {} | {}",
                frame_index + 1,
                options.frames,
                frame_start.elapsed().as_secs_f64() * 1000.0,
                result.stats.rendered_gaussians,
                result.stats.active_tiles,
                if result.stats.sort_reused {
                    "sort reused"
                } else {
                    "sort recomputed"
                }
            );
        }
    }

    let elapsed = record_start.elapsed();
    println!(
        "\nRendered {} frames in {:.2}s ({:.2} ms/frame)",
        options.frames,
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1000.0 / options.frames as f64
    );

    if !options.no_encode {
        encode_with_ffmpeg(&options.output, &docs_dir, options.fps)?;
    }
    println!("\nDemo recording complete.");
    Ok(())
}
