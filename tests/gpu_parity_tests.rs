//! GPU/CPU kernel parity and golden-image tests.
//!
//! These tests require a working WGPU adapter; when none is available they print
//! a skip notice and return successfully so headless CI without a GPU still runs
//! the rest of the suite.

use cinder_gs::deformation::{DeformationUniforms, GaussianBinding, TetMesh, deform_scene_cpu};
use cinder_gs::render::camera::CameraUniforms;
use cinder_gs::render::gpu::GpuCompute;
use cinder_gs::render::gpu_bridge::{composite_gpu, deform_geometry_gpu, project_gaussians_gpu};
use cinder_gs::render::rasterizer::{CompositorUniforms, rasterize_scene_cpu};
use cinder_gs::render::{project_scene_cpu, render_scene_cpu};
use cinder_gs::scene::GaussianSceneSoa;
use cinder_gs::scene::fibonacci_cluster;
use glam::{Mat4, Vec3};

/// CubeCL registers devices in a process-global registry. Bring up exactly one
/// client for the whole test binary and serialise access to it.
static GPU_SESSION: std::sync::OnceLock<std::sync::Mutex<Option<GpuCompute>>> =
    std::sync::OnceLock::new();

struct GpuGuard {
    session: std::sync::MutexGuard<'static, Option<GpuCompute>>,
}

impl GpuGuard {
    fn gpu(&self) -> &GpuCompute {
        self.session
            .as_ref()
            .expect("GpuGuard is only constructed for a live session")
    }
}

fn gpu_session() -> Option<GpuGuard> {
    let session = GPU_SESSION.get_or_init(|| match GpuCompute::acquire() {
        Ok(gpu) => std::sync::Mutex::new(Some(gpu)),
        Err(error) => {
            eprintln!("GPU unavailable, skipping: {error}");
            std::sync::Mutex::new(None)
        }
    });

    let guard = session
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if guard.is_none() {
        return None;
    }
    Some(GpuGuard { session: guard })
}

fn test_camera(width: f32, height: f32) -> CameraUniforms {
    CameraUniforms::from_look_at(
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(0.0, 0.0, 2.0),
        Vec3::Y,
        50.0f32.to_radians(),
        width,
        height,
        0.2,
    )
}

fn varied_scene() -> GaussianSceneSoa {
    let mut scene = GaussianSceneSoa::with_capacity(48);
    let golden = (1.0 + 5.0f32.sqrt()) / 2.0;

    for i in 0..48 {
        let t = (i as f32 + 0.5) / 48.0;
        let theta = (1.0 - 2.0 * t).clamp(-1.0, 1.0).acos();
        let phi = 2.0 * std::f32::consts::PI * (i as f32 / golden);
        let radius = 0.6;

        let position = [
            radius * theta.sin() * phi.cos(),
            radius * theta.sin() * phi.sin(),
            radius * theta.cos() + 2.0,
        ];

        // Anisotropic and rotated covariances to exercise the EWA path.
        let sx = 0.02 + 0.01 * (i as f32 * 0.3).sin().abs();
        let sy = 0.03 + 0.01 * (i as f32 * 0.7).cos().abs();
        let sz = 0.015 + 0.005 * (i as f32 * 0.11).sin().abs();
        let covariance = [sx * sx, 0.0004, 0.0002, sy * sy, 0.0003, sz * sz];

        let opacity = 0.35 + 0.5 * ((i as f32 * 0.27).sin() * 0.5 + 0.5);

        let mut sh = [0.0f32; 48];
        for (k, value) in sh.iter_mut().enumerate() {
            *value = ((i * 17 + k * 31) as f32 * 0.013).sin() * 0.4;
        }
        sh[0] = 1.2;
        sh[1] = 0.4;

        scene.push(position, covariance, opacity, &sh);
    }

    scene
}

#[test]
fn test_projection_kernel_matches_cpu_reference() {
    let Some(guard) = gpu_session() else {
        return;
    };
    let gpu = guard.gpu();

    let scene = varied_scene();
    let camera = test_camera(320.0, 240.0);

    let cpu = project_scene_cpu(&scene, &camera);
    let gpu_output = project_gaussians_gpu(
        gpu.client(),
        &scene.positions,
        &scene.covariances_3d,
        &camera,
    )
    .expect("GPU projection must succeed");

    assert_eq!(gpu_output.count, cpu.count);

    for i in 0..cpu.count {
        assert!(
            (gpu_output.depths[i] - cpu.depths[i]).abs() < 1e-4,
            "depth mismatch at {i}"
        );
        assert_eq!(
            gpu_output.radii[i] > 0,
            cpu.radii[i] > 0,
            "cull decision mismatch at {i}: gpu radius {} cpu radius {}",
            gpu_output.radii[i],
            cpu.radii[i]
        );

        if cpu.radii[i] > 0 {
            assert!(
                (gpu_output.radii[i] - cpu.radii[i]).abs() <= 1,
                "radius mismatch at {i}: gpu {} cpu {}",
                gpu_output.radii[i],
                cpu.radii[i]
            );

            for axis in 0..2 {
                let a = gpu_output.points_xy[i * 2 + axis];
                let b = cpu.points_xy[i * 2 + axis];
                assert!(
                    (a - b).abs() < 2e-2,
                    "center mismatch at {i}.{axis}: {a} vs {b}"
                );
            }

            for coefficient in 0..3 {
                let a = gpu_output.conics[i * 3 + coefficient];
                let b = cpu.conics[i * 3 + coefficient];
                let relative = (a - b).abs() / b.abs().max(1e-6);
                assert!(
                    relative < 2e-2,
                    "conic mismatch at {i}.{coefficient}: {a} vs {b}"
                );
            }
        }
    }
}

#[test]
fn test_deformation_kernel_matches_cpu_reference() {
    let Some(guard) = gpu_session() else {
        return;
    };
    let gpu = guard.gpu();

    let scene = varied_scene();
    let mesh = TetMesh::create_box_cage([-1.0, -1.0, 1.0], [1.0, 1.0, 3.0]);
    let positions: Vec<[f32; 3]> = (0..scene.count)
        .map(|i| {
            [
                scene.positions[i * 3],
                scene.positions[i * 3 + 1],
                scene.positions[i * 3 + 2],
            ]
        })
        .collect();
    let bindings: Vec<GaussianBinding> = mesh.bind_gaussians(&positions);

    let mut deformed_vertices = mesh.rest_vertices.clone();
    deformed_vertices[0][0] += 0.15;
    deformed_vertices[2][1] -= 0.1;
    deformed_vertices[5][2] += 0.05;

    let uniforms =
        DeformationUniforms::new([0.0; 3], 0, scene.count as u32, mesh.num_tets() as u32);
    let cpu = deform_scene_cpu(&scene, &mesh, &bindings, &deformed_vertices, &uniforms);
    let gpu_output = deform_geometry_gpu(
        gpu.client(),
        &scene.positions,
        &scene.covariances_3d,
        &scene.opacities,
        &mesh,
        &bindings,
        &deformed_vertices,
    )
    .expect("GPU deformation must succeed");

    assert_eq!(gpu_output.positions.len(), cpu.scene.positions.len());

    for i in 0..scene.count {
        for axis in 0..3 {
            let a = gpu_output.positions[i * 3 + axis];
            let b = cpu.scene.positions[i * 3 + axis];
            assert!(
                (a - b).abs() < 2e-3,
                "position mismatch at {i}.{axis}: {a} vs {b}"
            );
        }
        for component in 0..6 {
            let a = gpu_output.covariances[i * 6 + component];
            let b = cpu.scene.covariances_3d[i * 6 + component];
            let relative = (a - b).abs() / b.abs().max(1e-5);
            assert!(
                relative < 5e-2,
                "covariance mismatch at {i}.{component}: {a} vs {b}"
            );
        }
        let a = gpu_output.opacities[i];
        let b = cpu.scene.opacities[i];
        assert!((a - b).abs() < 5e-3, "opacity mismatch at {i}: {a} vs {b}");
    }
}

#[test]
fn test_compositor_kernel_matches_cpu_reference() {
    let Some(guard) = gpu_session() else {
        return;
    };
    let gpu = guard.gpu();

    let scene = varied_scene();
    let camera = test_camera(96.0, 96.0);
    let projection = project_scene_cpu(&scene, &camera);
    let binning = projection.bin_and_sort(96, 96);
    let compositor = CompositorUniforms::new(
        96,
        96,
        [0.02, 0.03, 0.05, 1.0],
        camera.view_matrix[12..15].try_into().unwrap_or([0.0; 3]),
        3,
    );

    let cpu = rasterize_scene_cpu(
        &projection.points_xy,
        &projection.conics,
        &scene.opacities,
        &scene.sh_coeffs,
        &scene.positions,
        &binning.sorted_entries,
        &binning.tile_ranges,
        &compositor,
    );

    let gpu_image = composite_gpu(
        gpu.client(),
        &projection,
        &scene.opacities,
        &scene.sh_coeffs,
        &scene.positions,
        &binning,
        &compositor,
    )
    .expect("GPU compositor must succeed");

    assert_eq!(cpu.rgba_f32.len(), gpu_image.rgba_f32.len());

    let mut max_difference = 0.0f32;
    let mut sum_difference = 0.0f64;
    for (a, b) in cpu.rgba_f32.iter().zip(gpu_image.rgba_f32.iter()) {
        let difference = (a - b).abs();
        max_difference = max_difference.max(difference);
        sum_difference += f64::from(difference);
    }
    let mean_difference = sum_difference / cpu.rgba_f32.len() as f64;

    assert!(
        mean_difference < 5e-3,
        "mean pixel difference too large: {mean_difference}"
    );
    assert!(
        max_difference < 5e-2,
        "max pixel difference too large: {max_difference}"
    );
}

/// Golden image: a deterministic synthetic frame must not change silently.
#[test]
fn test_golden_image_cpu_render_is_stable() {
    let scene = fibonacci_cluster(256, 0.5, [0.0, 0.0, 2.0]);
    let camera = CameraUniforms::new(Mat4::IDENTITY, 180.0, 180.0, 80.0, 60.0, 160.0, 120.0, 0.2);
    let compositor = CompositorUniforms::new(160, 120, [0.05, 0.05, 0.07, 1.0], [0.0, 0.0, 0.0], 2);

    let image = render_scene_cpu(&scene, &camera, &compositor);
    let hash = image.pixel_hash();

    // Re-rendering is deterministic.
    let again = render_scene_cpu(&scene, &camera, &compositor);
    assert_eq!(again.pixel_hash(), hash);

    // Golden value captured from the CPU reference implementation. Updating this
    // constant is a deliberate visual change and must be reviewed as such.
    assert_eq!(
        hash, 0xbdd6_3815_3115_0f06,
        "golden image hash changed; got {hash:#018x}"
    );
}
