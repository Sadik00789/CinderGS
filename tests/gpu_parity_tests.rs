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

#[test]
fn test_gpu_deform_with_splitting_matches_cpu_strain_detection() {
    use cinder_gs::deformation::deform_scene_adaptive_cpu;
    use cinder_gs::render::gpu_bridge::deform_geometry_with_splitting_gpu;

    let Some(guard) = gpu_session() else {
        return;
    };
    let gpu = guard.gpu();

    let mut scene = GaussianSceneSoa::with_capacity(2);
    scene.push(
        [0.4, 0.5, 0.5],
        [0.02, 0.0, 0.0, 0.02, 0.0, 0.02],
        0.9,
        &[0.5; 48],
    );
    scene.push(
        [0.6, 0.5, 0.5],
        [0.02, 0.0, 0.0, 0.02, 0.0, 0.02],
        0.85,
        &[0.4; 48],
    );

    let mesh = TetMesh::create_box_cage([0.0; 3], [1.0; 3]);
    let bindings = mesh.bind_flat_positions(&scene.positions);
    let mut deformed = mesh.rest_vertices.clone();
    for v in &mut deformed {
        v[0] *= 2.4;
    }

    let gpu_out = deform_geometry_with_splitting_gpu(
        gpu.client(),
        &scene.positions,
        &scene.covariances_3d,
        &scene.opacities,
        &mesh,
        &bindings,
        &deformed,
        1.75,
        0.001,
    )
    .expect("GPU adaptive splitting must succeed");

    let uniforms = DeformationUniforms::new([0.0; 3], 0, 2, mesh.num_tets() as u32)
        .with_strain_splitting(1.75)
        .with_mip_filter_3d(0.001);
    let cpu_out = deform_scene_adaptive_cpu(&scene, &mesh, &bindings, &deformed, &uniforms);

    assert_eq!(gpu_out.split_count, 2);
    assert_eq!(gpu_out.positions.len(), cpu_out.scene.positions.len());
    for (g, c) in gpu_out.positions.iter().zip(cpu_out.scene.positions.iter()) {
        assert!((g - c).abs() < 2e-2, "split position mismatch: {g} vs {c}");
    }
    for (g, c) in gpu_out.opacities.iter().zip(cpu_out.scene.opacities.iter()) {
        assert!((g - c).abs() < 2e-2, "split opacity mismatch: {g} vs {c}");
    }
}

#[test]
fn test_gpu_cull_projection_and_indirect_dispatch_match_cpu() {
    use cinder_gs::math::projection::ProjectionOptions;
    use cinder_gs::render::gpu_bridge::project_gaussians_cull_gpu;
    use cinder_gs::render::project_scene_cull_cpu;

    let Some(guard) = gpu_session() else {
        return;
    };
    let gpu = guard.gpu();

    let mut scene = varied_scene();
    // Append one behind-camera outlier and one ultra-transparent outlier.
    scene.push(
        [0.0, 0.0, -3.0],
        [0.05, 0.0, 0.0, 0.05, 0.0, 0.05],
        0.9,
        &[0.4; 48],
    );
    scene.push(
        [0.0, 0.0, 2.0],
        [0.05, 0.0, 0.0, 0.05, 0.0, 0.05],
        0.001,
        &[0.4; 48],
    );

    let camera = test_camera(96.0, 96.0);
    let opts = ProjectionOptions {
        mip_filter_3d_var: 0.001,
        min_opacity: 0.01,
        min_radius_px: 0.5,
        ..ProjectionOptions::default()
    };

    let cpu_out = project_scene_cull_cpu(&scene, &camera, &opts);
    let gpu_out = project_gaussians_cull_gpu(
        gpu.client(),
        &scene.positions,
        &scene.covariances_3d,
        &scene.opacities,
        &camera,
        opts.mip_filter_3d_var,
        opts.min_opacity,
        opts.min_radius_px,
    )
    .expect("GPU culling projection must succeed");

    assert_eq!(
        gpu_out.surviving_count,
        cpu_out.projection.num_surviving() as u32
    );
    assert_eq!(gpu_out.indirect_dispatch, cpu_out.indirect_dispatch);
    assert_eq!(gpu_out.visibility_flags, cpu_out.visibility_flags);
    for (i, &vis) in gpu_out.visibility_flags.iter().enumerate() {
        if vis != 0 {
            assert!((gpu_out.projection.depths[i] - cpu_out.projection.depths[i]).abs() < 1e-3);
            assert!((gpu_out.opacities[i] - cpu_out.opacities[i]).abs() < 5e-3);
        }
    }
}

#[test]
fn test_gpu_quantized_projection_and_compositor_match_cpu() {
    use cinder_gs::math::projection::ProjectionOptions;
    use cinder_gs::render::gpu_bridge::{composite_quantized_gpu, project_gaussians_quantized_gpu};
    use cinder_gs::render::project_scene_cull_cpu;

    let Some(guard) = gpu_session() else {
        return;
    };
    let gpu = guard.gpu();

    let scene = varied_scene();
    let quantized = scene.quantize();
    let camera = test_camera(96.0, 96.0);
    let opts = ProjectionOptions {
        mip_filter_3d_var: 0.001,
        min_opacity: 0.005,
        min_radius_px: 0.5,
        ..ProjectionOptions::default()
    };

    let cpu_proj = project_scene_cull_cpu(&scene, &camera, &opts);
    let gpu_proj = project_gaussians_quantized_gpu(
        gpu.client(),
        &quantized,
        &camera,
        opts.mip_filter_3d_var,
        opts.min_opacity,
        opts.min_radius_px,
    )
    .expect("GPU quantized projection must succeed");
    assert_eq!(
        gpu_proj.surviving_count,
        cpu_proj.projection.num_surviving() as u32
    );

    let binning = gpu_proj.projection.bin_and_sort_active(96, 96);
    let compositor = CompositorUniforms::new(96, 96, [0.02, 0.03, 0.05, 1.0], [0.0, 0.0, -1.0], 3);

    let cpu_binning = cpu_proj.projection.bin_and_sort_active(96, 96);
    let cpu_image = rasterize_scene_cpu(
        &cpu_proj.projection.points_xy,
        &cpu_proj.projection.conics,
        &cpu_proj.opacities,
        &scene.sh_coeffs,
        &scene.positions,
        &cpu_binning.sorted_entries,
        &cpu_binning.tile_ranges,
        &compositor,
    );
    let gpu_image = composite_quantized_gpu(
        gpu.client(),
        &gpu_proj.projection,
        &gpu_proj.opacities,
        &quantized,
        &binning,
        &compositor,
    )
    .expect("GPU quantized compositor must succeed");

    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f64;
    for (a, b) in cpu_image.rgba_f32.iter().zip(gpu_image.rgba_f32.iter()) {
        let d = (a - b).abs();
        max_diff = max_diff.max(d);
        sum_diff += f64::from(d);
    }
    let mean_diff = sum_diff / cpu_image.rgba_f32.len() as f64;
    assert!(
        mean_diff < 2e-2,
        "quantized GPU/CPU mean pixel error too large: {mean_diff}"
    );
    assert!(
        max_diff < 0.1,
        "quantized GPU/CPU max pixel error too large: {max_diff}"
    );
}

#[test]
fn test_gpu_exclusive_scan_matches_cpu() {
    use cinder_gs::render::kernels::{ScanScratch, gpu_exclusive_scan_total};

    let Some(guard) = gpu_session() else {
        return;
    };
    let client = guard.gpu().client();
    let mut scratch = ScanScratch::new();

    for &n in &[0usize, 1, 17, 512, 513, 100_000, 300_000] {
        let input: Vec<u32> = (0..n).map(|i| ((i * 37 + 11) % 7) as u32).collect();
        let mut expected = Vec::with_capacity(n);
        let mut acc = 0u32;
        for &v in &input {
            expected.push(acc);
            acc += v;
        }

        let in_handle = if n == 0 {
            client.create_from_slice(bytemuck::bytes_of(&0u32))
        } else {
            client.create_from_slice(bytemuck::cast_slice(&input))
        };
        let out_handle = client.empty((n * 4).max(4));

        let total = gpu_exclusive_scan_total(client, &in_handle, &out_handle, n, &mut scratch)
            .expect("GPU exclusive scan must succeed");
        assert_eq!(total, acc, "total mismatch for n={n}");

        if n > 0 {
            let bytes = client.read_one(out_handle).expect("read scan output");
            let gpu_out: &[u32] = bytemuck::cast_slice(&bytes[..n * 4]);
            assert_eq!(
                gpu_out,
                expected.as_slice(),
                "prefix scan mismatch at n={n}"
            );
        }
    }
}

#[test]
fn test_gpu_radix_sort_pairs_matches_cpu() {
    use cinder_gs::render::kernels::{
        RadixSortScratch, gpu_radix_sort_u32_pairs, gpu_radix_sort_u64_pairs,
    };

    let Some(guard) = gpu_session() else {
        return;
    };
    let gpu = guard.gpu();
    let client = gpu.client();
    let mut scratch = RadixSortScratch::new();

    let mut modes = vec![false];
    if gpu.supports_subgroups() {
        modes.push(true);
    }

    for use_subgroups in modes {
        for &n in &[0usize, 1, 317, 50_000] {
            let keys: Vec<u32> = (0..n)
                .map(|i| (((i as u64).wrapping_mul(2654435761u64) >> 16) as u32) & 0x00FF_0FFF)
                .collect();
            let vals: Vec<u32> = (0..n as u32).collect();

            let mut expected: Vec<(u32, u32)> =
                keys.iter().copied().zip(vals.iter().copied()).collect();
            expected.sort_by_key(|&(k, v)| (k, v));

            let k_h = if n == 0 {
                client.create_from_slice(bytemuck::bytes_of(&0u32))
            } else {
                client.create_from_slice(bytemuck::cast_slice(&keys))
            };
            let v_h = if n == 0 {
                client.create_from_slice(bytemuck::bytes_of(&0u32))
            } else {
                client.create_from_slice(bytemuck::cast_slice(&vals))
            };

            let in_alt =
                gpu_radix_sort_u32_pairs(client, &k_h, &v_h, n, 24, use_subgroups, &mut scratch)
                    .expect("radix sort u32 pairs must succeed");

            if n > 0 {
                let (sk_h, sv_h) = scratch.u32_sorted_outputs(&k_h, &v_h, in_alt);
                let sk_bytes = client.read_one(sk_h.clone()).expect("read sorted keys");
                let sv_bytes = client.read_one(sv_h.clone()).expect("read sorted vals");
                let gpu_k: &[u32] = bytemuck::cast_slice(&sk_bytes[..n * 4]);
                let gpu_v: &[u32] = bytemuck::cast_slice(&sv_bytes[..n * 4]);
                let gpu_pairs: Vec<(u32, u32)> =
                    gpu_k.iter().copied().zip(gpu_v.iter().copied()).collect();
                assert_eq!(
                    gpu_pairs, expected,
                    "u32 radix sort mismatch (subgroups={use_subgroups}, n={n})"
                );
            }
        }
    }

    // 64-bit fallback path with duplicate tile_ids + varying depth bits
    let n = 4096usize;
    let tile_ids: Vec<u32> = (0..n).map(|i| ((i * 73 + 5) % 19) as u32).collect();
    let depths_u32: Vec<u32> = (0..n)
        .map(|i| (0.5f32 + ((i * 19 + 3) % 100) as f32 * 0.1).to_bits())
        .collect();
    let splat_ids: Vec<u32> = (0..n as u32).collect();

    let mut expected_u64: Vec<(u32, u32, u32)> = tile_ids
        .iter()
        .copied()
        .zip(depths_u32.iter().copied())
        .zip(splat_ids.iter().copied())
        .map(|((t, d), s)| (t, d, s))
        .collect();
    expected_u64.sort_by_key(|&(t, d, s)| (t, d, s));

    let t_h = client.create_from_slice(bytemuck::cast_slice(&tile_ids));
    let d_h = client.create_from_slice(bytemuck::cast_slice(&depths_u32));
    let s_h = client.create_from_slice(bytemuck::cast_slice(&splat_ids));
    let in_alt = gpu_radix_sort_u64_pairs(
        client,
        &t_h,
        &d_h,
        &s_h,
        n,
        5,
        gpu.supports_subgroups(),
        &mut scratch,
    )
    .expect("radix sort u64 fallback must succeed");

    let (st_h, ss_h) = scratch.u64_sorted_outputs(in_alt);
    let st_bytes = client
        .read_one(st_h.clone())
        .expect("read u64 sorted tiles");
    let ss_bytes = client
        .read_one(ss_h.clone())
        .expect("read u64 sorted splats");
    let gpu_tiles: &[u32] = bytemuck::cast_slice(&st_bytes[..n * 4]);
    let gpu_splats: &[u32] = bytemuck::cast_slice(&ss_bytes[..n * 4]);
    for i in 0..n {
        assert_eq!(gpu_tiles[i], expected_u64[i].0, "u64 tile mismatch at {i}");
        assert_eq!(
            gpu_splats[i], expected_u64[i].2,
            "u64 splat mismatch at {i}"
        );
    }
}

#[test]
fn test_gpu_stage3_binning_and_ranges_match_cpu() {
    use cinder_gs::render::kernels::{
        RadixSortScratch, ScanScratch, clear_tile_ranges_kernel, compute_packed_key_bits,
        count_tiles_kernel, dispatch_cube_count, emit_tile_keys_kernel, gpu_exclusive_scan_total,
        gpu_radix_sort_u32_pairs, identify_tile_ranges_kernel, pack_relative_depth_bits,
        project_gaussians_v2_kernel,
    };
    use cinder_gs::render::tiles::{compute_gaussian_tile_bounds_xy, ellipse_intersects_tile};
    use cubecl::prelude::*;

    let Some(guard) = gpu_session() else {
        return;
    };
    let gpu = guard.gpu();
    let client = gpu.client();

    let scene = varied_scene();
    let n = scene.count;
    let width = 96u32;
    let height = 96u32;
    let tiles_x = width.div_ceil(16);
    let tiles_y = height.div_ceil(16);
    let total_tiles = (tiles_x * tiles_y) as usize;
    let camera = test_camera(width as f32, height as f32);

    let pos_h = client.create_from_slice(scene.positions_bytes());
    let cov_h = client.create_from_slice(scene.covariances_bytes());
    let opa_h = client.create_from_slice(scene.opacities_bytes());
    let sh_h = client.create_from_slice(scene.sh_coeffs_bytes());
    let dummy_u32 = client.create_from_slice(bytemuck::cast_slice(&[0u32, 0u32]));
    let dummy_f32 = client.create_from_slice(bytemuck::cast_slice(&[0.0f32; 9]));
    let split_cnt_h = client.create_from_slice(bytemuck::bytes_of(&0u32));
    let cam_h = client.create_from_slice(camera.as_bytes());
    let attrs_h = client.empty(n * 10 * 4);
    let extents_h = client.empty(n * 4);
    let surv_ids_h = client.empty(n * 4);
    let surv_cnt_h = client.create_from_slice(bytemuck::bytes_of(&0u32));

    unsafe {
        project_gaussians_v2_kernel::launch(
            client,
            dispatch_cube_count((n as u32).div_ceil(256)),
            CubeDim::new_1d(256),
            BufferArg::from_raw_parts(pos_h, n * 3),
            BufferArg::from_raw_parts(cov_h, n * 6),
            BufferArg::from_raw_parts(opa_h, n),
            BufferArg::from_raw_parts(sh_h, n * 48),
            BufferArg::from_raw_parts(dummy_u32.clone(), 1),
            BufferArg::from_raw_parts(dummy_f32, 9),
            BufferArg::from_raw_parts(dummy_u32, 2),
            BufferArg::from_raw_parts(split_cnt_h.clone(), 1),
            BufferArg::from_raw_parts(cam_h, 24),
            BufferArg::from_raw_parts(attrs_h.clone(), n * 10),
            BufferArg::from_raw_parts(extents_h.clone(), n),
            BufferArg::from_raw_parts(surv_ids_h, n),
            BufferArg::from_raw_parts(surv_cnt_h, 1),
            0.0,
            0.0,
            -1.0,
            0.001,
            0.005,
            0.5,
            3,
            0,
            0,
            n as u32,
        );
    }

    let touched_h = client.empty(n * 4);
    let offsets_h = client.empty(n * 4);
    unsafe {
        count_tiles_kernel::launch(
            client,
            dispatch_cube_count((n as u32).div_ceil(256)),
            CubeDim::new_1d(256),
            BufferArg::from_raw_parts(attrs_h.clone(), n * 10),
            BufferArg::from_raw_parts(extents_h.clone(), n),
            BufferArg::from_raw_parts(split_cnt_h.clone(), 1),
            BufferArg::from_raw_parts(touched_h.clone(), n),
            width,
            height,
            tiles_x,
            tiles_y,
            n as u32,
            n as u32,
        );
    }

    let mut scan_scratch = ScanScratch::new();
    let total_entries =
        gpu_exclusive_scan_total(client, &touched_h, &offsets_h, n, &mut scan_scratch)
            .expect("Stage 3 scan must succeed") as usize;
    assert!(total_entries > 0, "test scene must produce tile entries");

    // Verify per-splat tiles_touched matches CPU exact ellipse intersection test
    let attrs_bytes = client.read_one(attrs_h).expect("read attrs");
    let ext_bytes = client.read_one(extents_h.clone()).expect("read extents");
    let touched_bytes = client.read_one(touched_h).expect("read touched");
    let attrs: &[f32] = bytemuck::cast_slice(&attrs_bytes[..n * 40]);
    let exts: &[u32] = bytemuck::cast_slice(&ext_bytes[..n * 4]);
    let gpu_touched: &[u32] = bytemuck::cast_slice(&touched_bytes[..n * 4]);

    let mut cpu_total = 0u32;
    for i in 0..n {
        let p = exts[i];
        let mut expected = 0u32;
        if p != 0 {
            let ex = (p & 0xFFFF) as f32;
            let ey = ((p >> 16) & 0xFFFF) as f32;
            let b = i * 10;
            if let Some((tx0, tx1, ty0, ty1)) = compute_gaussian_tile_bounds_xy(
                attrs[b],
                attrs[b + 1],
                ex,
                ey,
                tiles_x,
                tiles_y,
                width,
                height,
            ) {
                for ty in ty0..=ty1 {
                    for tx in tx0..=tx1 {
                        if ellipse_intersects_tile(
                            attrs[b],
                            attrs[b + 1],
                            [attrs[b + 2], attrs[b + 3], attrs[b + 4]],
                            attrs[b + 5],
                            tx,
                            ty,
                            width,
                            height,
                        ) {
                            expected += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(
            gpu_touched[i], expected,
            "tiles_touched mismatch at splat {i}"
        );
        cpu_total += expected;
    }
    assert_eq!(total_entries as u32, cpu_total);

    let keys_h = client.empty(total_entries * 4);
    let ids_h = client.empty(total_entries * 4);
    let ranges_h = client.empty(total_tiles * 8);
    let attrs_h2 = client.create_from_slice(&attrs_bytes[..n * 40]);
    let (tile_bits, depth_bits) = compute_packed_key_bits(total_tiles);
    let base_depth_bits = camera.near_plane.max(1e-6).to_bits() & 0x7F80_0000;
    let sort_bits = (tile_bits + depth_bits).min(32);
    unsafe {
        emit_tile_keys_kernel::launch(
            client,
            dispatch_cube_count((n as u32).div_ceil(256)),
            CubeDim::new_1d(256),
            BufferArg::from_raw_parts(attrs_h2, n * 10),
            BufferArg::from_raw_parts(extents_h, n),
            BufferArg::from_raw_parts(offsets_h, n),
            BufferArg::from_raw_parts(split_cnt_h, 1),
            BufferArg::from_raw_parts(keys_h.clone(), total_entries),
            BufferArg::from_raw_parts(ids_h.clone(), total_entries),
            width,
            height,
            tiles_x,
            tiles_y,
            n as u32,
            base_depth_bits,
            depth_bits,
        );
    }

    let mut radix_scratch = RadixSortScratch::new();
    let in_alt = gpu_radix_sort_u32_pairs(
        client,
        &keys_h,
        &ids_h,
        total_entries,
        sort_bits,
        gpu.supports_subgroups(),
        &mut radix_scratch,
    )
    .expect("Stage 3 sort must succeed");
    let (sk_h, si_h) = radix_scratch.u32_sorted_outputs(&keys_h, &ids_h, in_alt);

    unsafe {
        clear_tile_ranges_kernel::launch(
            client,
            dispatch_cube_count((total_tiles as u32).div_ceil(256)),
            CubeDim::new_1d(256),
            BufferArg::from_raw_parts(ranges_h.clone(), total_tiles * 2),
            total_tiles as u32,
        );
        identify_tile_ranges_kernel::launch(
            client,
            dispatch_cube_count((total_entries as u32).div_ceil(256)),
            CubeDim::new_1d(256),
            BufferArg::from_raw_parts(sk_h.clone(), total_entries),
            BufferArg::from_raw_parts(ranges_h.clone(), total_tiles * 2),
            total_entries as u32,
            total_tiles as u32,
            depth_bits,
        );
    }

    let r_bytes = client.read_one(ranges_h).expect("read tile ranges");
    let i_bytes = client
        .read_one(si_h.clone())
        .expect("read sorted splat ids");
    let ranges: &[u32] = bytemuck::cast_slice(&r_bytes[..total_tiles * 8]);
    let sorted_ids: &[u32] = bytemuck::cast_slice(&i_bytes[..total_entries * 4]);

    let mut covered = 0usize;
    for t in 0..total_tiles {
        let start = ranges[t * 2] as usize;
        let end = ranges[t * 2 + 1] as usize;
        assert!(start <= end && end <= total_entries);
        covered += end - start;
        for w in sorted_ids[start..end].windows(2) {
            let d0 = pack_relative_depth_bits(
                attrs[w[0] as usize * 10 + 9],
                camera.near_plane,
                depth_bits,
            );
            let d1 = pack_relative_depth_bits(
                attrs[w[1] as usize * 10 + 9],
                camera.near_plane,
                depth_bits,
            );
            assert!(d0 <= d1, "tile {t} not depth-sorted: {d0} > {d1}");
        }
    }
    assert_eq!(covered, total_entries);

    // Regression check: 1 mm separation at z=3.0m must stay distinct at 540p, 1080p, and 4K.
    for &(label, tw, th) in &[
        ("540p", 960u32, 540u32),
        ("1080p", 1920, 1080),
        ("4k", 3840, 2160),
    ] {
        let tiles_count = (tw.div_ceil(16) * th.div_ceil(16)) as usize;
        let (_, db) = compute_packed_key_bits(tiles_count);
        let q0 = pack_relative_depth_bits(3.000, 0.01, db);
        let q1 = pack_relative_depth_bits(3.001, 0.01, db);
        assert!(
            q0 < q1,
            "{label} ({db} depth bits) failed 1 mm depth separation at z=3.0m: {q0} vs {q1}"
        );
    }
}

#[test]
fn test_gpu_polar_rotation_and_inverse_ray_sh_match_cpu_all_degrees() {
    use cinder_gs::deformation::{Cage, CageScene, compute_tet_transforms_cpu};
    use cinder_gs::render::gpu_state::GpuPipelineState;
    use cinder_gs::render::kernels::compute_tet_transforms_kernel;
    use cinder_gs::render::pipeline::RenderPipelineOrchestrator;
    use cubecl::prelude::*;

    let Some(guard) = gpu_session() else {
        return;
    };
    let gpu = guard.gpu();
    let client = gpu.client();

    let mut mesh = TetMesh::create_box_cage([-1.0, -1.0, 1.0], [1.0, 1.0, 3.0]);
    let rot_q = glam::Quat::from_euler(glam::EulerRot::XYZ, 0.55, -0.4, 0.7).normalize();
    let pivot = Vec3::new(0.0, 0.0, 2.0);
    let deformed_verts: Vec<[f32; 3]> = mesh
        .rest_vertices
        .iter()
        .map(|&v| {
            let local = Vec3::from(v) - pivot;
            let stretched = Vec3::new(local.x * 1.4, local.y * 0.75, local.z * 1.15);
            (pivot + rot_q * stretched).into()
        })
        .collect();

    // 1. Verify compute_tet_transforms_kernel polar rotation and max_stretch match CPU
    let num_tets = mesh.num_tets();
    let flat_verts: Vec<f32> = deformed_verts.iter().flat_map(|v| *v).collect();
    let flat_elems: Vec<u32> = mesh.elements.iter().flat_map(|e| e.indices).collect();
    let flat_inv: Vec<f32> = mesh.precomputed.iter().flat_map(|p| p.inv_dm).collect();

    let v_h = client.create_from_slice(bytemuck::cast_slice(&flat_verts));
    let e_h = client.create_from_slice(bytemuck::cast_slice(&flat_elems));
    let i_h = client.create_from_slice(bytemuck::cast_slice(&flat_inv));
    let j_h = client.empty(num_tets * 9 * 4);
    let r_h = client.empty(num_tets * 9 * 4);
    let s_h = client.empty(num_tets * 4);

    unsafe {
        compute_tet_transforms_kernel::launch(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(num_tets as u32),
            BufferArg::from_raw_parts(v_h, flat_verts.len()),
            BufferArg::from_raw_parts(e_h, flat_elems.len()),
            BufferArg::from_raw_parts(i_h, flat_inv.len()),
            BufferArg::from_raw_parts(j_h.clone(), num_tets * 9),
            BufferArg::from_raw_parts(r_h.clone(), num_tets * 9),
            BufferArg::from_raw_parts(s_h, num_tets),
            num_tets as u32,
        );
    }

    let (_, cpu_rots) = compute_tet_transforms_cpu(&mesh, &deformed_verts);
    let r_bytes = client.read_one(r_h).expect("read GPU polar rotations");
    let gpu_rots: &[f32] = bytemuck::cast_slice(&r_bytes[..num_tets * 36]);
    for t in 0..num_tets {
        let cpu_cols = cpu_rots[t].to_cols_array();
        for k in 0..9 {
            assert!(
                (gpu_rots[t * 9 + k] - cpu_cols[k]).abs() < 1e-4,
                "polar rotation mismatch at tet {t} elem {k}: gpu={} cpu={}",
                gpu_rots[t * 9 + k],
                cpu_cols[k]
            );
        }
    }

    // 2. Verify R^T * d inverse-ray GPU SH matches CPU ShRotator across degrees 0..=3
    let scene = varied_scene();
    let mut cage = Cage::new("rot", mesh.clone(), [1.0; 4]);
    for (i, pos) in cage.simulator.current_positions.iter_mut().enumerate() {
        *pos = Vec3::from(deformed_verts[i]);
    }
    mesh.rest_vertices = mesh.rest_vertices.clone();
    let cages = CageScene::auto_bind(vec![cage], &scene.positions);
    let camera = test_camera(80.0, 80.0);

    let mut gpu_state = GpuPipelineState::new();
    for deg in 0..=3 {
        let comp = CompositorUniforms::new(80, 80, [0.02, 0.03, 0.05, 1.0], [0.0, 0.0, -1.0], deg);
        let (cpu_img, _, _, _, _) = RenderPipelineOrchestrator::execute_cpu_reference_with_options(
            &scene,
            Some(&cages),
            &camera,
            &comp,
            80,
            80,
            None,
            0.0,
            0.001,
        );

        let _ = gpu_state
            .execute_frame_gpu(
                gpu,
                &scene,
                Some(&cages),
                &camera,
                &comp,
                0.0,
                0.001,
                cinder_gs::math::MIN_CULL_OPACITY,
                cinder_gs::math::MIN_EFFECTIVE_RADIUS_PX,
                false,
                false,
            )
            .expect("GPU v2 SH degree render must succeed");
        let gpu_img = gpu_state
            .readback_framebuffer(gpu, 80, 80)
            .expect("readback GPU framebuffer");

        let mut max_diff = 0.0f32;
        let mut sum_diff = 0.0f64;
        for (a, b) in cpu_img.rgba8.iter().zip(gpu_img.rgba8.iter()) {
            let d = ((*a as f32) - (*b as f32)).abs() / 255.0;
            max_diff = max_diff.max(d);
            sum_diff += f64::from(d);
        }
        let mean_diff = sum_diff / cpu_img.rgba8.len() as f64;
        assert!(
            mean_diff < 1e-2,
            "SH degree {deg} inverse-ray mean diff too large: {mean_diff}"
        );
        assert!(
            max_diff < 6e-2,
            "SH degree {deg} inverse-ray max diff too large: {max_diff}"
        );
    }
}

#[test]
fn test_gpu_persistent_v2_pipeline_and_dirty_flag_skip() {
    use cinder_gs::deformation::{Cage, CageScene};
    use cinder_gs::render::gpu_state::GpuPipelineState;
    use cinder_gs::render::pipeline::RenderPipelineOrchestrator;

    let Some(guard) = gpu_session() else {
        return;
    };
    let gpu = guard.gpu();

    let scene = varied_scene();
    let mut cage = Cage::new(
        "deform",
        TetMesh::create_box_cage([-1.2, -1.2, 0.8], [1.2, 1.2, 3.2]),
        [1.0; 4],
    );
    cage.simulator.current_positions[0].x += 0.25;
    let mut cages = CageScene::auto_bind(vec![cage], &scene.positions);

    let cam1 = test_camera(96.0, 96.0);
    let comp = CompositorUniforms::new(96, 96, [0.02, 0.02, 0.04, 1.0], [0.0, 0.0, -1.0], 2);

    let mut gpu_state = GpuPipelineState::new();
    let mip = 0.001f32;
    let min_opa = cinder_gs::math::MIN_CULL_OPACITY;
    let min_rad = cinder_gs::math::MIN_EFFECTIVE_RADIUS_PX;
    // Frame 1: cold deform + Stage 3
    let t1 = gpu_state
        .execute_frame_gpu(
            gpu,
            &scene,
            Some(&cages),
            &cam1,
            &comp,
            0.0,
            mip,
            min_opa,
            min_rad,
            false,
            false,
        )
        .expect("frame 1 must succeed");
    assert!(!t1.stage1_skipped);
    assert!(!t1.stage3_cached);
    let img1 = gpu_state.readback_framebuffer(gpu, 96, 96).expect("img1");

    let (cpu_img, _, _, _, _) = RenderPipelineOrchestrator::execute_cpu_reference_with_options(
        &scene,
        Some(&cages),
        &cam1,
        &comp,
        96,
        96,
        None,
        0.0,
        mip,
    );
    let mut max_rgb_diff = 0.0f32;
    let mut sum_rgb_diff = 0.0f64;
    let mut rgb_count = 0usize;
    for (pa, pb) in cpu_img
        .rgba8
        .chunks_exact(4)
        .zip(img1.rgba8.chunks_exact(4))
    {
        for ch in 0..3 {
            let d = ((pa[ch] as f32) - (pb[ch] as f32)).abs() / 255.0;
            max_rgb_diff = max_rgb_diff.max(d);
            sum_rgb_diff += f64::from(d);
            rgb_count += 1;
        }
    }
    let mean_rgb_diff = sum_rgb_diff / rgb_count as f64;
    assert!(
        mean_rgb_diff < 5e-3,
        "v2 pipeline vs CPU mean RGB diff too large: {mean_rgb_diff}"
    );
    assert!(
        max_rgb_diff < 5e-2,
        "v2 pipeline vs CPU max RGB diff too large: {max_rgb_diff}"
    );

    // Frame 2: camera moved, cage stationary -> Stage 1 skipped, Stage 3 recomputed
    let cam2 = CameraUniforms::from_look_at(
        Vec3::new(0.4, 0.2, -1.0),
        Vec3::new(0.0, 0.0, 2.0),
        Vec3::Y,
        50.0f32.to_radians(),
        96.0,
        96.0,
        0.2,
    );
    let comp2 = CompositorUniforms::new(96, 96, [0.02, 0.02, 0.04, 1.0], [0.4, 0.2, -1.0], 2);
    let t2 = gpu_state
        .execute_frame_gpu(
            gpu,
            &scene,
            Some(&cages),
            &cam2,
            &comp2,
            0.0,
            mip,
            min_opa,
            min_rad,
            false,
            false,
        )
        .expect("frame 2 must succeed");
    assert!(t2.stage1_skipped, "stationary cage must skip Stage 1");
    assert!(!t2.stage3_cached, "camera move must invalidate Stage 3");
    let img2 = gpu_state.readback_framebuffer(gpu, 96, 96).expect("img2");

    // Frame 3: static camera + static cage -> Stage 1 skipped, Stage 3 cached, identical pixels
    let t3 = gpu_state
        .execute_frame_gpu(
            gpu,
            &scene,
            Some(&cages),
            &cam2,
            &comp2,
            0.0,
            mip,
            min_opa,
            min_rad,
            false,
            false,
        )
        .expect("frame 3 must succeed");
    assert!(t3.stage1_skipped);
    assert!(t3.stage3_cached);
    let img3 = gpu_state.readback_framebuffer(gpu, 96, 96).expect("img3");
    assert_eq!(img2.rgba8, img3.rgba8);

    // Frame 4: cage moved -> Stage 1 runs again
    cages.cages[0].simulator.current_positions[2].y -= 0.3;
    let t4 = gpu_state
        .execute_frame_gpu(
            gpu,
            &scene,
            Some(&cages),
            &cam2,
            &comp2,
            0.0,
            mip,
            min_opa,
            min_rad,
            false,
            false,
        )
        .expect("frame 4 must succeed");
    assert!(!t4.stage1_skipped, "cage motion must trigger Stage 1");
    assert!(!t4.stage3_cached);
}

#[test]
fn test_gpu_headless_zero_copy_frame_succeeds() {
    use cinder_gs::deformation::{Cage, CageScene};
    use cinder_gs::render::headless::HeadlessRenderer;
    use cinder_gs::render::pipeline::ExecutionBackend;

    let mut renderer = match HeadlessRenderer::new(128, 128, ExecutionBackend::CubeClGpu) {
        Ok(r) => r,
        Err(err) => {
            eprintln!("Headless WGPU unavailable, skipping: {err}");
            return;
        }
    };

    let scene = varied_scene();
    let cage = Cage::new(
        "box",
        TetMesh::create_box_cage([-1.5, -1.5, 0.5], [1.5, 1.5, 3.5]),
        [1.0; 4],
    );
    let cages = CageScene::auto_bind(vec![cage], &scene.positions);
    let camera = test_camera(128.0, 128.0);
    let comp = CompositorUniforms::new(128, 128, [0.05, 0.05, 0.05, 1.0], [0.0, 0.0, -1.0], 2);

    let stats = renderer
        .render_zero_copy(&scene, Some(&cages), &camera, &comp, true, None)
        .expect("zero-copy headless render must succeed");
    assert_eq!(stats.total_gaussians, scene.count);
    assert!(stats.rendered_gaussians > 0);
    assert!(
        renderer.orchestrator.last_backend_error.is_none(),
        "GPU zero-copy fell back unexpectedly: {:?}",
        renderer.orchestrator.last_backend_error
    );

    let full = renderer
        .render(&scene, Some(&cages), &camera, &comp, false, None)
        .expect("readback render must succeed");
    assert_eq!(full.image.rgba8.len(), 128 * 128 * 4);
    assert!(
        full.image
            .rgba8
            .chunks_exact(4)
            .any(|px| px[0] > 20 || px[1] > 20)
    );
}

#[test]
fn test_gpu_unpack_f16_matches_native_f16() {
    use cinder_gs::render::kernels::{
        project_gaussians_quantized_f16_kernel, project_gaussians_quantized_kernel,
    };
    use cinder_gs::scene::soa::QuantizedGaussianSceneSoa;
    use cubecl::prelude::*;

    let Some(guard) = gpu_session() else {
        return;
    };
    let gpu = guard.gpu();
    let client = gpu.client();

    // Craft quantized covariances spanning subnormal, negative, normal, and large values
    let samples_f16: Vec<[half::f16; 6]> = vec![
        [
            half::f16::from_f32(0.05),
            half::f16::from_bits(0x0001), // smallest positive subnormal
            half::f16::from_bits(0x8001), // negative subnormal
            half::f16::from_f32(0.04),
            half::f16::from_bits(0x03FF), // largest positive subnormal
            half::f16::from_f32(0.06),
        ],
        [
            half::f16::from_f32(12.5),
            half::f16::from_f32(-0.75),
            half::f16::from_bits(0x8000), // -0.0
            half::f16::from_f32(8.0),
            half::f16::from_bits(0x0000), // +0.0
            half::f16::from_f32(10.0),
        ],
        [
            half::f16::from_bits(0x7BFF), // 65504.0 max finite f16
            half::f16::from_f32(0.0),
            half::f16::from_f32(0.0),
            half::f16::from_f32(1.0),
            half::f16::from_f32(0.0),
            half::f16::from_f32(1.0),
        ],
    ];

    let n = samples_f16.len();
    let mut positions = Vec::with_capacity(n * 3);
    let mut cov_packed = Vec::with_capacity(n * 3);
    let mut opacities = Vec::with_capacity(n);
    for (i, c) in samples_f16.iter().enumerate() {
        positions.extend_from_slice(&[0.0, 0.0, 2.5 + i as f32 * 0.5]);
        let p0 = (c[0].to_bits() as u32) | ((c[1].to_bits() as u32) << 16);
        let p1 = (c[2].to_bits() as u32) | ((c[3].to_bits() as u32) << 16);
        let p2 = (c[4].to_bits() as u32) | ((c[5].to_bits() as u32) << 16);
        cov_packed.extend_from_slice(&[p0, p1, p2]);
        opacities.push(0.8);
    }

    let qscene = QuantizedGaussianSceneSoa {
        positions: positions.clone(),
        covariances_f16_packed: cov_packed.clone(),
        octahedral_rotations: vec![0; n],
        opacities: opacities.clone(),
        sh_coeffs_f16_packed: vec![0; n * 2],
        sh_degree: 0,
        count: n,
    };
    let ref_soa = qscene.to_soa();
    let cam = test_camera(96.0, 96.0);
    let cpu_proj = project_scene_cpu(&ref_soa, &cam);

    let pos_h = client.create_from_slice(bytemuck::cast_slice(&positions));
    let cov_h = client.create_from_slice(bytemuck::cast_slice(&cov_packed));
    let opa_h = client.create_from_slice(bytemuck::cast_slice(&opacities));
    let cam_h = client.create_from_slice(cam.as_bytes());
    let pts_h = client.empty(n * 8);
    let dep_h = client.empty(n * 4);
    let rad_h = client.empty(n * 4);
    let con_h = client.empty(n * 12);
    let out_o_h = client.empty(n * 4);
    let vis_h = client.empty(n * 4);

    unsafe {
        project_gaussians_quantized_kernel::launch(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(256),
            BufferArg::from_raw_parts(pos_h.clone(), n * 3),
            BufferArg::from_raw_parts(cov_h.clone(), n * 3),
            BufferArg::from_raw_parts(opa_h.clone(), n),
            BufferArg::from_raw_parts(cam_h.clone(), 24),
            0.0,
            0.004,
            0.3,
            BufferArg::from_raw_parts(pts_h.clone(), n * 2),
            BufferArg::from_raw_parts(dep_h.clone(), n),
            BufferArg::from_raw_parts(rad_h.clone(), n),
            BufferArg::from_raw_parts(con_h.clone(), n * 3),
            BufferArg::from_raw_parts(out_o_h.clone(), n),
            BufferArg::from_raw_parts(vis_h.clone(), n),
            n as u32,
        );
    }

    let con_bytes = client
        .read_one(con_h.clone())
        .expect("read bit-twiddle conics");
    let conics_bit: &[f32] = bytemuck::cast_slice(&con_bytes[..n * 12]);
    for (i, (&bit, &cpu)) in conics_bit.iter().zip(cpu_proj.conics.iter()).enumerate() {
        assert!(
            (bit - cpu).abs() < 1e-4,
            "unpack_f16 conic mismatch at {i}: {bit} vs {cpu}"
        );
    }

    if client
        .properties()
        .supports_type(cubecl::ir::ElemType::Float(cubecl::ir::FloatKind::F16))
    {
        unsafe {
            project_gaussians_quantized_f16_kernel::launch(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(256),
                BufferArg::from_raw_parts(pos_h, n * 3),
                BufferArg::from_raw_parts(cov_h, n * 6),
                BufferArg::from_raw_parts(opa_h, n),
                BufferArg::from_raw_parts(cam_h, 24),
                0.0,
                0.004,
                0.3,
                BufferArg::from_raw_parts(pts_h, n * 2),
                BufferArg::from_raw_parts(dep_h, n),
                BufferArg::from_raw_parts(rad_h, n),
                BufferArg::from_raw_parts(con_h.clone(), n * 3),
                BufferArg::from_raw_parts(out_o_h, n),
                BufferArg::from_raw_parts(vis_h, n),
                n as u32,
            );
        }
        let native_bytes = client.read_one(con_h).expect("read native f16 conics");
        let conics_native: &[f32] = bytemuck::cast_slice(&native_bytes[..n * 12]);
        for (i, (&bit, &nat)) in conics_bit.iter().zip(conics_native.iter()).enumerate() {
            assert!(
                (bit - nat).abs() < 1e-4,
                "unpack_f16 vs native f16 divergence at {i}: {bit} vs {nat}"
            );
        }
    }
}

#[test]
fn test_gpu_atomic_compaction_and_splitting_v2() {
    use cinder_gs::deformation::deform_scene_adaptive_cpu;
    use cinder_gs::math::projection::ProjectionOptions;
    use cinder_gs::render::gpu_bridge::{
        deform_geometry_with_splitting_gpu, project_gaussians_cull_gpu,
    };
    use cinder_gs::render::project_scene_cull_cpu;

    let Some(guard) = gpu_session() else {
        return;
    };
    let client = guard.gpu().client();

    // Multi-workgroup (> 256 splats) visibility compaction test
    let large_scene = fibonacci_cluster(1500, 1.5, [0.0, 0.0, 2.5]);
    let camera = test_camera(160.0, 120.0);
    let opts = ProjectionOptions {
        mip_filter_3d_var: 0.002,
        min_opacity: 0.01,
        min_radius_px: 0.4,
        ..ProjectionOptions::default()
    };

    let cpu_cull = project_scene_cull_cpu(&large_scene, &camera, &opts);
    let gpu_cull = project_gaussians_cull_gpu(
        client,
        &large_scene.positions,
        &large_scene.covariances_3d,
        &large_scene.opacities,
        &camera,
        opts.mip_filter_3d_var,
        opts.min_opacity,
        opts.min_radius_px,
    )
    .expect("multi-workgroup cull must succeed");
    assert_eq!(
        gpu_cull.surviving_count,
        cpu_cull.projection.num_surviving() as u32
    );
    assert_eq!(gpu_cull.indirect_dispatch, cpu_cull.indirect_dispatch);

    // Compact splitting test across >256 splats under high stretch
    let mesh = TetMesh::create_box_cage([-2.0, -2.0, 0.5], [2.0, 2.0, 4.5]);
    let bindings = mesh.bind_flat_positions(&large_scene.positions);
    let mut deformed = mesh.rest_vertices.clone();
    for v in &mut deformed {
        v[0] *= 2.1; // strong stretch > 1.5
    }
    let gpu_split = deform_geometry_with_splitting_gpu(
        client,
        &large_scene.positions,
        &large_scene.covariances_3d,
        &large_scene.opacities,
        &mesh,
        &bindings,
        &deformed,
        1.5,
        0.001,
    )
    .expect("compact split deform must succeed");

    let uniforms = DeformationUniforms::new(
        [0.0, 0.0, -1.0],
        0,
        large_scene.count as u32,
        mesh.num_tets() as u32,
    )
    .with_strain_splitting(1.5)
    .with_mip_filter_3d(0.001);
    let cpu_res = deform_scene_adaptive_cpu(&large_scene, &mesh, &bindings, &deformed, &uniforms);
    let cpu_split_count = cpu_res.scene.count - large_scene.count;

    assert!(
        gpu_split.split_count > 0,
        "expected splits under 2.1x stretch"
    );
    assert_eq!(gpu_split.split_count, cpu_split_count);
    assert_eq!(gpu_split.opacities.len(), cpu_res.scene.count);
    assert_eq!(gpu_split.split_source_indices.len(), cpu_split_count);
}
