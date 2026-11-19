//! Honest, self-measuring benchmark suite.
//!
//! Every number printed here is measured on the machine that runs it. There are
//! no hard-coded stage budgets: if the GPU backend is unavailable the table is
//! simply omitted with the reason.

use anyhow::{Result, anyhow};
use cinder_gs::deformation::{Cage, CageScene, DeformationUniforms, TetMesh};
use cinder_gs::render::camera::CameraUniforms;
use cinder_gs::render::gpu_state::{GpuPipelineState, GpuStageTimings};
use cinder_gs::render::headless::HeadlessRenderer;
use cinder_gs::render::pipeline::ExecutionBackend;
use cinder_gs::render::project_scene_cpu;
use cinder_gs::render::rasterizer::{CompositorUniforms, rasterize_scene_cpu};
use cinder_gs::scene::autoframe::compute_framing_from_positions;
use cinder_gs::scene::fibonacci_cluster;
use glam::Vec3;
use std::time::Instant;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const SPLAT_COUNT: usize = 100_000;
const CPU_ITERS: u32 = 5;
const GPU_ITERS: u32 = 20;

fn build_scene() -> Result<(cinder_gs::scene::GaussianSceneSoa, CageScene)> {
    let scene = fibonacci_cluster(SPLAT_COUNT, 1.5, [0.0, 0.0, 3.0]);
    if scene.count == 0 {
        return Err(anyhow!("synthetic scene generation produced no splats"));
    }

    let framing = compute_framing_from_positions(&scene.positions);
    let cage = TetMesh::create_box_cage(framing.hero_min, framing.hero_max);
    let mut cages = CageScene::auto_bind(
        vec![Cage::new("hero", cage, [0.0, 0.85, 1.0, 0.45])],
        &scene.positions,
    );

    // Non-trivial deformation so the Jacobian and covariance paths are exercised.
    if let Some(cage) = cages.cages.first_mut() {
        if !cage.simulator.current_positions.is_empty() {
            cage.simulator.current_positions[0].x += 0.2;
            cage.simulator.current_positions[1].y -= 0.15;
        }
    }

    Ok((scene, cages))
}

fn print_header(device: &str) {
    println!("============================================================");
    println!("   CinderGS Benchmark Suite — measured on this machine");
    println!(
        "   CPU threads available: {}",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    );
    println!("   Device: {device}");
    println!("   Resolution: {WIDTH}x{HEIGHT}, {} splats", SPLAT_COUNT);
    println!("============================================================");
}

fn main() -> Result<()> {
    println!(
        "Generating synthetic scene with {} Gaussians...",
        SPLAT_COUNT
    );
    let (scene, cages) = build_scene()?;
    print_header("CPU reference (Rayon) / detected GPU below");

    let camera = CameraUniforms::from_look_at(
        Vec3::new(0.0, 0.0, -1.0),
        Vec3::new(0.0, 0.0, 3.0),
        Vec3::Y,
        45.0f32.to_radians(),
        WIDTH as f32,
        HEIGHT as f32,
        0.2,
    );

    let compositor =
        CompositorUniforms::new(WIDTH, HEIGHT, [0.05, 0.05, 0.05, 1.0], [0.0, 0.0, -1.0], 3);

    let num_tets: usize = cages.cages.iter().map(|c| c.mesh.num_tets()).sum();
    let def_uniforms = DeformationUniforms::new(
        [0.0, 0.0, -1.0],
        compositor.sh_degree,
        scene.count as u32,
        num_tets as u32,
    );

    // ---------------------------------------------------------------------
    // Profile A: CPU reference stages
    // ---------------------------------------------------------------------
    println!("\nWarming up the CPU reference pipeline...");
    for _ in 0..3 {
        let deformed = cinder_gs::deformation::deform_cage_scene_cpu(&scene, &cages, &def_uniforms);
        let proj = project_scene_cpu(&deformed.scene, &camera);
        let binning = proj.bin_and_sort(WIDTH, HEIGHT);
        let _ = rasterize_scene_cpu(
            &proj.points_xy,
            &proj.conics,
            &deformed.scene.opacities,
            &deformed.scene.sh_coeffs,
            &deformed.scene.positions,
            &binning.sorted_entries,
            &binning.tile_ranges,
            &compositor,
        );
    }

    println!("Timing CPU reference stages ({CPU_ITERS} iterations)...");
    let mut t_deform = 0.0f64;
    let mut t_project = 0.0f64;
    let mut t_binning = 0.0f64;
    let mut t_composite = 0.0f64;

    for _ in 0..CPU_ITERS {
        let start = Instant::now();
        let deformed = cinder_gs::deformation::deform_cage_scene_cpu(&scene, &cages, &def_uniforms);
        t_deform += start.elapsed().as_secs_f64();

        let start = Instant::now();
        let proj = project_scene_cpu(&deformed.scene, &camera);
        t_project += start.elapsed().as_secs_f64();

        let start = Instant::now();
        let binning = proj.bin_and_sort(WIDTH, HEIGHT);
        t_binning += start.elapsed().as_secs_f64();

        let start = Instant::now();
        let _ = rasterize_scene_cpu(
            &proj.points_xy,
            &proj.conics,
            &deformed.scene.opacities,
            &deformed.scene.sh_coeffs,
            &deformed.scene.positions,
            &binning.sorted_entries,
            &binning.tile_ranges,
            &compositor,
        );
        t_composite += start.elapsed().as_secs_f64();
    }

    let cpu_stage_ms = [
        t_deform / CPU_ITERS as f64 * 1000.0,
        t_project / CPU_ITERS as f64 * 1000.0,
        t_binning / CPU_ITERS as f64 * 1000.0,
        t_composite / CPU_ITERS as f64 * 1000.0,
    ];
    let cpu_stage_total: f64 = cpu_stage_ms.iter().sum();

    println!("\n=== CPU REFERENCE — per stage (no upload/blit) ===");
    println!(
        "Stage 1  Cage deformation (multi-cage): {:9.2} ms",
        cpu_stage_ms[0]
    );
    println!(
        "Stage 2  EWA projection:                {:9.2} ms",
        cpu_stage_ms[1]
    );
    println!(
        "Stage 3  Tile binning + depth sort:     {:9.2} ms",
        cpu_stage_ms[2]
    );
    println!(
        "Stage 4  Tile compositor:               {:9.2} ms",
        cpu_stage_ms[3]
    );
    println!(
        "Sum of stages:                          {:9.2} ms",
        cpu_stage_total
    );

    // End-to-end includes the offscreen texture upload and fullscreen blit.
    match HeadlessRenderer::new(WIDTH, HEIGHT, ExecutionBackend::CpuReference) {
        Ok(mut renderer) => {
            for _ in 0..2 {
                renderer.render(&scene, Some(&cages), &camera, &compositor, false, None)?;
            }
            let iterations = 5;
            let start = Instant::now();
            for _ in 0..iterations {
                renderer.render(&scene, Some(&cages), &camera, &compositor, false, None)?;
            }
            let e2e = start.elapsed().as_secs_f64() / iterations as f64 * 1000.0;
            println!("End-to-end frame (incl. upload + blit):  {e2e:9.2} ms");
            println!(
                "Effective framerate:                     {:9.1} FPS",
                1000.0 / e2e
            );
        }
        Err(err) => {
            println!("End-to-end CPU frame: unavailable ({err})");
        }
    }

    // ---------------------------------------------------------------------
    // Profile B: CubeCL GPU stages
    // ---------------------------------------------------------------------
    println!("\nInitializing the CubeCL GPU backend...");
    let mut renderer = match HeadlessRenderer::new(WIDTH, HEIGHT, ExecutionBackend::CubeClGpu) {
        Ok(renderer) => renderer,
        Err(err) => {
            println!("GPU backend unavailable: {err}");
            println!("\nOnly measured CPU numbers are reported for this run.");
            return Ok(());
        }
    };

    let mut cages = cages;
    if let Some(gpu) = renderer.orchestrator.compute() {
        println!(
            "GPU compute device: {} | f16={} | subgroups={} | timestamps={} | shared={}",
            gpu.describe(),
            gpu.supports_f16(),
            gpu.supports_subgroups(),
            gpu.supports_timestamps(),
            gpu.is_shared_device()
        );

        let mut gpu_state = GpuPipelineState::new();
        println!("Warming up persistent v2 GPU pipeline (3 iterations)...");
        for i in 0..3 {
            if let Some(cage) = cages.cages.first_mut()
                && !cage.simulator.current_positions.is_empty()
            {
                cage.simulator.current_positions[0].x += if i % 2 == 0 { 0.05 } else { -0.05 };
            }
            let _ = gpu_state.execute_frame_gpu(
                gpu,
                &scene,
                Some(&cages),
                &camera,
                &compositor,
                0.0,
                0.0,
                cinder_gs::math::MIN_CULL_OPACITY,
                cinder_gs::math::MIN_EFFECTIVE_RADIUS_PX,
                true,
                true,
            )?;
        }

        // Scenario 1: Dynamic cage deformation every frame
        let mut dyn_sum = GpuStageTimings::default();
        let dyn_wall_start = Instant::now();
        for i in 0..GPU_ITERS {
            if let Some(cage) = cages.cages.first_mut()
                && !cage.simulator.current_positions.is_empty()
            {
                let delta = 0.02;
                cage.simulator.current_positions[0].x += if i % 2 == 0 { delta } else { -delta };
            }
            let t = gpu_state.execute_frame_gpu(
                gpu,
                &scene,
                Some(&cages),
                &camera,
                &compositor,
                0.0,
                0.0,
                cinder_gs::math::MIN_CULL_OPACITY,
                cinder_gs::math::MIN_EFFECTIVE_RADIUS_PX,
                true,
                true,
            )?;
            dyn_sum.stage1_deform_ms += t.stage1_deform_ms;
            dyn_sum.stage2_project_ms += t.stage2_project_ms;
            dyn_sum.stage3_bin_sort_ms += t.stage3_bin_sort_ms;
            dyn_sum.stage4_composite_ms += t.stage4_composite_ms;
            dyn_sum.used_hardware_timestamps |= t.used_hardware_timestamps;
            dyn_sum.surviving_gaussians = t.surviving_gaussians;
            dyn_sum.total_tile_entries = t.total_tile_entries;
        }
        let dyn_wall_ms = dyn_wall_start.elapsed().as_secs_f64() / GPU_ITERS as f64 * 1000.0;

        // Scenario 2: Camera orbit with stationary cage (Stage 1 = 0.0 ms)
        let _ = gpu_state.execute_frame_gpu(
            gpu,
            &scene,
            Some(&cages),
            &camera,
            &compositor,
            0.0,
            0.0,
            cinder_gs::math::MIN_CULL_OPACITY,
            cinder_gs::math::MIN_EFFECTIVE_RADIUS_PX,
            false,
            false,
        )?;

        let mut orb_sum = GpuStageTimings::default();
        for i in 0..GPU_ITERS {
            let angle = (i as f32 + 1.0) * 0.03;
            let eye = Vec3::new(angle.sin() * 4.0, 0.3, 3.0 - angle.cos() * 4.0);
            let orbit_cam = CameraUniforms::from_look_at(
                eye,
                Vec3::new(0.0, 0.0, 3.0),
                Vec3::Y,
                45.0f32.to_radians(),
                WIDTH as f32,
                HEIGHT as f32,
                0.2,
            );
            let orbit_comp = CompositorUniforms::new(
                WIDTH,
                HEIGHT,
                compositor.background_color,
                eye.into(),
                compositor.sh_degree,
            );
            let t = gpu_state.execute_frame_gpu(
                gpu,
                &scene,
                Some(&cages),
                &orbit_cam,
                &orbit_comp,
                0.0,
                0.0,
                cinder_gs::math::MIN_CULL_OPACITY,
                cinder_gs::math::MIN_EFFECTIVE_RADIUS_PX,
                true,
                true,
            )?;
            orb_sum.stage1_deform_ms += t.stage1_deform_ms;
            orb_sum.stage2_project_ms += t.stage2_project_ms;
            orb_sum.stage3_bin_sort_ms += t.stage3_bin_sort_ms;
            orb_sum.stage4_composite_ms += t.stage4_composite_ms;
        }

        // Scenario 3: Static camera & cage (Stage 1 skipped + Stage 3 cached)
        let _ = gpu_state.execute_frame_gpu(
            gpu,
            &scene,
            Some(&cages),
            &camera,
            &compositor,
            0.0,
            0.0,
            cinder_gs::math::MIN_CULL_OPACITY,
            cinder_gs::math::MIN_EFFECTIVE_RADIUS_PX,
            false,
            false,
        )?;
        let mut st_sum = GpuStageTimings::default();
        for _ in 0..GPU_ITERS {
            let t = gpu_state.execute_frame_gpu(
                gpu,
                &scene,
                Some(&cages),
                &camera,
                &compositor,
                0.0,
                0.0,
                cinder_gs::math::MIN_CULL_OPACITY,
                cinder_gs::math::MIN_EFFECTIVE_RADIUS_PX,
                true,
                false,
            )?;
            st_sum.stage1_deform_ms += t.stage1_deform_ms;
            st_sum.stage2_project_ms += t.stage2_project_ms;
            st_sum.stage3_bin_sort_ms += t.stage3_bin_sort_ms;
            st_sum.stage4_composite_ms += t.stage4_composite_ms;
        }

        let n_f = GPU_ITERS as f64;
        let dyn_stages = [
            dyn_sum.stage1_deform_ms / n_f,
            dyn_sum.stage2_project_ms / n_f,
            dyn_sum.stage3_bin_sort_ms / n_f,
            dyn_sum.stage4_composite_ms / n_f,
        ];
        let orb_stages = [
            orb_sum.stage1_deform_ms / n_f,
            orb_sum.stage2_project_ms / n_f,
            orb_sum.stage3_bin_sort_ms / n_f,
            orb_sum.stage4_composite_ms / n_f,
        ];
        let st_stages = [
            st_sum.stage1_deform_ms / n_f,
            st_sum.stage2_project_ms / n_f,
            st_sum.stage3_bin_sort_ms / n_f,
            st_sum.stage4_composite_ms / n_f,
        ];

        let timing_src = if dyn_sum.used_hardware_timestamps {
            "WGPU TIMESTAMP_QUERY"
        } else {
            "host flush fallback"
        };

        println!("\n=== Persistent CubeCL GPU Pipeline v2 — Dynamic Cage ({timing_src}) ===");
        println!(
            "Stage 1  Deform + Polar Rot (GPU):      {:9.3} ms",
            dyn_stages[0]
        );
        println!(
            "Stage 2  EWA Proj + R^T d SH (GPU):     {:9.3} ms",
            dyn_stages[1]
        );
        println!(
            "Stage 3  Scan + RadixSort + Bin (GPU):  {:9.3} ms",
            dyn_stages[2]
        );
        println!(
            "Stage 4  Tile Compositor RGBA8 (GPU):   {:9.3} ms",
            dyn_stages[3]
        );
        println!(
            "Sum of GPU stages:                      {:9.3} ms  ({:.1} FPS)",
            dyn_stages.iter().sum::<f64>(),
            1000.0 / dyn_stages.iter().sum::<f64>().max(1e-6)
        );
        println!(
            "Wall time w/ profiling overhead:        {:9.3} ms  (surviving={}, entries={})",
            dyn_wall_ms, dyn_sum.surviving_gaussians, dyn_sum.total_tile_entries
        );

        println!("\n=== Persistent GPU v2 — Camera Orbit (Stationary Cage) ===");
        println!(
            "Stage 1 (skipped)={:.3} ms | Stage 2={:.3} ms | Stage 3={:.3} ms | Stage 4={:.3} ms => Total {:7.3} ms ({:.1} FPS)",
            orb_stages[0],
            orb_stages[1],
            orb_stages[2],
            orb_stages[3],
            orb_stages.iter().sum::<f64>(),
            1000.0 / orb_stages.iter().sum::<f64>().max(1e-6)
        );

        println!("\n=== Persistent GPU v2 — Static View (Full Temporal Cache) ===");
        println!(
            "Stage 1 (skipped)={:.3} ms | Stage 2={:.3} ms | Stage 3 (cached)={:.3} ms | Stage 4={:.3} ms => Total {:7.3} ms ({:.1} FPS)",
            st_stages[0],
            st_stages[1],
            st_stages[2],
            st_stages[3],
            st_stages.iter().sum::<f64>(),
            1000.0 / st_stages.iter().sum::<f64>().max(1e-6)
        );

        // End-to-end zero-copy headless GPU render (no PCIe framebuffer readback, no profiling flushes)
        for _ in 0..3 {
            let _ = renderer.render_zero_copy(
                &scene,
                Some(&cages),
                &camera,
                &compositor,
                false,
                None,
            )?;
        }
        let e2e_iters = 20;
        let start = Instant::now();
        for i in 0..e2e_iters {
            if let Some(cage) = cages.cages.first_mut()
                && !cage.simulator.current_positions.is_empty()
            {
                cage.simulator.current_positions[0].x += if i % 2 == 0 { 0.02 } else { -0.02 };
            }
            let _ = renderer.render_zero_copy(
                &scene,
                Some(&cages),
                &camera,
                &compositor,
                false,
                None,
            )?;
        }
        let e2e_dyn = start.elapsed().as_secs_f64() / e2e_iters as f64 * 1000.0;

        // Orbit zero-copy (cage stationary)
        let _ =
            renderer.render_zero_copy(&scene, Some(&cages), &camera, &compositor, false, None)?;
        let start_orb = Instant::now();
        for i in 0..e2e_iters {
            let angle = (i as f32 + 1.0) * 0.03;
            let eye = Vec3::new(angle.sin() * 4.0, 0.3, 3.0 - angle.cos() * 4.0);
            let orbit_cam = CameraUniforms::from_look_at(
                eye,
                Vec3::new(0.0, 0.0, 3.0),
                Vec3::Y,
                45.0f32.to_radians(),
                WIDTH as f32,
                HEIGHT as f32,
                0.2,
            );
            let _ = renderer.render_zero_copy(
                &scene,
                Some(&cages),
                &orbit_cam,
                &compositor,
                false,
                None,
            )?;
        }
        let e2e_orb = start_orb.elapsed().as_secs_f64() / e2e_iters as f64 * 1000.0;

        println!(
            "\n=== End-to-end Zero-Copy Headless GPU Frame (incl. copy_buffer_to_texture + submit) ==="
        );
        println!(
            "Dynamic cage:   {e2e_dyn:9.2} ms  ({:7.1} FPS)",
            1000.0 / e2e_dyn
        );
        println!(
            "Camera orbit:   {e2e_orb:9.2} ms  ({:7.1} FPS)",
            1000.0 / e2e_orb
        );
    } else {
        println!(
            "GPU compute unavailable: {}",
            renderer
                .orchestrator
                .gpu_error
                .clone()
                .unwrap_or_else(|| "no client".to_string())
        );
    }

    if let Some(error) = &renderer.orchestrator.last_backend_error {
        println!("note: the last frame fell back to CPU: {error}");
    }

    Ok(())
}
