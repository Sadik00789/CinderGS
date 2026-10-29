//! Honest, self-measuring benchmark suite.
//!
//! Every number printed here is measured on the machine that runs it. There are
//! no hard-coded stage budgets: if the GPU backend is unavailable the table is
//! simply omitted with the reason.

use anyhow::{Result, anyhow};
use cinder_gs::deformation::{Cage, CageScene, DeformationUniforms, GaussianBinding, TetMesh};
use cinder_gs::render::camera::CameraUniforms;
use cinder_gs::render::gpu_bridge::{
    DeformedGeometry, composite_gpu, deform_geometry_gpu, project_gaussians_gpu,
};
use cinder_gs::render::headless::HeadlessRenderer;
use cinder_gs::render::pipeline::ExecutionBackend;
use cinder_gs::render::project_scene_cpu;
use cinder_gs::render::rasterizer::{CompositorUniforms, rasterize_scene_cpu};
use cinder_gs::render::tiles::TileBinningOutputs;
use cinder_gs::scene::autoframe::compute_framing_from_positions;
use cinder_gs::scene::fibonacci_cluster;
use glam::Vec3;
use std::time::Instant;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const SPLAT_COUNT: usize = 100_000;
const CPU_ITERS: u32 = 10;
const GPU_ITERS: u32 = 5;

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

    let mut gpu_table = None;
    if let Some(gpu) = renderer.orchestrator.compute() {
        println!("GPU compute device: {}", gpu.describe());
        let client = gpu.client();

        let cage = cages
            .cages
            .first()
            .ok_or_else(|| anyhow!("benchmark cage missing"))?;
        let bindings: Vec<GaussianBinding> = cages
            .bindings
            .iter()
            .map(|binding| {
                if binding.is_bound() && binding.cage == 0 {
                    GaussianBinding::new(binding.tet, binding.weights)
                } else {
                    GaussianBinding::unbound()
                }
            })
            .collect();
        let deformed_vertices = cage.vertices();

        // Warmup: JIT-compile kernels and stabilise the memory pool.
        println!("Warming up GPU kernels (2 iterations)...");
        for _ in 0..2 {
            let geometry = deform_geometry_gpu(
                client,
                &scene.positions,
                &scene.covariances_3d,
                &scene.opacities,
                &cage.mesh,
                &bindings,
                &deformed_vertices,
            )?;
            let projection =
                project_gaussians_gpu(client, &geometry.positions, &geometry.covariances, &camera)?;
            let binning = projection.bin_and_sort(WIDTH, HEIGHT);
            let _ = composite_gpu(
                client,
                &projection,
                &geometry.opacities,
                &scene.sh_coeffs,
                &geometry.positions,
                &binning,
                &compositor,
            )?;
        }

        println!("Timing GPU stages ({GPU_ITERS} iterations, dispatch + readback)...");
        let mut t_def = 0.0f64;
        let mut t_proj = 0.0f64;
        let mut t_bin = 0.0f64;
        let mut t_comp = 0.0f64;
        let mut last_geometry: Option<DeformedGeometry> = None;
        let mut last_projection = None;
        let mut last_binning: Option<TileBinningOutputs> = None;

        for _ in 0..GPU_ITERS {
            let start = Instant::now();
            let geometry = deform_geometry_gpu(
                client,
                &scene.positions,
                &scene.covariances_3d,
                &scene.opacities,
                &cage.mesh,
                &bindings,
                &deformed_vertices,
            )?;
            t_def += start.elapsed().as_secs_f64();

            let start = Instant::now();
            let projection =
                project_gaussians_gpu(client, &geometry.positions, &geometry.covariances, &camera)?;
            t_proj += start.elapsed().as_secs_f64();

            // Stage 3 is CPU-side in this build (documented in README).
            let start = Instant::now();
            let binning = projection.bin_and_sort(WIDTH, HEIGHT);
            t_bin += start.elapsed().as_secs_f64();

            let start = Instant::now();
            let _ = composite_gpu(
                client,
                &projection,
                &geometry.opacities,
                &scene.sh_coeffs,
                &geometry.positions,
                &binning,
                &compositor,
            )?;
            t_comp += start.elapsed().as_secs_f64();

            last_geometry = Some(geometry);
            last_projection = Some(projection);
            last_binning = Some(binning);
        }

        let gpu_stage_ms = [
            t_def / GPU_ITERS as f64 * 1000.0,
            t_proj / GPU_ITERS as f64 * 1000.0,
            t_bin / GPU_ITERS as f64 * 1000.0,
            t_comp / GPU_ITERS as f64 * 1000.0,
        ];
        gpu_table = Some(gpu_stage_ms);

        // Keep the last results alive so the compiler cannot elide work.
        if let (Some(geometry), Some(projection), Some(binning)) = (
            last_geometry.as_ref(),
            last_projection.as_ref(),
            last_binning.as_ref(),
        ) {
            println!(
                "  (last frame: {} deformed splats, {} projected, {} tile entries)",
                geometry.opacities.len(),
                projection.count,
                binning.total_entries
            );
        }
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

    if let Some(gpu_ms) = gpu_table {
        println!("\n=== CubeCL GPU — per stage (dispatch + readback) ===");
        println!(
            "Stage 1  Cage deformation (GPU):        {:9.2} ms",
            gpu_ms[0]
        );
        println!(
            "Stage 2  EWA projection (GPU):          {:9.2} ms",
            gpu_ms[1]
        );
        println!(
            "Stage 3  Tile binning + sort (CPU):     {:9.2} ms",
            gpu_ms[2]
        );
        println!(
            "Stage 4  Tile compositor (GPU+readback):{:9.2} ms",
            gpu_ms[3]
        );
        println!(
            "Sum of stages:                          {:9.2} ms",
            gpu_ms.iter().sum::<f64>()
        );
    }

    // End-to-end GPU frame through the real orchestrator (includes texture upload,
    // blit, and wireframe pass setup).
    for _ in 0..2 {
        renderer.render(&scene, Some(&cages), &camera, &compositor, false, None)?;
    }
    let iterations = 5;
    let start = Instant::now();
    for _ in 0..iterations {
        renderer.render(&scene, Some(&cages), &camera, &compositor, false, None)?;
    }
    let e2e = start.elapsed().as_secs_f64() / iterations as f64 * 1000.0;
    println!("\n=== End-to-end GPU frame (incl. upload + blit) ===");
    println!("Frame latency:  {e2e:9.2} ms");
    println!("Framerate:      {:9.1} FPS", 1000.0 / e2e);
    if let Some(error) = &renderer.orchestrator.last_backend_error {
        println!("note: the last frame fell back to CPU: {error}");
    }

    Ok(())
}
