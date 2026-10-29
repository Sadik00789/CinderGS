//! Criterion benchmarks for the CPU reference stages.
//!
//! Workload: 20,000 Gaussians at 1280x720. Each stage is benchmarked in isolation
//! after the previous stage has been computed once, so the numbers are per-stage
//! costs rather than a cumulative pipeline.

use cinder_gs::deformation::{Cage, CageScene, DeformationUniforms, GaussianBinding, TetMesh};
use cinder_gs::render::camera::CameraUniforms;
use cinder_gs::render::project_scene_cpu;
use cinder_gs::render::rasterizer::{CompositorUniforms, rasterize_scene_cpu};
use cinder_gs::scene::autoframe::compute_framing_from_positions;
use cinder_gs::scene::fibonacci_cluster;
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use glam::Vec3;

const SPLATS: usize = 20_000;
const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

struct Fixture {
    scene: cinder_gs::scene::GaussianSceneSoa,
    cages: CageScene,
    bindings: Vec<GaussianBinding>,
    uniforms: DeformationUniforms,
    camera: CameraUniforms,
    compositor: CompositorUniforms,
}

fn fixture() -> Fixture {
    let scene = fibonacci_cluster(SPLATS, 1.0, [0.0, 0.0, 3.0]);
    let framing = compute_framing_from_positions(&scene.positions);
    let mesh = TetMesh::create_box_cage(framing.hero_min, framing.hero_max);
    let cages = CageScene::auto_bind(
        vec![Cage::new("hero", mesh, [0.0, 1.0, 1.0, 0.5])],
        &scene.positions,
    );

    let bindings = cages
        .bindings
        .iter()
        .map(|binding| {
            if binding.is_bound() {
                GaussianBinding::new(binding.tet, binding.weights)
            } else {
                GaussianBinding::unbound()
            }
        })
        .collect();

    let camera = CameraUniforms::from_look_at(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 3.0),
        Vec3::Y,
        45.0f32.to_radians(),
        WIDTH as f32,
        HEIGHT as f32,
        0.2,
    );

    let compositor =
        CompositorUniforms::new(WIDTH, HEIGHT, [0.05, 0.05, 0.05, 1.0], [0.0, 0.0, 0.0], 3);

    let uniforms = DeformationUniforms::new(
        [0.0, 0.0, 0.0],
        3,
        scene.count as u32,
        cages.cages.first().map(|c| c.mesh.num_tets()).unwrap_or(0) as u32,
    );

    Fixture {
        scene,
        cages,
        bindings,
        uniforms,
        camera,
        compositor,
    }
}

fn bench_stages(c: &mut Criterion) {
    let fixture = fixture();
    let deformed = cinder_gs::deformation::deform_cage_scene_cpu(
        &fixture.scene,
        &fixture.cages,
        &fixture.uniforms,
    );
    let projection = project_scene_cpu(&deformed.scene, &fixture.camera);
    let binning = projection.bin_and_sort(WIDTH, HEIGHT);

    let mut group = c.benchmark_group("cpu_stages");
    group.sample_size(10);

    let cage = fixture.cages.cages.first().expect("fixture cage");
    let vertices = cage.vertices();
    group.bench_function("stage1_deformation_20k", |b| {
        b.iter(|| {
            black_box(cinder_gs::deformation::deform_scene_cpu(
                &fixture.scene,
                &cage.mesh,
                &fixture.bindings,
                &vertices,
                &fixture.uniforms,
            ))
        })
    });

    group.bench_function("stage2_projection_20k_720p", |b| {
        b.iter(|| black_box(project_scene_cpu(&deformed.scene, &fixture.camera)))
    });

    group.bench_function("stage3_binning_20k_720p", |b| {
        b.iter(|| black_box(projection.bin_and_sort(WIDTH, HEIGHT)))
    });

    group.bench_function("stage4_compositor_720p", |b| {
        b.iter(|| {
            black_box(rasterize_scene_cpu(
                &projection.points_xy,
                &projection.conics,
                &deformed.scene.opacities,
                &deformed.scene.sh_coeffs,
                &deformed.scene.positions,
                &binning.sorted_entries,
                &binning.tile_ranges,
                &fixture.compositor,
            ))
        })
    });

    group.bench_function("end_to_end_cpu_720p", |b| {
        b.iter(|| {
            black_box(cinder_gs::render::render_scene_cpu(
                &deformed.scene,
                &fixture.camera,
                &fixture.compositor,
            ))
        })
    });

    group.finish();
}

criterion_group!(benches, bench_stages);
criterion_main!(benches);
