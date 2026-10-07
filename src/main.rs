use cinder_gs::app::{CinderApp, OrbitCamera};
use cinder_gs::deformation::TetMesh;
use cinder_gs::scene::{GaussianSceneSoa, PlyLoader};
use glam::Vec3;
use std::path::Path;
use winit::event_loop::EventLoop;

fn generate_procedural_synthetic_cluster() -> (GaussianSceneSoa, TetMesh, Vec<cinder_gs::deformation::GaussianBinding>, Vec3, f32) {
    const NUM_SPLATS: usize = 2000;
    let mut scene = GaussianSceneSoa::with_capacity(NUM_SPLATS);

    let golden_ratio = (1.0 + 5.0f32.sqrt()) / 2.0;

    let mut min_pos = Vec3::splat(f32::INFINITY);
    let mut max_pos = Vec3::splat(f32::NEG_INFINITY);
    let mut positions = Vec::with_capacity(NUM_SPLATS);

    for i in 0..NUM_SPLATS {
        let t = (i as f32 + 0.5) / NUM_SPLATS as f32;
        // Fibonacci sphere distribution scaled inside radius 0.40
        let r = 0.40 * t.cbrt();
        let theta = (1.0 - 2.0 * t).clamp(-1.0, 1.0).acos();
        let phi = 2.0 * std::f32::consts::PI * (i as f32 / golden_ratio);

        let pos = Vec3::new(
            r * theta.sin() * phi.cos(),
            r * theta.sin() * phi.sin(),
            r * theta.cos(),
        );

        min_pos = min_pos.min(pos);
        max_pos = max_pos.max(pos);
        positions.push(pos.to_array());

        // Anisotropic diagonal covariance
        let cov = [
            0.0004, 0.0, 0.0,
            0.0004, 0.0,
            0.0004,
        ];
        let opacity = 0.88;

        // Vibrant rainbow spherical harmonics degree 0
        let mut sh = [0.0f32; 48];
        let cr = 0.5 + 0.5 * (i as f32 * 0.02).sin();
        let cg = 0.5 + 0.5 * (i as f32 * 0.02 + 2.094).sin();
        let cb = 0.5 + 0.5 * (i as f32 * 0.02 + 4.188).sin();

        // Convert RGB in [0, 1] to SH DC coefficient
        const SH_C0: f32 = 0.28209479;
        sh[0] = (cr - 0.5) / SH_C0;
        sh[1] = (cg - 0.5) / SH_C0;
        sh[2] = (cb - 0.5) / SH_C0;

        scene.push(pos.to_array(), cov, opacity, &sh);
    }

    // AABB Padding: center +- 0.55 * extent
    let center = (min_pos + max_pos) * 0.5;
    let extent = (max_pos - min_pos).max(Vec3::splat(0.05));
    let min_prime = center - 0.55 * extent;
    let max_prime = center + 0.55 * extent;

    let cage = TetMesh::create_box_cage(min_prime.to_array(), max_prime.to_array());
    let bindings = cage.bind_gaussians(&positions);

    (scene, cage, bindings, center, extent.length())
}

fn main() -> anyhow::Result<()> {
    let cli_path = std::env::args().nth(1);

    let (scene, cage, bindings, cam_target, cam_dist) = if let Some(path_str) = cli_path {
        let path = Path::new(&path_str);
        if !path.exists() {
            eprintln!("Warning: Specified PLY file '{}' does not exist. Falling back to procedural demo box.", path_str);
            generate_procedural_synthetic_cluster()
        } else {
            println!("Ingesting 3DGS binary PLY from '{}' via memory-mapped parser...", path.display());
            let load_timer = std::time::Instant::now();
            let loaded_scene = PlyLoader::load_file(path)?;
            let load_elapsed = load_timer.elapsed();
            println!("Loaded {} Gaussians in {:.2} ms", loaded_scene.len(), load_elapsed.as_secs_f64() * 1000.0);

            // Calculate scene AABB
            let mut min_pos = Vec3::splat(f32::INFINITY);
            let mut max_pos = Vec3::splat(f32::NEG_INFINITY);

            for i in 0..loaded_scene.count {
                let p = Vec3::new(
                    loaded_scene.positions[i * 3],
                    loaded_scene.positions[i * 3 + 1],
                    loaded_scene.positions[i * 3 + 2],
                );
                min_pos = min_pos.min(p);
                max_pos = max_pos.max(p);
            }

            if loaded_scene.count == 0 {
                min_pos = Vec3::splat(-0.5);
                max_pos = Vec3::splat(0.5);
            }

            // Padded AABB calculation
            let center = (min_pos + max_pos) * 0.5;
            let raw_extent = max_pos - min_pos;
            let extent = raw_extent.max(Vec3::splat(0.05));
            let min_prime = center - 0.55 * extent;
            let max_prime = center + 0.55 * extent;

            println!("Scene Center: {:?}, Extent: {:?}", center, extent);
            println!("Constructing 5-tetrahedron cage with padded bounds [min', max']...");
            let tet_cage = TetMesh::create_box_cage(min_prime.to_array(), max_prime.to_array());

            println!("Binding {} Gaussians using Rayon multi-threading...", loaded_scene.len());
            let bind_timer = std::time::Instant::now();
            let tet_bindings = tet_cage.bind_flat_positions(&loaded_scene.positions);
            let bind_elapsed = bind_timer.elapsed();
            println!("Bound all Gaussians in {:.2} ms", bind_elapsed.as_secs_f64() * 1000.0);

            (loaded_scene, tet_cage, tet_bindings, center, extent.length().max(1.0))
        }
    } else {
        println!("No PLY file specified. Generating procedural synthetic cluster (2,000 Gaussians in unit cage)...");
        generate_procedural_synthetic_cluster()
    };

    let event_loop = EventLoop::new()?;
    let mut app = CinderApp::new(scene, Some(cage), Some(bindings));
    app.camera = OrbitCamera::new(cam_target, cam_dist * 1.8);

    println!("Starting CinderGS engine window...");
    event_loop.run_app(&mut app)?;

    Ok(())
}
