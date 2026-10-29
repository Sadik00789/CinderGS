use cinder_gs::app::{CinderApp, HERO_CAGE_COLOR, OrbitCamera};
use cinder_gs::deformation::{Cage, CageScene, TetMesh};
use cinder_gs::scene::autoframe::compute_scene_framing;
use cinder_gs::scene::{PlyLoader, fibonacci_cluster};
use glam::Vec3;
use std::path::Path;
use winit::event_loop::EventLoop;

fn main() -> anyhow::Result<()> {
    let cli_path = std::env::args().nth(1);

    let scene = match cli_path {
        Some(path_str) => {
            let path = Path::new(&path_str);
            if !path.exists() {
                eprintln!(
                    "warning: '{}' does not exist; falling back to the procedural cluster",
                    path_str
                );
                fibonacci_cluster(2000, 0.4, [0.0, 0.0, 0.0])
            } else {
                println!(
                    "Ingesting 3DGS binary PLY from '{}' via the schema-driven parser...",
                    path.display()
                );
                let timer = std::time::Instant::now();
                let loaded = PlyLoader::load_file(path)?;
                println!(
                    "Loaded {} Gaussians in {:.2} ms",
                    loaded.len(),
                    timer.elapsed().as_secs_f64() * 1000.0
                );
                loaded
            }
        }
        None => {
            println!("No PLY specified; generating a procedural cluster (2,000 Gaussians).");
            fibonacci_cluster(2000, 0.4, [0.0, 0.0, 0.0])
        }
    };

    // Percentile framing replaces the old hard-coded "bonsai" translation: the
    // hero cage and orbit distance follow the actual capture.
    let framing = compute_scene_framing(&scene);
    let cage = TetMesh::create_box_cage(framing.hero_min, framing.hero_max);
    println!(
        "Auto-framed hero cage: {:?} .. {:?} (centre {:?}, distance {:.2})",
        framing.hero_min, framing.hero_max, framing.center, framing.camera_distance
    );

    let cages = CageScene::auto_bind(
        vec![Cage::new("hero", cage, HERO_CAGE_COLOR)],
        &scene.positions,
    );
    println!(
        "Bound {}/{} Gaussians to the hero cage",
        cages.bound_count(),
        scene.len()
    );

    let event_loop = EventLoop::new()?;
    let mut app = CinderApp::new(scene, cages);
    app.camera = OrbitCamera::new(Vec3::from(framing.center), framing.camera_distance);
    app.camera.pitch = -0.12;

    println!("Starting CinderGS engine window...");
    event_loop.run_app(&mut app)?;

    Ok(())
}
