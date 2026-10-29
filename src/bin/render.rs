//! Headless render farm CLI.
//!
//! ```text
//! cargo run --release --bin render -- \
//!     --input scene.ply --camera-path path.json --out frames/
//! ```
//!
//! Renders every frame offscreen (no window, no swapchain) and writes PNG files.
//! Without `--camera-path` it generates a one-revolution orbit around the
//! automatically framed subject.

use anyhow::{Result, anyhow};
use cinder_gs::deformation::{Cage, CageScene, TetMesh};
use cinder_gs::render::camera::CameraUniforms;
use cinder_gs::render::camera_path::CameraPath;
use cinder_gs::render::headless::HeadlessRenderer;
use cinder_gs::render::pipeline::ExecutionBackend;
use cinder_gs::render::rasterizer::CompositorUniforms;
use cinder_gs::scene::autoframe::{SceneFraming, compute_scene_framing};
use cinder_gs::scene::{GaussianSceneSoa, PlyLoader, fibonacci_cluster};
use glam::Vec3;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

struct Options {
    input: Option<PathBuf>,
    camera_path: Option<PathBuf>,
    output: PathBuf,
    width: Option<u32>,
    height: Option<u32>,
    fps: Option<u32>,
    frames: Option<u32>,
    backend: ExecutionBackend,
    cage: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            input: None,
            camera_path: None,
            output: PathBuf::from("frames"),
            width: None,
            height: None,
            fps: None,
            frames: None,
            backend: ExecutionBackend::CpuReference,
            cage: true,
        }
    }
}

fn parse_args() -> Result<Options> {
    let mut options = Options::default();
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
            "--camera-path" | "-c" => {
                options.camera_path = Some(PathBuf::from(take_value("--camera-path")?));
            }
            "--out" | "-o" => options.output = PathBuf::from(take_value("--out")?),
            "--width" => options.width = Some(take_value("--width")?.parse()?),
            "--height" => options.height = Some(take_value("--height")?.parse()?),
            "--fps" => options.fps = Some(take_value("--fps")?.parse()?),
            "--frames" => options.frames = Some(take_value("--frames")?.parse()?),
            "--backend" => {
                let value = take_value("--backend")?;
                options.backend = match value.as_str() {
                    "gpu" | "cubecl" | "wgpu" => ExecutionBackend::CubeClGpu,
                    "cpu" | "reference" => ExecutionBackend::CpuReference,
                    other => return Err(anyhow!("unknown backend '{other}' (use cpu or gpu)")),
                };
            }
            "--no-cage" => options.cage = false,
            "--cage" => options.cage = true,
            "--help" | "-h" => {
                println!(
                    "CinderGS headless renderer\n\
                     --input <scene.ply>        input PLY (synthetic scene when omitted)\n\
                     --camera-path <path.json>  camera keyframes\n\
                     --out <dir>                output frame directory\n\
                     --width/--height <px>      output resolution\n\
                     --fps <n> --frames <n>     frame count\n\
                     --backend <cpu|gpu>        execution backend\n\
                     --no-cage                  disable cage deformation"
                );
                std::process::exit(0);
            }
            other if !other.starts_with("--") => options.input = Some(PathBuf::from(other)),
            other => return Err(anyhow!("unknown option '{other}'")),
        }
        index += 1;
    }

    Ok(options)
}

fn load_scene(input: &Option<PathBuf>) -> Result<GaussianSceneSoa> {
    match input {
        Some(path) if path.exists() => {
            println!("Loading '{}'...", path.display());
            let timer = Instant::now();
            let scene = PlyLoader::load_file(path)?;
            println!(
                "Loaded {} Gaussians in {:.2} ms",
                scene.len(),
                timer.elapsed().as_secs_f64() * 1000.0
            );
            Ok(scene)
        }
        Some(path) => Err(anyhow!("input file '{}' does not exist", path.display())),
        None => {
            println!("No input specified; rendering a synthetic 50k splat cluster.");
            Ok(fibonacci_cluster(50_000, 1.0, [0.0, 0.0, 0.0]))
        }
    }
}

fn default_orbit(framing: &SceneFraming, duration: f32) -> CameraPath {
    let center = Vec3::from(framing.center);
    let distance = framing.camera_distance;
    let frames = 4;
    let keyframes = (0..=frames)
        .map(|i| {
            let t = i as f32 / frames as f32;
            let angle = t * std::f32::consts::TAU;
            let eye = center
                + Vec3::new(
                    angle.sin() * distance,
                    0.25 * distance,
                    angle.cos() * distance,
                );
            cinder_gs::render::camera_path::CameraKeyframe {
                time: t * duration,
                eye: eye.to_array(),
                target: center.to_array(),
                up: None,
                fov_y_degrees: None,
            }
        })
        .collect();

    CameraPath {
        width: None,
        height: None,
        fps: None,
        frames: None,
        keyframes,
    }
}

fn main() -> Result<()> {
    let options = parse_args()?;
    let scene = load_scene(&options.input)?;
    if scene.count == 0 {
        return Err(anyhow!("scene contains no Gaussians"));
    }

    let framing = compute_scene_framing(&scene);
    let mut cages = if options.cage {
        let mesh = TetMesh::create_box_cage(framing.hero_min, framing.hero_max);
        let cages = CageScene::auto_bind(
            vec![Cage::new("hero", mesh, [0.0, 0.85, 1.0, 0.45])],
            &scene.positions,
        );
        println!(
            "Hero cage bound to {}/{} Gaussians",
            cages.bound_count(),
            scene.len()
        );
        cages
    } else {
        CageScene::default()
    };

    let path = match &options.camera_path {
        Some(path) => {
            println!("Loading camera path '{}'...", path.display());
            CameraPath::load(path)?
        }
        None => default_orbit(&framing, 4.0),
    };

    let width = options.width.or(path.width).unwrap_or(960).max(1);
    let height = options.height.or(path.height).unwrap_or(540).max(1);
    let fps = options.fps.or(path.fps).unwrap_or(30).max(1);
    let frame_count = options
        .frames
        .or(path.frames)
        .unwrap_or_else(|| path.frame_count(fps))
        .max(1);

    fs::create_dir_all(&options.output)?;

    let mut renderer = HeadlessRenderer::new(width, height, options.backend)?;
    if let Some(description) = renderer.gpu_description() {
        println!("Compute device: {description}");
    }
    println!(
        "Rendering {frame_count} frames at {width}x{height} @ {fps} FPS to '{}'",
        options.output.display()
    );

    let start = Instant::now();

    for frame_index in 0..frame_count {
        let time = frame_index as f32 / fps as f32;
        let keyframe = path.sample(time);

        let eye = Vec3::from(keyframe.eye);
        let target = Vec3::from(keyframe.target);
        let up = Vec3::from(keyframe.up.unwrap_or([0.0, 1.0, 0.0]));
        let fov = keyframe.fov_y_degrees.unwrap_or(45.0).to_radians();

        let camera =
            CameraUniforms::from_look_at(eye, target, up, fov, width as f32, height as f32, 0.2);
        let compositor =
            CompositorUniforms::new(width, height, [0.05, 0.05, 0.07, 1.0], eye.to_array(), 3);

        let result = renderer.render(
            &scene,
            if cages.is_empty() { None } else { Some(&cages) },
            &camera,
            &compositor,
            false,
            None,
        )?;

        let path = options.output.join(format!("frame_{frame_index:04}.png"));
        image::save_buffer(
            &path,
            &result.image.rgba8,
            result.image.width,
            result.image.height,
            image::ExtendedColorType::Rgba8,
        )?;

        if frame_index % 10 == 0 || frame_index + 1 == frame_count {
            println!(
                "[{:4}/{}] t={:.2}s splats={} tiles={}",
                frame_index + 1,
                frame_count,
                time,
                result.stats.rendered_gaussians,
                result.stats.active_tiles
            );
        }

        // Reset any spring drift so a repeated run is deterministic.
        cages.reset();
    }

    let elapsed = start.elapsed();
    println!(
        "Done: {} frames in {:.2}s ({:.2} ms/frame)",
        frame_count,
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1000.0 / frame_count as f64
    );

    Ok(())
}
