//! Keyframe timeline for camera and cage-deformation playback.

use crate::deformation::cpu_reference::deform_cage_scene_cpu;
use crate::deformation::{CageScene, DeformationUniforms};
use crate::render::camera::CameraUniforms;
use crate::render::camera_path::{CameraKeyframe, CameraPath};
use crate::render::pipeline::RenderPipelineOrchestrator;
use crate::render::rasterizer::{CompositorUniforms, RasterizedImage};
use crate::scene::GaussianSceneSoa;
use crate::scene::ply_loader::{PlyError, PlyLoader};
use glam::Vec3;

/// Camera pose stored in a keyframe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraPose {
    pub target: [f32; 3],
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
}

impl CameraPose {
    /// Computes the world-space eye position matching `OrbitCamera::eye_position`.
    pub fn eye_position(&self) -> [f32; 3] {
        let cos_pitch = self.pitch.cos();
        let sin_pitch = self.pitch.sin();
        let cos_yaw = self.yaw.cos();
        let sin_yaw = self.yaw.sin();
        [
            self.target[0] + self.distance * cos_pitch * sin_yaw,
            self.target[1] + self.distance * sin_pitch,
            self.target[2] + self.distance * cos_pitch * cos_yaw,
        ]
    }
}

/// A timeline keyframe. Cage offsets are displacements from each cage's rest pose,
/// flattened across cages in cage order.
#[derive(Clone, Debug, PartialEq)]
pub struct Keyframe {
    pub time: f32,
    pub camera: CameraPose,
    pub cage_offsets: Vec<[f32; 3]>,
}

impl Keyframe {
    /// World-space eye position for this keyframe camera pose.
    #[inline]
    pub fn eye_position(&self) -> [f32; 3] {
        self.camera.eye_position()
    }
}

fn lerp_angle(a: f32, b: f32, t: f32) -> f32 {
    let mut delta = (b - a) % std::f32::consts::TAU;
    if delta > std::f32::consts::PI {
        delta -= std::f32::consts::TAU;
    } else if delta < -std::f32::consts::PI {
        delta += std::f32::consts::TAU;
    }
    a + delta * t
}

/// Ordered keyframes plus playback state.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Timeline {
    pub keyframes: Vec<Keyframe>,
    pub playhead: f32,
    pub playing: bool,
}

impl Timeline {
    /// Creates an empty timeline.
    pub fn new() -> Self {
        Self::default()
    }

    /// Duration of the timeline in seconds.
    pub fn duration(&self) -> f32 {
        self.keyframes.last().map(|key| key.time).unwrap_or(0.0)
    }

    /// Returns true when the timeline has at least one keyframe.
    pub fn is_active(&self) -> bool {
        !self.keyframes.is_empty()
    }

    /// Inserts (or replaces) a keyframe and keeps the list sorted by time.
    pub fn insert(&mut self, keyframe: Keyframe) {
        let time = keyframe.time.max(0.0);
        let keyframe = Keyframe { time, ..keyframe };

        if let Some(existing) = self
            .keyframes
            .iter_mut()
            .find(|existing| (existing.time - time).abs() < 1e-4)
        {
            *existing = keyframe;
        } else {
            self.keyframes.push(keyframe);
            self.keyframes.sort_by(|a, b| {
                a.time
                    .partial_cmp(&b.time)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        self.playhead = self.playhead.clamp(0.0, self.duration());
    }

    /// Removes all keyframes and stops playback.
    pub fn clear(&mut self) {
        self.keyframes.clear();
        self.playhead = 0.0;
        self.playing = false;
    }

    /// Advances playback by `dt` seconds, stopping at the end.
    pub fn advance(&mut self, dt: f32) {
        if !self.playing || self.keyframes.is_empty() {
            return;
        }
        self.playhead += dt.max(0.0);
        let duration = self.duration();
        if self.playhead >= duration {
            self.playhead = duration;
            self.playing = false;
        }
    }

    /// Samples the timeline at `time` with linear interpolation.
    pub fn sample(&self, time: f32) -> Option<Keyframe> {
        if self.keyframes.is_empty() {
            return None;
        }

        let first = &self.keyframes[0];
        let last = self.keyframes.last()?;
        if time <= first.time {
            return Some(first.clone());
        }
        if time >= last.time {
            return Some(last.clone());
        }

        let upper = self.keyframes.partition_point(|key| key.time <= time);
        let a = &self.keyframes[upper - 1];
        let b = &self.keyframes[upper];
        let span = (b.time - a.time).max(1e-6);
        let t = ((time - a.time) / span).clamp(0.0, 1.0);

        let target = Vec3::from(a.camera.target).lerp(Vec3::from(b.camera.target), t);
        let distance = a.camera.distance + (b.camera.distance - a.camera.distance) * t;
        let yaw = lerp_angle(a.camera.yaw, b.camera.yaw, t);
        let pitch = a.camera.pitch + (b.camera.pitch - a.camera.pitch) * t;

        let count = a.cage_offsets.len().min(b.cage_offsets.len());
        let mut cage_offsets = Vec::with_capacity(count);
        for i in 0..count {
            cage_offsets.push([
                a.cage_offsets[i][0] + (b.cage_offsets[i][0] - a.cage_offsets[i][0]) * t,
                a.cage_offsets[i][1] + (b.cage_offsets[i][1] - a.cage_offsets[i][1]) * t,
                a.cage_offsets[i][2] + (b.cage_offsets[i][2] - a.cage_offsets[i][2]) * t,
            ]);
        }

        Some(Keyframe {
            time,
            camera: CameraPose {
                target: target.to_array(),
                distance,
                yaw,
                pitch,
            },
            cage_offsets,
        })
    }

    /// Samples the current playhead.
    pub fn sample_current(&self) -> Option<Keyframe> {
        self.sample(self.playhead)
    }

    /// Samples `time` and writes interpolated cage displacements onto `cages`,
    /// returning the interpolated [`CameraPose`] if keyframes are present.
    pub fn apply_keyframe_to_cages(&self, time: f32, cages: &mut CageScene) -> Option<CameraPose> {
        let sample = self.sample(time)?;
        let mut offset_idx = 0usize;
        for cage in &mut cages.cages {
            for (vertex, rest) in cage
                .simulator
                .current_positions
                .iter_mut()
                .zip(cage.simulator.rest_positions.iter())
            {
                if let Some(delta) = sample.cage_offsets.get(offset_idx) {
                    *vertex = *rest + Vec3::from(*delta);
                }
                offset_idx += 1;
            }
            cage.simulator.velocities.fill(Vec3::ZERO);
        }
        Some(sample.camera)
    }

    /// Evaluates and returns a deformed [`GaussianSceneSoa`] snapshot at timeline `time`.
    pub fn deform_scene_at(
        &self,
        time: f32,
        rest_scene: &GaussianSceneSoa,
        cages: &CageScene,
        sh_degree: u32,
        split_threshold: f32,
        mip_filter_3d_var: f32,
    ) -> GaussianSceneSoa {
        let mut posed = cages.clone();
        let eye = self
            .apply_keyframe_to_cages(time, &mut posed)
            .map(|cam| cam.eye_position())
            .unwrap_or([0.0, 0.0, 4.0]);

        let total_tets: u32 = posed.cages.iter().map(|c| c.mesh.num_tets() as u32).sum();
        let uniforms =
            DeformationUniforms::new(eye, sh_degree.min(3), rest_scene.count as u32, total_tets)
                .with_strain_splitting(split_threshold)
                .with_mip_filter_3d(mip_filter_3d_var);

        deform_cage_scene_cpu(rest_scene, &posed, &uniforms).into_soa()
    }

    /// Serializes a deformed `.ply` snapshot at timeline `time` into binary PLY bytes.
    pub fn export_deformed_ply_bytes_at(
        &self,
        time: f32,
        rest_scene: &GaussianSceneSoa,
        cages: &CageScene,
        sh_degree: u32,
        split_threshold: f32,
        mip_filter_3d_var: f32,
    ) -> Vec<u8> {
        let snapshot = self.deform_scene_at(
            time,
            rest_scene,
            cages,
            sh_degree,
            split_threshold,
            mip_filter_3d_var,
        );
        PlyLoader::write_binary_ply_bytes(&snapshot)
    }

    /// Exports a deformed `.ply` snapshot at timeline `time` to `path`.
    #[cfg(not(target_arch = "wasm32"))]
    #[allow(clippy::too_many_arguments)]
    pub fn export_deformed_ply_at(
        &self,
        path: impl AsRef<std::path::Path>,
        time: f32,
        rest_scene: &GaussianSceneSoa,
        cages: &CageScene,
        sh_degree: u32,
        split_threshold: f32,
        mip_filter_3d_var: f32,
    ) -> Result<(), PlyError> {
        let snapshot = self.deform_scene_at(
            time,
            rest_scene,
            cages,
            sh_degree,
            split_threshold,
            mip_filter_3d_var,
        );
        PlyLoader::save_file(path, &snapshot)
    }

    /// Converts the timeline keyframes into a headless [`CameraPath`].
    pub fn to_camera_path(&self, fps: f32) -> anyhow::Result<CameraPath> {
        if self.keyframes.is_empty() {
            anyhow::bail!("cannot build CameraPath from an empty timeline");
        }
        let fps_u32 = fps.round().max(1.0) as u32;
        let keyframes = self
            .keyframes
            .iter()
            .map(|kf| CameraKeyframe {
                time: kf.time,
                eye: kf.eye_position(),
                target: kf.camera.target,
                up: Some([0.0, 1.0, 0.0]),
                fov_y_degrees: Some(45.0),
            })
            .collect();
        Ok(CameraPath {
            width: None,
            height: None,
            fps: Some(fps_u32),
            frames: None,
            keyframes,
        })
    }

    /// Deterministically renders all timeline frames via the CPU reference pipeline.
    #[allow(clippy::too_many_arguments)]
    pub fn render_deterministic_frames(
        &self,
        rest_scene: &GaussianSceneSoa,
        cages: &CageScene,
        width: u32,
        height: u32,
        fps: f32,
        sh_degree: u32,
        split_threshold: f32,
        mip_filter_3d_var: f32,
    ) -> Vec<RasterizedImage> {
        let width = width.max(1);
        let height = height.max(1);
        let duration = self.duration();
        let fps = fps.max(1.0);
        let total_frames = if duration <= 0.0 {
            1usize
        } else {
            ((duration * fps).round() as usize).max(2)
        };

        let mut frames = Vec::with_capacity(total_frames);
        for frame_idx in 0..total_frames {
            let t = if total_frames <= 1 {
                self.playhead.max(0.0)
            } else {
                duration * (frame_idx as f32) / ((total_frames - 1) as f32)
            };

            let mut posed = cages.clone();
            let pose = self
                .apply_keyframe_to_cages(t, &mut posed)
                .unwrap_or(CameraPose {
                    target: [0.0, 0.0, 0.0],
                    distance: 4.0,
                    yaw: 0.0,
                    pitch: 0.2,
                });
            let eye = pose.eye_position();
            let camera = CameraUniforms::from_look_at(
                Vec3::from(eye),
                Vec3::from(pose.target),
                Vec3::Y,
                45.0_f32.to_radians(),
                width as f32,
                height as f32,
                0.2,
            );
            let compositor =
                CompositorUniforms::new(width, height, [0.0, 0.0, 0.0, 1.0], eye, sh_degree.min(3));
            let (image, _, _, _, _) =
                RenderPipelineOrchestrator::execute_cpu_reference_with_options(
                    rest_scene,
                    Some(&posed),
                    &camera,
                    &compositor,
                    width,
                    height,
                    None,
                    split_threshold,
                    mip_filter_3d_var,
                );
            frames.push(image);
        }
        frames
    }

    /// Deterministically renders all timeline frames as packed `u32` RGBA scanlines.
    #[allow(clippy::too_many_arguments)]
    pub fn render_deterministic_sequence(
        &self,
        rest_scene: &GaussianSceneSoa,
        cages: &CageScene,
        width: u32,
        height: u32,
        fps: f32,
        sh_degree: u32,
        split_threshold: f32,
        mip_filter_3d_var: f32,
    ) -> Vec<Vec<u32>> {
        self.render_deterministic_frames(
            rest_scene,
            cages,
            width,
            height,
            fps,
            sh_degree,
            split_threshold,
            mip_filter_3d_var,
        )
        .into_iter()
        .map(|img| {
            img.rgba8
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        })
        .collect()
    }

    /// Renders and saves deterministic timeline PNG frames (`frame_0000.png`, ...).
    #[cfg(not(target_arch = "wasm32"))]
    #[allow(clippy::too_many_arguments)]
    pub fn export_deterministic_renders(
        &self,
        out_dir: impl AsRef<std::path::Path>,
        rest_scene: &GaussianSceneSoa,
        cages: &CageScene,
        width: u32,
        height: u32,
        fps: f32,
        sh_degree: u32,
        split_threshold: f32,
        mip_filter_3d_var: f32,
    ) -> Result<Vec<std::path::PathBuf>, String> {
        let dir = out_dir.as_ref();
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("failed to create render directory {}: {e}", dir.display()))?;

        let frames = self.render_deterministic_frames(
            rest_scene,
            cages,
            width,
            height,
            fps,
            sh_degree,
            split_threshold,
            mip_filter_3d_var,
        );
        let mut paths = Vec::with_capacity(frames.len());
        for (idx, frame) in frames.iter().enumerate() {
            let path = dir.join(format!("frame_{idx:04}.png"));
            image::save_buffer(
                &path,
                &frame.rgba8,
                frame.width,
                frame.height,
                image::ExtendedColorType::Rgba8,
            )
            .map_err(|e| format!("failed to write {}: {e}", path.display()))?;
            paths.push(path);
        }
        Ok(paths)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pose(target: [f32; 3], distance: f32, yaw: f32) -> CameraPose {
        CameraPose {
            target,
            distance,
            yaw,
            pitch: 0.1,
        }
    }

    #[test]
    fn test_insert_keeps_order_and_replaces_duplicates() {
        let mut timeline = Timeline::new();
        timeline.insert(Keyframe {
            time: 2.0,
            camera: pose([0.0; 3], 5.0, 0.0),
            cage_offsets: Vec::new(),
        });
        timeline.insert(Keyframe {
            time: 0.0,
            camera: pose([1.0; 3], 3.0, 1.0),
            cage_offsets: Vec::new(),
        });
        timeline.insert(Keyframe {
            time: 2.0,
            camera: pose([2.0; 3], 8.0, 2.0),
            cage_offsets: Vec::new(),
        });

        assert_eq!(timeline.keyframes.len(), 2);
        assert_eq!(timeline.keyframes[0].time, 0.0);
        assert_eq!(timeline.keyframes[1].time, 2.0);
        assert_eq!(timeline.keyframes[1].camera.distance, 8.0);
    }

    #[test]
    fn test_sample_interpolates_camera_and_offsets() {
        let mut timeline = Timeline::new();
        timeline.insert(Keyframe {
            time: 0.0,
            camera: pose([0.0, 0.0, 0.0], 4.0, 0.0),
            cage_offsets: vec![[0.0, 0.0, 0.0]],
        });
        timeline.insert(Keyframe {
            time: 2.0,
            camera: pose([2.0, 0.0, 0.0], 8.0, 1.0),
            cage_offsets: vec![[2.0, 4.0, 6.0]],
        });

        let sample = timeline.sample(1.0).expect("sample");
        assert!((sample.camera.target[0] - 1.0).abs() < 1e-5);
        assert!((sample.camera.distance - 6.0).abs() < 1e-5);
        assert!((sample.camera.yaw - 0.5).abs() < 1e-5);
        assert!((sample.cage_offsets[0][1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_advance_stops_at_duration() {
        let mut timeline = Timeline::new();
        timeline.insert(Keyframe {
            time: 0.0,
            camera: pose([0.0; 3], 4.0, 0.0),
            cage_offsets: Vec::new(),
        });
        timeline.insert(Keyframe {
            time: 1.0,
            camera: pose([1.0; 3], 4.0, 0.0),
            cage_offsets: Vec::new(),
        });
        timeline.playing = true;
        timeline.advance(0.6);
        assert!((timeline.playhead - 0.6).abs() < 1e-5);
        assert!(timeline.playing);
        timeline.advance(1.0);
        assert_eq!(timeline.playhead, 1.0);
        assert!(!timeline.playing);
    }

    #[test]
    fn test_clear_resets_state() {
        let mut timeline = Timeline::new();
        timeline.insert(Keyframe {
            time: 1.0,
            camera: pose([0.0; 3], 4.0, 0.0),
            cage_offsets: Vec::new(),
        });
        timeline.playing = true;
        timeline.clear();
        assert!(timeline.keyframes.is_empty());
        assert!(!timeline.playing);
        assert_eq!(timeline.playhead, 0.0);
    }

    #[test]
    fn test_yaw_takes_shortest_path() {
        let mut timeline = Timeline::new();
        timeline.insert(Keyframe {
            time: 0.0,
            camera: pose([0.0; 3], 4.0, 3.0),
            cage_offsets: Vec::new(),
        });
        timeline.insert(Keyframe {
            time: 2.0,
            camera: pose([0.0; 3], 4.0, -3.0),
            cage_offsets: Vec::new(),
        });
        let mid = timeline.sample(1.0).expect("sample");
        // 3.0 -> -3.0 across the wrap point should pass through ±π, not through zero.
        assert!(mid.camera.yaw.abs() > 3.0, "yaw was {}", mid.camera.yaw);
    }

    #[test]
    fn test_deform_ply_export_and_deterministic_render() {
        use crate::deformation::{Cage, TetMesh};

        let mut scene = GaussianSceneSoa::with_capacity(1);
        scene.push(
            [0.0, 0.0, 0.0],
            [0.03, 0.0, 0.0, 0.03, 0.0, 0.03],
            0.9,
            &[0.6; 48],
        );

        let tet = TetMesh::create_box_cage([-1.0; 3], [1.0; 3]);
        let mut cages = CageScene::auto_bind(
            vec![Cage::new("box", tet, [0.2, 0.8, 1.0, 1.0])],
            &scene.positions,
        );
        let n_verts = cages.cages[0].mesh.num_vertices();

        let mut timeline = Timeline::new();
        timeline.insert(Keyframe {
            time: 0.0,
            camera: pose([0.0; 3], 3.0, 0.0),
            cage_offsets: vec![[0.0; 3]; n_verts],
        });
        timeline.insert(Keyframe {
            time: 0.5,
            camera: pose([0.0; 3], 3.0, 0.25),
            cage_offsets: vec![[0.5, 0.0, 0.0]; n_verts],
        });

        let ply_bytes = timeline.export_deformed_ply_bytes_at(0.5, &scene, &cages, 0, 0.0, 0.0);
        let reloaded = PlyLoader::load_from_bytes(&ply_bytes).expect("deformed snapshot reloads");
        assert_eq!(reloaded.count, 1);
        assert!((reloaded.positions[0] - 0.5).abs() < 1e-4);

        let path = timeline.to_camera_path(10.0).expect("camera path builds");
        assert_eq!(path.keyframes.len(), 2);

        let seq_a =
            timeline.render_deterministic_sequence(&scene, &cages, 32, 32, 4.0, 0, 0.0, 0.0);
        let seq_b =
            timeline.render_deterministic_sequence(&scene, &cages, 32, 32, 4.0, 0, 0.0, 0.0);
        assert_eq!(seq_a.len(), 2);
        assert_eq!(seq_a, seq_b);

        let _ = &mut cages;
    }
}
