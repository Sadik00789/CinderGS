//! Camera-path JSON used by the headless render farm.

use glam::Vec3;
use serde::{Deserialize, Serialize};

/// One camera keyframe for a headless render.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CameraKeyframe {
    /// Time in seconds along the path.
    pub time: f32,
    /// Camera eye position in world space.
    pub eye: [f32; 3],
    /// Look-at target in world space.
    pub target: [f32; 3],
    /// Optional up vector (defaults to +Y).
    #[serde(default)]
    pub up: Option<[f32; 3]>,
    /// Optional vertical field of view in degrees (defaults to 45).
    #[serde(default)]
    pub fov_y_degrees: Option<f32>,
}

/// A complete camera path plus output settings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CameraPath {
    /// Output width in pixels.
    #[serde(default)]
    pub width: Option<u32>,
    /// Output height in pixels.
    #[serde(default)]
    pub height: Option<u32>,
    /// Frames per second.
    #[serde(default)]
    pub fps: Option<u32>,
    /// Total frame count (defaults to `duration * fps`).
    #[serde(default)]
    pub frames: Option<u32>,
    /// Ordered keyframes.
    pub keyframes: Vec<CameraKeyframe>,
}

impl CameraPath {
    /// Parses a camera path from JSON text.
    pub fn from_json(text: &str) -> anyhow::Result<Self> {
        let path: Self = serde_json::from_str(text)?;
        if path.keyframes.is_empty() {
            anyhow::bail!("camera path contains no keyframes");
        }
        Ok(path)
    }

    /// Loads a camera path from a JSON file.
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|err| anyhow::anyhow!("failed to read {}: {err}", path.display()))?;
        Self::from_json(&text)
            .map_err(|err| anyhow::anyhow!("failed to parse {}: {err}", path.display()))
    }

    /// Duration covered by the keyframes, in seconds.
    pub fn duration(&self) -> f32 {
        self.keyframes
            .last()
            .map(|k| k.time)
            .unwrap_or(0.0)
            .max(0.0)
    }

    /// Resolves the frame count, defaulting to `duration * fps`.
    pub fn frame_count(&self, fps: u32) -> u32 {
        self.frames
            .unwrap_or_else(|| ((self.duration() * fps as f32).ceil() as u32).max(1))
    }

    /// Samples the path at `time` seconds with linear interpolation.
    pub fn sample(&self, time: f32) -> CameraKeyframe {
        if self.keyframes.is_empty() {
            return CameraKeyframe {
                time: 0.0,
                eye: [0.0, 0.0, -3.0],
                target: [0.0, 0.0, 0.0],
                up: None,
                fov_y_degrees: None,
            };
        }

        let first = &self.keyframes[0];
        let last = &self.keyframes[self.keyframes.len() - 1];

        if time <= first.time {
            return first.clone();
        }
        if time >= last.time {
            return last.clone();
        }

        let upper = self
            .keyframes
            .partition_point(|keyframe| keyframe.time <= time);
        let a = &self.keyframes[upper - 1];
        let b = &self.keyframes[upper];
        let span = (b.time - a.time).max(1e-6);
        let t = ((time - a.time) / span).clamp(0.0, 1.0);

        let eye = Vec3::from(a.eye).lerp(Vec3::from(b.eye), t);
        let target = Vec3::from(a.target).lerp(Vec3::from(b.target), t);
        let up_a = a.up.unwrap_or([0.0, 1.0, 0.0]);
        let up_b = b.up.unwrap_or([0.0, 1.0, 0.0]);
        let up = Vec3::from(up_a)
            .lerp(Vec3::from(up_b), t)
            .normalize_or_zero();
        let fov_a = a.fov_y_degrees.unwrap_or(45.0);
        let fov_b = b.fov_y_degrees.unwrap_or(45.0);

        CameraKeyframe {
            time,
            eye: eye.to_array(),
            target: target.to_array(),
            up: Some(up.to_array()),
            fov_y_degrees: Some(fov_a + (fov_b - fov_a) * t),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_path() -> CameraPath {
        CameraPath {
            width: Some(320),
            height: Some(180),
            fps: Some(30),
            frames: None,
            keyframes: vec![
                CameraKeyframe {
                    time: 0.0,
                    eye: [0.0, 0.0, -5.0],
                    target: [0.0, 0.0, 0.0],
                    up: None,
                    fov_y_degrees: Some(45.0),
                },
                CameraKeyframe {
                    time: 2.0,
                    eye: [10.0, 0.0, -5.0],
                    target: [2.0, 0.0, 0.0],
                    up: None,
                    fov_y_degrees: Some(60.0),
                },
            ],
        }
    }

    #[test]
    fn test_sample_interpolates_midpoint() {
        let path = sample_path();
        let sample = path.sample(1.0);
        assert!((sample.eye[0] - 5.0).abs() < 1e-5);
        assert!((sample.target[0] - 1.0).abs() < 1e-5);
        assert!((sample.fov_y_degrees.unwrap() - 52.5).abs() < 1e-5);
    }

    #[test]
    fn test_sample_clamps_outside_range() {
        let path = sample_path();
        assert_eq!(path.sample(-1.0), path.keyframes[0]);
        assert_eq!(path.sample(100.0), path.keyframes[1]);
    }

    #[test]
    fn test_frame_count_and_duration() {
        let path = sample_path();
        assert_eq!(path.duration(), 2.0);
        assert_eq!(path.frame_count(30), 60);
    }

    #[test]
    fn test_json_roundtrip() {
        let path = sample_path();
        let json = serde_json::to_string(&path).expect("serialize");
        let parsed = CameraPath::from_json(&json).expect("parse");
        assert_eq!(parsed, path);
    }

    #[test]
    fn test_empty_path_is_rejected() {
        assert!(CameraPath::from_json("{\"keyframes\":[]}").is_err());
    }
}
