//! Keyframe timeline for camera and cage-deformation playback.

use glam::Vec3;

/// Camera pose stored in a keyframe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraPose {
    pub target: [f32; 3],
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
}

/// A timeline keyframe. Cage offsets are displacements from each cage's rest pose,
/// flattened across cages in cage order.
#[derive(Clone, Debug, PartialEq)]
pub struct Keyframe {
    pub time: f32,
    pub camera: CameraPose,
    pub cage_offsets: Vec<[f32; 3]>,
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
}
