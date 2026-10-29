//! Automatic scene framing.
//!
//! Captures routinely contain a few distant background splats (sky, stray
//! floaters, room geometry). A raw axis-aligned bounding box would frame all of
//! them, pushing the actual subject to a handful of pixels. We instead use the
//! 2nd/98th percentile of each axis to reject outliers, then derive the hero cage
//! and camera distance from the trimmed box.

use crate::scene::GaussianSceneSoa;

/// Fraction of samples rejected at each end of each axis.
pub const OUTLIER_PERCENTILE: f32 = 0.02;

/// Maximum number of samples used to estimate the percentiles.
const MAX_SAMPLES: usize = 200_000;

/// Automatic framing derived from a scene's position distribution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SceneFraming {
    /// Centre of the trimmed bounding box.
    pub center: [f32; 3],
    /// Inclusive lower corner of the hero cage.
    pub hero_min: [f32; 3],
    /// Inclusive upper corner of the hero cage.
    pub hero_max: [f32; 3],
    /// Suggested orbit distance.
    pub camera_distance: f32,
}

impl SceneFraming {
    /// Fallback framing for an empty scene.
    pub const EMPTY: Self = Self {
        center: [0.0; 3],
        hero_min: [-0.5; 3],
        hero_max: [0.5; 3],
        camera_distance: 2.5,
    };

    /// Size of the hero box along each axis.
    pub fn hero_extent(&self) -> [f32; 3] {
        [
            self.hero_max[0] - self.hero_min[0],
            self.hero_max[1] - self.hero_min[1],
            self.hero_max[2] - self.hero_min[2],
        ]
    }
}

fn percentile(sorted: &[f32], fraction: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() as f32 - 1.0) * fraction.clamp(0.0, 1.0)).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

/// Computes framing from flattened `[N * 3]` positions.
pub fn compute_framing_from_positions(positions: &[f32]) -> SceneFraming {
    let count = positions.len() / 3;
    if count == 0 {
        return SceneFraming::EMPTY;
    }

    let step = (count / MAX_SAMPLES).max(1);
    let sample_count = count.div_ceil(step);

    let mut xs = Vec::with_capacity(sample_count);
    let mut ys = Vec::with_capacity(sample_count);
    let mut zs = Vec::with_capacity(sample_count);

    for i in (0..count).step_by(step) {
        let x = positions[i * 3];
        let y = positions[i * 3 + 1];
        let z = positions[i * 3 + 2];
        if x.is_finite() && y.is_finite() && z.is_finite() {
            xs.push(x);
            ys.push(y);
            zs.push(z);
        }
    }

    if xs.is_empty() {
        return SceneFraming::EMPTY;
    }

    xs.sort_unstable_by(f32::total_cmp);
    ys.sort_unstable_by(f32::total_cmp);
    zs.sort_unstable_by(f32::total_cmp);

    let low = [
        percentile(&xs, OUTLIER_PERCENTILE),
        percentile(&ys, OUTLIER_PERCENTILE),
        percentile(&zs, OUTLIER_PERCENTILE),
    ];
    let high = [
        percentile(&xs, 1.0 - OUTLIER_PERCENTILE),
        percentile(&ys, 1.0 - OUTLIER_PERCENTILE),
        percentile(&zs, 1.0 - OUTLIER_PERCENTILE),
    ];

    let mut extent = [high[0] - low[0], high[1] - low[1], high[2] - low[2]];
    // Degenerate axes (e.g. a planar capture) still need a manipulable cage.
    let largest = extent.iter().copied().fold(0.0f32, f32::max);
    let minimum = (largest * 0.01).max(1e-3);
    for value in &mut extent {
        if *value < minimum {
            *value = minimum;
        }
    }

    let center = [
        0.5 * (low[0] + high[0]),
        0.5 * (low[1] + high[1]),
        0.5 * (low[2] + high[2]),
    ];

    // 5% padding keeps surface splats inside the cage.
    let padding = [extent[0] * 0.05, extent[1] * 0.05, extent[2] * 0.05];
    let hero_min = [
        low[0] - padding[0],
        low[1] - padding[1],
        low[2] - padding[2],
    ];
    let hero_max = [
        high[0] + padding[0],
        high[1] + padding[1],
        high[2] + padding[2],
    ];

    let diagonal = (extent[0] * extent[0] + extent[1] * extent[1] + extent[2] * extent[2])
        .sqrt()
        .max(0.1);
    let camera_distance = (diagonal * 1.5).clamp(0.5, 5000.0);

    SceneFraming {
        center,
        hero_min,
        hero_max,
        camera_distance,
    }
}

/// Computes framing directly from a scene.
pub fn compute_scene_framing(scene: &GaussianSceneSoa) -> SceneFraming {
    compute_framing_from_positions(&scene.positions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentiles_reject_distant_outliers() {
        // Dense cluster in [0, 1]^3 plus extreme outliers.
        let mut positions = Vec::new();
        for i in 0..1000 {
            let t = i as f32 / 1000.0;
            positions.extend_from_slice(&[t, t * 0.5, 1.0 - t]);
        }
        positions.extend_from_slice(&[1000.0, 0.0, 0.0]);
        positions.extend_from_slice(&[-1000.0, 0.0, 0.0]);
        positions.extend_from_slice(&[0.0, 1000.0, 0.0]);
        positions.extend_from_slice(&[0.0, -1000.0, 0.0]);

        let framing = compute_framing_from_positions(&positions);
        assert!(
            framing.hero_min[0] > -0.2 && framing.hero_max[0] < 1.2,
            "x hero bounds must ignore outliers: {:?}..{:?}",
            framing.hero_min,
            framing.hero_max
        );
        assert!(framing.hero_min[1] > -0.2 && framing.hero_max[1] < 0.7);
        assert!(!framing.hero_min.iter().any(|v| v.abs() > 10.0));
        assert!(framing.camera_distance > 0.3 && framing.camera_distance < 10.0);
    }

    #[test]
    fn test_empty_scene_returns_defaults() {
        assert_eq!(compute_framing_from_positions(&[]), SceneFraming::EMPTY);
    }

    #[test]
    fn test_planar_scene_gets_a_thickness() {
        // All points on the z = 2 plane.
        let mut positions = Vec::new();
        for i in 0..100 {
            positions.extend_from_slice(&[i as f32 * 0.01, 0.0, 2.0]);
        }
        let framing = compute_framing_from_positions(&positions);
        let extent = framing.hero_extent();
        assert!(extent[2] > 0.0, "planar axis must get a finite thickness");
    }

    #[test]
    fn test_framing_from_scene_matches_positions() {
        let mut scene = GaussianSceneSoa::with_capacity(2);
        let sh = [0.0f32; 48];
        scene.push([0.0, 0.0, 0.0], [0.01, 0.0, 0.0, 0.01, 0.0, 0.01], 0.5, &sh);
        scene.push([1.0, 1.0, 1.0], [0.01, 0.0, 0.0, 0.01, 0.0, 0.01], 0.5, &sh);
        let a = compute_scene_framing(&scene);
        let b = compute_framing_from_positions(&scene.positions);
        assert_eq!(a, b);
    }
}
