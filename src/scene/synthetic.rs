//! Deterministic procedural scenes for demos, benchmarks, and tests.

use crate::scene::soa::GaussianSceneSoa;

/// Builds a Fibonacci-sphere cluster of `count` splats with rainbow DC colour.
pub fn fibonacci_cluster(count: usize, radius: f32, center: [f32; 3]) -> GaussianSceneSoa {
    let mut scene = GaussianSceneSoa::with_capacity(count);
    if count == 0 {
        return scene;
    }

    let golden_ratio = (1.0 + 5.0f32.sqrt()) / 2.0;
    const SH_C0: f32 = 0.28209479;

    for i in 0..count {
        let t = (i as f32 + 0.5) / count as f32;
        let r = radius * t.cbrt();
        let theta = (1.0 - 2.0 * t).clamp(-1.0, 1.0).acos();
        let phi = 2.0 * std::f32::consts::PI * (i as f32 / golden_ratio);

        let position = [
            center[0] + r * theta.sin() * phi.cos(),
            center[1] + r * theta.sin() * phi.sin(),
            center[2] + r * theta.cos(),
        ];

        let covariance = [0.0004, 0.0, 0.0, 0.0004, 0.0, 0.0004];
        let opacity = 0.88;

        let mut sh = [0.0f32; 48];
        let cr = 0.5 + 0.5 * (i as f32 * 0.02).sin();
        let cg = 0.5 + 0.5 * (i as f32 * 0.02 + 2.094).sin();
        let cb = 0.5 + 0.5 * (i as f32 * 0.02 + 4.188).sin();
        sh[0] = (cr - 0.5) / SH_C0;
        sh[1] = (cg - 0.5) / SH_C0;
        sh[2] = (cb - 0.5) / SH_C0;

        scene.push(position, covariance, opacity, &sh);
    }

    scene
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fibonacci_cluster_shape() {
        let scene = fibonacci_cluster(128, 1.0, [0.0, 1.0, 2.0]);
        assert_eq!(scene.len(), 128);
        scene.validate().expect("valid SoA");

        // Every splat stays within the requested radius of the centre.
        for i in 0..scene.len() {
            let p = scene.position(i);
            let dx = p[0] - 0.0;
            let dy = p[1] - 1.0;
            let dz = p[2] - 2.0;
            assert!((dx * dx + dy * dy + dz * dz).sqrt() <= 1.0 + 1e-4);
        }
    }

    #[test]
    fn test_empty_cluster() {
        assert_eq!(fibonacci_cluster(0, 1.0, [0.0; 3]).len(), 0);
    }
}
