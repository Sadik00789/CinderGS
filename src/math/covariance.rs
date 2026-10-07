use glam::{Mat3, Quat, Vec3};

/// Hard lower bound for activated scales to prevent singular covariance matrices.
pub const MIN_SCALE: f32 = 1e-4;

/// Lower bound threshold for quaternion norm fallback.
pub const EPSILON_QUAT_NORM: f32 = 1e-6;

/// Opacity activation using standard sigmoid function:
///
/// $$\alpha = \frac{1.0}{1.0 + \exp(-\text{raw\_opacity})}$$
#[inline]
pub fn activate_opacity(raw_opacity: f32) -> f32 {
    1.0 / (1.0 + (-raw_opacity).exp())
}

/// Scale activation using exponential function with strict hard clamping:
///
/// $$s_i = \max(\exp(\text{raw\_scale}_i), 1e-4) \quad \text{for } i \in \{0, 1, 2\}$$
#[inline]
pub fn activate_scale(raw_scale: [f32; 3]) -> Vec3 {
    Vec3::new(
        raw_scale[0].exp().max(MIN_SCALE),
        raw_scale[1].exp().max(MIN_SCALE),
        raw_scale[2].exp().max(MIN_SCALE),
    )
}

/// Rotation activation and normalization using WXYZ convention from PLY:
/// - PLY layout stores: `rot_0 = w`, `rot_1 = x`, `rot_2 = y`, `rot_3 = z`.
/// - Constructs quaternion: `glam::Quat::from_xyzw(rot_1, rot_2, rot_3, rot_0)`.
/// - Calculates Euclidean norm $L = \sqrt{x^2 + y^2 + z^2 + w^2}$.
/// - If $L < 1e-6$, falls back to identity: `glam::Quat::IDENTITY`.
/// - Otherwise, normalizes: $q = q / L$.
#[inline]
pub fn activate_rotation(raw_rot: [f32; 4]) -> Quat {
    let w = raw_rot[0];
    let x = raw_rot[1];
    let y = raw_rot[2];
    let z = raw_rot[3];

    let norm_sq = x * x + y * y + z * z + w * w;
    let norm = norm_sq.sqrt();

    if norm < EPSILON_QUAT_NORM {
        Quat::IDENTITY
    } else {
        let inv_norm = 1.0 / norm;
        Quat::from_xyzw(x * inv_norm, y * inv_norm, z * inv_norm, w * inv_norm)
    }
}

/// Computes the full 3D symmetric covariance matrix $\Sigma$:
///
/// - Rotation matrix: $R = \text{Mat3::from\_quat}(q)$
/// - Scale matrix: $S = \text{Mat3::from\_diagonal}(s)$
/// - Intermediate transform: $M = R \cdot S$
/// - Covariance: $\Sigma = M \cdot M^T$
#[inline]
pub fn compute_covariance_3d_matrix(scale: Vec3, rotation: Quat) -> Mat3 {
    let r = Mat3::from_quat(rotation);
    let s = Mat3::from_diagonal(scale);
    let m = r * s;
    m * m.transpose()
}

/// Extracts the 6 symmetric upper-triangular components of $\Sigma$:
///
/// `cov[0] = Σ[0][0]` (xx)
/// `cov[1] = Σ[0][1]` (xy)
/// `cov[2] = Σ[0][2]` (xz)
/// `cov[3] = Σ[1][1]` (yy)
/// `cov[4] = Σ[1][2]` (yz)
/// `cov[5] = Σ[2][2]` (zz)
#[inline]
pub fn extract_covariance_3d(sigma: Mat3) -> [f32; 6] {
    [
        sigma.x_axis.x, // xx
        sigma.x_axis.y, // xy (symmetric: sigma.x_axis.y == sigma.y_axis.x)
        sigma.x_axis.z, // xz (symmetric: sigma.x_axis.z == sigma.z_axis.x)
        sigma.y_axis.y, // yy
        sigma.y_axis.z, // yz (symmetric: sigma.y_axis.z == sigma.z_axis.y)
        sigma.z_axis.z, // zz
    ]
}

/// Computes the 6 symmetric covariance components directly from scale and rotation:
///
/// Returns `[xx, xy, xz, yy, yz, zz]`.
#[inline]
pub fn compute_covariance_3d(scale: Vec3, rotation: Quat) -> [f32; 6] {
    let sigma = compute_covariance_3d_matrix(scale, rotation);
    extract_covariance_3d(sigma)
}

/// Reconstructs the 3x3 symmetric matrix from the 6 components:
/// `[xx, xy, xz, yy, yz, zz]`.
#[inline]
pub fn covariance_matrix_from_slice(cov: &[f32; 6]) -> Mat3 {
    Mat3::from_cols(
        Vec3::new(cov[0], cov[1], cov[2]), // col 0: [xx, xy, xz]
        Vec3::new(cov[1], cov[3], cov[4]), // col 1: [xy, yy, yz]
        Vec3::new(cov[2], cov[4], cov[5]), // col 2: [xz, yz, zz]
    )
}

/// Computes the determinant of a 3x3 symmetric covariance matrix directly from its 6 components:
///
/// $$\det(\Sigma) = xx(yy \cdot zz - yz^2) - xy(xy \cdot zz - xz \cdot yz) + xz(xy \cdot yz - xz \cdot yy)$$
///
/// Uses double precision internally to prevent catastrophic cancellation on ill-conditioned matrices.
#[inline]
pub fn covariance_determinant(cov: &[f32; 6]) -> f32 {
    let xx = cov[0] as f64;
    let xy = cov[1] as f64;
    let xz = cov[2] as f64;
    let yy = cov[3] as f64;
    let yz = cov[4] as f64;
    let zz = cov[5] as f64;

    let det = xx * (yy * zz - yz * yz) - xy * (xy * zz - xz * yz) + xz * (xy * yz - xz * yy);
    det as f32
}

/// Verifies positive-definiteness of a 3x3 symmetric matrix via Sylvester's criterion:
/// 1. Leading principal minor 1: $\Sigma_{xx} > 0$
/// 2. Leading principal minor 2: $\Sigma_{xx}\Sigma_{yy} - \Sigma_{xy}^2 > 0$
/// 3. Leading principal minor 3: $\det(\Sigma) > 0$
#[inline]
pub fn is_positive_definite(cov: &[f32; 6]) -> bool {
    let xx = cov[0] as f64;
    let xy = cov[1] as f64;
    let xz = cov[2] as f64;
    let yy = cov[3] as f64;
    let yz = cov[4] as f64;
    let zz = cov[5] as f64;

    // Minor 1x1
    if xx <= 0.0 {
        return false;
    }

    // Minor 2x2
    let minor_2 = xx * yy - xy * xy;
    if minor_2 <= 0.0 {
        return false;
    }

    // Minor 3x3 (determinant)
    let det = xx * (yy * zz - yz * yz) - xy * (xy * zz - xz * yz) + xz * (xy * yz - xz * yy);
    det > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_length_quaternion_fallback() {
        // Zero-length quaternion [rot_0, rot_1, rot_2, rot_3] = [0, 0, 0, 0]
        let raw_zero = [0.0, 0.0, 0.0, 0.0];
        let q_zero = activate_rotation(raw_zero);
        assert_eq!(q_zero, Quat::IDENTITY);
        assert_eq!(q_zero.x, 0.0);
        assert_eq!(q_zero.y, 0.0);
        assert_eq!(q_zero.z, 0.0);
        assert_eq!(q_zero.w, 1.0);

        // Sub-epsilon length (< 1e-6)
        let raw_sub_eps = [1e-7, 0.0, 0.0, 0.0];
        let q_sub_eps = activate_rotation(raw_sub_eps);
        assert_eq!(q_sub_eps, Quat::IDENTITY);

        // Valid non-zero quaternion with WXYZ ordering: rot_0=w, rot_1=x, rot_2=y, rot_3=z
        // e.g. w=0, x=1, y=0, z=0
        let raw_x = [0.0, 2.0, 0.0, 0.0];
        let q_x = activate_rotation(raw_x);
        assert!((q_x.x - 1.0).abs() < 1e-6);
        assert_eq!(q_x.y, 0.0);
        assert_eq!(q_x.z, 0.0);
        assert_eq!(q_x.w, 0.0);
    }

    #[test]
    fn test_scale_clamping_on_extreme_negative_values() {
        // Extreme negative values e.g. -50.0, -100.0, -1000.0
        let raw_scales = [-50.0, -100.0, -1000.0];
        let activated = activate_scale(raw_scales);

        assert_eq!(activated.x, MIN_SCALE);
        assert_eq!(activated.y, MIN_SCALE);
        assert_eq!(activated.z, MIN_SCALE);
        assert_eq!(activated.x, 1e-4);
        assert_eq!(activated.y, 1e-4);
        assert_eq!(activated.z, 1e-4);

        // Positive and unconstrained values
        let raw_normal = [0.0, 1.0, -2.0];
        let normal_act = activate_scale(raw_normal);
        assert!((normal_act.x - 1.0).abs() < 1e-6);
        assert!((normal_act.y - std::f32::consts::E).abs() < 1e-5);
        assert!((normal_act.z - (-2.0f32).exp()).abs() < 1e-6);
        assert!(normal_act.z > MIN_SCALE);
    }

    #[test]
    fn test_positive_definiteness_and_determinant() {
        // Test case 1: Minimum scale with identity rotation
        let scale_min = activate_scale([-50.0, -50.0, -50.0]);
        let rot_ident = activate_rotation([0.0, 0.0, 0.0, 0.0]);
        let cov_min = compute_covariance_3d(scale_min, rot_ident);

        let det_min = covariance_determinant(&cov_min);
        assert!(det_min > 0.0, "det(Sigma) must be > 0, got {}", det_min);
        assert!(is_positive_definite(&cov_min), "Sigma must be positive definite");

        // Analytical determinant for diagonal matrix with s_i = 1e-4:
        // det = (1e-4)^2 * (1e-4)^2 * (1e-4)^2 = (1e-4)^6 = 1e-24
        let expected_det = (1e-4f32).powi(6);
        assert!(
            (det_min - expected_det).abs() / expected_det < 1e-3,
            "Expected det {}, got {}",
            expected_det,
            det_min
        );

        // Test case 2: Diverse rotation quaternions and well-conditioned scales
        let test_rotations = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
            [std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2, 0.0, 0.0],
            [-0.3, 0.8, -0.4, 0.2],
        ];

        let test_scales = [
            [0.0, 0.0, 0.0],
            [-1.0, 0.5, 1.0],
            [1.5, 0.5, 2.0],
            [-5.0, -5.0, -5.0],
        ];

        for raw_r in &test_rotations {
            for raw_s in &test_scales {
                let s = activate_scale(*raw_s);
                let q = activate_rotation(*raw_r);
                let cov = compute_covariance_3d(s, q);

                let det = covariance_determinant(&cov);
                assert!(det > 0.0, "Determinant {} must be positive for s={:?}, r={:?}", det, raw_s, raw_r);
                assert!(is_positive_definite(&cov), "Matrix must be positive definite for s={:?}, r={:?}", raw_s, raw_r);

                // Cross-check with Glam Mat3 determinant
                let mat = covariance_matrix_from_slice(&cov);
                let glam_det = mat.determinant();
                assert!(
                    (det - glam_det).abs() / det.max(1e-12) < 1e-3,
                    "Determinant mismatch: formula {} vs glam {}",
                    det,
                    glam_det
                );
            }
        }

        // Test case 3: Clamped extreme negative values produce positive-definite matrix
        let extreme_scales = [[-50.0, -50.0, -50.0], [-100.0, 0.0, -50.0]];
        for raw_s in &extreme_scales {
            let s = activate_scale(*raw_s);
            let q = activate_rotation([1.0, 0.0, 0.0, 0.0]);
            let cov = compute_covariance_3d(s, q);
            let det = covariance_determinant(&cov);
            assert!(det > 0.0, "det(Sigma) must be > 0 for extreme scale {:?}", raw_s);
            assert!(is_positive_definite(&cov));
        }
    }

    #[test]
    fn test_opacity_activation() {
        // Zero -> 0.5
        let op_zero = activate_opacity(0.0);
        assert!((op_zero - 0.5).abs() < 1e-6);

        // Large positive -> ~1.0
        let op_pos = activate_opacity(10.0);
        assert!(op_pos > 0.999);

        // Large negative -> ~0.0
        let op_neg = activate_opacity(-10.0);
        assert!(op_neg < 0.001);
    }
}
