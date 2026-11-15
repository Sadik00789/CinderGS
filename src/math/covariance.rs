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

/// Default 3D Mip-Splatting isotropic regularization variance ($s_{3\text{D}}$)
/// preventing under-sampling / aliasing when compressed below screen pixel scale.
pub const DEFAULT_MIP_FILTER_3D_VAR: f32 = 0.0025;

/// Default 2D EWA screen-space low-pass filter variance ($s_{2\text{D}}$) in squared pixels.
pub const DEFAULT_MIP_FILTER_2D_VAR: f32 = 0.3;

/// Inverse of sigmoid opacity activation: recovers raw logit from $\alpha \in (0, 1)$.
#[inline]
pub fn inverse_activate_opacity(alpha: f32) -> f32 {
    let clamped = alpha.clamp(1e-6, 1.0 - 1e-6);
    (clamped / (1.0 - clamped)).ln()
}

/// Inverse of exponential scale activation: recovers log-scale from positive scale vector.
#[inline]
pub fn inverse_activate_scale(scale: Vec3) -> Vec3 {
    Vec3::new(
        scale.x.max(MIN_SCALE).ln(),
        scale.y.max(MIN_SCALE).ln(),
        scale.z.max(MIN_SCALE).ln(),
    )
}

/// Computes all eigenvalues (sorted descending) and orthonormal eigenvectors
/// (forming a right-handed $\mathrm{SO}(3)$ basis) of a symmetric 3x3 matrix
/// packed as `[xx, xy, xz, yy, yz, zz]`.
pub fn symmetric_eigen_3x3(cov: &[f32; 6]) -> ([f32; 3], Mat3) {
    let mut a = [
        [cov[0] as f64, cov[1] as f64, cov[2] as f64],
        [cov[1] as f64, cov[3] as f64, cov[4] as f64],
        [cov[2] as f64, cov[4] as f64, cov[5] as f64],
    ];
    let mut v = [[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    for _sweep in 0..12 {
        for (p, q) in [(0usize, 1usize), (0, 2), (1, 2)] {
            let apq = a[p][q];
            if apq.abs() <= 1e-15 {
                continue;
            }
            let tau = (a[q][q] - a[p][p]) / (2.0 * apq);
            let t = if tau >= 0.0 {
                1.0 / (tau + (1.0 + tau * tau).sqrt())
            } else {
                -1.0 / (-tau + (1.0 + tau * tau).sqrt())
            };
            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = t * c;

            let app = a[p][p];
            let aqq = a[q][q];
            a[p][p] = app - t * apq;
            a[q][q] = aqq + t * apq;
            a[p][q] = 0.0;
            a[q][p] = 0.0;

            for r in [0usize, 1, 2] {
                if r != p && r != q {
                    let arp = a[r][p];
                    let arq = a[r][q];
                    a[r][p] = c * arp - s * arq;
                    a[p][r] = a[r][p];
                    a[r][q] = s * arp + c * arq;
                    a[q][r] = a[r][q];
                }
            }

            for row in &mut v {
                let vrp = row[p];
                let vrq = row[q];
                row[p] = c * vrp - s * vrq;
                row[q] = s * vrp + c * vrq;
            }
        }
    }

    let mut pairs = [
        (
            a[0][0],
            glam::DVec3::new(v[0][0], v[1][0], v[2][0]).normalize(),
        ),
        (
            a[1][1],
            glam::DVec3::new(v[0][1], v[1][1], v[2][1]).normalize(),
        ),
        (
            a[2][2],
            glam::DVec3::new(v[0][2], v[1][2], v[2][2]).normalize(),
        ),
    ];
    pairs.sort_by(|lhs, rhs| rhs.0.total_cmp(&lhs.0));

    let c0 = pairs[0].1.as_vec3();
    let c1 = pairs[1].1.as_vec3();
    let mut c2 = pairs[2].1.as_vec3();
    let mut basis = Mat3::from_cols(c0, c1, c2);
    if basis.determinant() < 0.0 {
        c2 = -c2;
        basis = Mat3::from_cols(c0, c1, c2);
    }

    (
        [pairs[0].0 as f32, pairs[1].0 as f32, pairs[2].0 as f32],
        basis,
    )
}

/// Returns the major principal unit direction and sorted descending eigenvalues
/// of a packed symmetric 3x3 covariance matrix.
#[inline]
pub fn principal_axis_and_eigenvalues(cov: &[f32; 6]) -> (Vec3, [f32; 3]) {
    let (evals, basis) = symmetric_eigen_3x3(cov);
    (basis.x_axis.normalize_or_zero(), evals)
}

/// Decomposes a symmetric 3D covariance matrix into principal scales and rotation quaternion.
#[inline]
pub fn decompose_covariance_3d(cov: &[f32; 6]) -> (Vec3, Quat) {
    let (evals, basis) = symmetric_eigen_3x3(cov);
    let min_var = MIN_SCALE * MIN_SCALE;
    let scale = Vec3::new(
        evals[0].max(min_var).sqrt(),
        evals[1].max(min_var).sqrt(),
        evals[2].max(min_var).sqrt(),
    );
    let rotation = Quat::from_mat3(&basis).normalize();
    (scale, rotation)
}

/// Applies the 3D Mip-Splatting isotropic smoothing filter:
/// $\Sigma_{3\text{D}}^{\text{reg}} = \Sigma_{3\text{D}} + s_{3\text{D}} I_3$,
/// paired with volume-conserving peak opacity compensation
/// $\alpha' = \alpha \sqrt{\det(\Sigma_{3\text{D}}) / \det(\Sigma_{3\text{D}}^{\text{reg}})}$.
pub fn apply_mip_filter_3d(cov: &[f32; 6], opacity: f32, filter_var_3d: f32) -> ([f32; 6], f32) {
    if filter_var_3d <= 0.0 {
        return (*cov, opacity);
    }
    let det_orig = covariance_determinant(cov).max(1e-24);
    let regularized = [
        cov[0] + filter_var_3d,
        cov[1],
        cov[2],
        cov[3] + filter_var_3d,
        cov[4],
        cov[5] + filter_var_3d,
    ];
    let det_reg = covariance_determinant(&regularized).max(det_orig);
    let comp = (det_orig / det_reg).clamp(0.0, 1.0).sqrt();
    (regularized, (opacity * comp).clamp(0.0, 1.0))
}

/// Applies the 2D EWA screen-space low-pass covariance filter:
/// $\Sigma_{2\text{D}}^{\text{reg}} = \Sigma_{2\text{D}} + s_{2\text{D}} I_2$,
/// paired with area-conserving opacity compensation
/// $\alpha' = \alpha \sqrt{\det(\Sigma_{2\text{D}}) / \det(\Sigma_{2\text{D}}^{\text{reg}})}$.
pub fn apply_ewa_low_pass_2d(
    cov_2d: [f32; 3],
    opacity: f32,
    filter_var_2d: f32,
) -> ([f32; 3], f32) {
    if filter_var_2d <= 0.0 {
        return (cov_2d, opacity);
    }
    let det_orig = (cov_2d[0] * cov_2d[2] - cov_2d[1] * cov_2d[1]).max(0.0);
    let regularized = [
        cov_2d[0] + filter_var_2d,
        cov_2d[1],
        cov_2d[2] + filter_var_2d,
    ];
    let det_reg = (regularized[0] * regularized[2] - regularized[1] * regularized[1]).max(1e-12);
    let comp = (det_orig / det_reg).clamp(0.0, 1.0).sqrt();
    (regularized, (opacity * comp).clamp(0.0, 1.0))
}

/// Child splat produced by transiently splitting an elongated Gaussian along its major axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplitGaussianChild {
    /// Center offset relative to the parent Gaussian center in world space.
    pub offset: Vec3,
    /// Packed symmetric 3D covariance `[xx, xy, xz, yy, yz, zz]` with halved major std-dev.
    pub covariance_3d: [f32; 6],
    /// Opacity for the child splat.
    pub opacity: f32,
}

/// Splits an elongated 3D Gaussian into two transient sub-splats placed at
/// $\pm 0.5 \sqrt{\lambda_{\max}} \mathbf{u}_1$ along its major principal axis,
/// scaling major variance by $1/4$ while preserving orthogonal variances.
pub fn split_gaussian_along_major_axis(
    _position: Vec3,
    cov: &[f32; 6],
    opacity: f32,
) -> [SplitGaussianChild; 2] {
    let (evals, basis) = symmetric_eigen_3x3(cov);
    let min_var = MIN_SCALE * MIN_SCALE;
    let l0 = (evals[0] * 0.25).max(min_var);
    let l1 = evals[1].max(min_var);
    let l2 = evals[2].max(min_var);

    let major_axis = basis.x_axis.normalize_or_zero();
    let offset_mag = 0.5 * evals[0].max(min_var).sqrt();
    let offset = major_axis * offset_mag;

    let diag = Mat3::from_diagonal(Vec3::new(l0, l1, l2));
    let child_mat = basis * diag * basis.transpose();
    let child_cov = [
        child_mat.x_axis.x,
        child_mat.x_axis.y,
        child_mat.x_axis.z,
        child_mat.y_axis.y,
        child_mat.y_axis.z,
        child_mat.z_axis.z,
    ];

    [
        SplitGaussianChild {
            offset: -offset,
            covariance_3d: child_cov,
            opacity,
        },
        SplitGaussianChild {
            offset,
            covariance_3d: child_cov,
            opacity,
        },
    ]
}

#[cfg(test)]
mod mip_strain_tests {
    use super::*;

    #[test]
    fn test_symmetric_eigen_and_decompose_roundtrip() {
        let scale = Vec3::new(0.4, 0.12, 0.05);
        let rot = Quat::from_euler(glam::EulerRot::XYZ, 0.35, -0.45, 0.72);
        let cov = compute_covariance_3d(scale, rot);
        let (evals, basis) = symmetric_eigen_3x3(&cov);
        assert!(evals[0] >= evals[1] && evals[1] >= evals[2]);
        assert!((basis.determinant() - 1.0).abs() < 1e-4);

        let (rec_scale, rec_rot) = decompose_covariance_3d(&cov);
        let rec_cov = compute_covariance_3d(rec_scale, rec_rot);
        for i in 0..6 {
            assert!((cov[i] - rec_cov[i]).abs() < 1e-4);
        }
    }

    #[test]
    fn test_mip_filter_3d_and_ewa_2d_attenuate_subpixel_opacity() {
        let tiny_cov = compute_covariance_3d(Vec3::splat(0.01), Quat::IDENTITY);
        let (reg_cov, reg_alpha) = apply_mip_filter_3d(&tiny_cov, 0.9, DEFAULT_MIP_FILTER_3D_VAR);
        assert!(reg_cov[0] > tiny_cov[0]);
        assert!(reg_alpha < 0.9 && reg_alpha > 0.0);

        let (reg_2d, reg_alpha_2d) =
            apply_ewa_low_pass_2d([0.05, 0.0, 0.05], 0.85, DEFAULT_MIP_FILTER_2D_VAR);
        assert!((reg_2d[0] - 0.35).abs() < 1e-6);
        assert!(reg_alpha_2d < 0.85);
    }

    #[test]
    fn test_split_gaussian_along_major_axis_quarters_major_variance() {
        let cov = compute_covariance_3d(Vec3::new(0.8, 0.1, 0.1), Quat::IDENTITY);
        let children = split_gaussian_along_major_axis(Vec3::ZERO, &cov, 0.7);
        assert!((children[0].offset.x + 0.4).abs() < 1e-4);
        assert!((children[1].offset.x - 0.4).abs() < 1e-4);
        assert!((children[0].covariance_3d[0] - 0.16).abs() < 1e-4);
        assert!((children[0].covariance_3d[3] - 0.01).abs() < 1e-4);
    }
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
        assert!(
            is_positive_definite(&cov_min),
            "Sigma must be positive definite"
        );

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
            [
                std::f32::consts::FRAC_1_SQRT_2,
                std::f32::consts::FRAC_1_SQRT_2,
                0.0,
                0.0,
            ],
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
                assert!(
                    det > 0.0,
                    "Determinant {} must be positive for s={:?}, r={:?}",
                    det,
                    raw_s,
                    raw_r
                );
                assert!(
                    is_positive_definite(&cov),
                    "Matrix must be positive definite for s={:?}, r={:?}",
                    raw_s,
                    raw_r
                );

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
            assert!(
                det > 0.0,
                "det(Sigma) must be > 0 for extreme scale {:?}",
                raw_s
            );
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
