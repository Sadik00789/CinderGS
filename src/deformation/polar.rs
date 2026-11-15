use glam::Mat3;

/// Computes the Higham Polar Decomposition of a 3x3 deformation gradient $J = R \cdot S$:
///
/// - $R \in \text{SO}(3)$ is a pure rigid rotation matrix ($\det(R) = +1, R R^T = I$).
/// - $S \in \mathbb{R}^{3 \times 3}$ is a symmetric positive semi-definite stretch tensor ($S = S^T$).
///
/// Uses a 5-step Newton-Schulz iteration with determinant pre-normalization:
/// 1. $R_0 = J \cdot |\det(J)|^{-1/3}$
/// 2. For $k \in [0, 5)$: $R_{k+1} = 0.5 \cdot (R_k + R_k^{-T})$
/// 3. Chirality conservation: If $\det(R_5) < 0.0$, negate column 2 ($R_5[:, 2] = -R_5[:, 2]$) to enforce $\det(R) = +1$.
/// 4. Stretch computation: $S = 0.5 \cdot (R^T J + (R^T J)^T)$.
pub fn higham_polar_decomposition(j: Mat3) -> (Mat3, Mat3) {
    let det = j.determinant();

    // Guard against singular / degenerate matrices
    if det.abs() < 1e-8 {
        return (Mat3::IDENTITY, j);
    }

    // Step 1: Pre-normalization
    let gamma = det.abs().powf(-1.0 / 3.0);
    let mut r = j * gamma;

    // Step 2: 5-step Newton-Schulz iteration
    for _ in 0..5 {
        let r_inv = r.inverse();
        let r_inv_t = r_inv.transpose();
        r = 0.5 * (r + r_inv_t);
    }

    // Step 3: Enforce SO(3) chirality conservation (det(R) = +1)
    if r.determinant() < 0.0 {
        r.z_axis = -r.z_axis;
    }

    // Step 4: S = R^T * J (symmetrized)
    let s_raw = r.transpose() * j;
    let s = 0.5 * (s_raw + s_raw.transpose());

    (r, s)
}

/// Convenience function extracting only the pure rotation matrix $R \in \text{SO}(3)$ from $J$.
#[inline]
pub fn extract_rotation(j: Mat3) -> Mat3 {
    higham_polar_decomposition(j).0
}

/// Singular values and principal stretch axis extracted from a 3D deformation gradient $J$.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JacobianStrain {
    /// Singular values sorted descending: $\sigma_1 \ge \sigma_2 \ge \sigma_3 \ge 0$.
    pub sigma: [f32; 3],
    /// Unit principal stretch direction in world space (left singular vector $\mathbf{u}_1$).
    pub major_axis_world: glam::Vec3,
    /// Signed volume ratio $\det(J)$.
    pub determinant: f32,
}

impl JacobianStrain {
    /// Maximum principal stretch ratio ($\sigma_1$).
    #[inline]
    pub fn max_stretch(&self) -> f32 {
        self.sigma[0]
    }

    /// Minimum principal stretch ratio ($\sigma_3$).
    #[inline]
    pub fn min_stretch(&self) -> f32 {
        self.sigma[2]
    }

    /// Stretch anisotropy ratio $\sigma_1 / \max(\sigma_3, 10^{-4})$.
    #[inline]
    pub fn anisotropy_ratio(&self) -> f32 {
        self.sigma[0] / self.sigma[2].max(1e-4)
    }

    /// Returns `true` when the maximum principal stretch exceeds `threshold`.
    #[inline]
    pub fn exceeds_stretch(&self, threshold: f32) -> bool {
        threshold > 0.0 && self.sigma[0] > threshold
    }
}

/// Computes the singular values of the deformation gradient $J$ via eigendecomposition
/// of the right Cauchy-Green strain tensor $C = J^T J$.
pub fn jacobian_singular_values(j: Mat3) -> JacobianStrain {
    let c = j.transpose() * j;
    let packed = [
        c.x_axis.x, c.x_axis.y, c.x_axis.z, c.y_axis.y, c.y_axis.z, c.z_axis.z,
    ];
    let (evals, right_basis) = crate::math::covariance::symmetric_eigen_3x3(&packed);
    let sigma = [
        evals[0].max(0.0).sqrt(),
        evals[1].max(0.0).sqrt(),
        evals[2].max(0.0).sqrt(),
    ];
    let major_axis_world = (j * right_basis.x_axis)
        .try_normalize()
        .unwrap_or(glam::Vec3::X);

    JacobianStrain {
        sigma,
        major_axis_world,
        determinant: j.determinant(),
    }
}

#[cfg(test)]
mod strain_tests {
    use super::*;

    #[test]
    fn test_jacobian_singular_values_anisotropic_stretch() {
        let j = Mat3::from_diagonal(glam::Vec3::new(2.5, 1.0, 0.5));
        let strain = jacobian_singular_values(j);
        assert!((strain.max_stretch() - 2.5).abs() < 1e-4);
        assert!((strain.min_stretch() - 0.5).abs() < 1e-4);
        assert!(strain.exceeds_stretch(1.75));
        assert!(!strain.exceeds_stretch(2.6));
        assert!(
            strain.major_axis_world.abs_diff_eq(glam::Vec3::X, 1e-4)
                || strain.major_axis_world.abs_diff_eq(-glam::Vec3::X, 1e-4)
        );
    }

    #[test]
    fn test_jacobian_singular_values_rotation_invariant() {
        let rot = Mat3::from_rotation_y(0.7) * Mat3::from_rotation_x(-0.4);
        let j = rot * Mat3::from_diagonal(glam::Vec3::new(3.0, 1.2, 0.8));
        let strain = jacobian_singular_values(j);
        assert!((strain.max_stretch() - 3.0).abs() < 1e-4);
        assert!((strain.sigma[1] - 1.2).abs() < 1e-4);
        assert!((strain.min_stretch() - 0.8).abs() < 1e-4);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn is_orthogonal(m: Mat3, eps: f32) -> bool {
        let diff = (m * m.transpose()) - Mat3::IDENTITY;
        diff.abs_diff_eq(Mat3::ZERO, eps)
    }

    #[test]
    fn test_polar_identity() {
        let j = Mat3::IDENTITY;
        let (r, s) = higham_polar_decomposition(j);

        assert!(is_orthogonal(r, 1e-6));
        assert!((r.determinant() - 1.0).abs() < 1e-6);
        assert!(r.abs_diff_eq(Mat3::IDENTITY, 1e-6));
        assert!(s.abs_diff_eq(Mat3::IDENTITY, 1e-6));
    }

    #[test]
    fn test_polar_pure_rotation() {
        let angles = [0.3f32, -0.7f32, 1.2f32];
        let rot_x = Mat3::from_rotation_x(angles[0]);
        let rot_y = Mat3::from_rotation_y(angles[1]);
        let rot_z = Mat3::from_rotation_z(angles[2]);
        let j = rot_z * rot_y * rot_x;

        let (r, s) = higham_polar_decomposition(j);

        assert!(is_orthogonal(r, 1e-5));
        assert!((r.determinant() - 1.0).abs() < 1e-5);
        assert!(r.abs_diff_eq(j, 1e-5));
        assert!(s.abs_diff_eq(Mat3::IDENTITY, 1e-5));
    }

    #[test]
    fn test_polar_rotation_plus_scale() {
        let rot = Mat3::from_rotation_z(std::f32::consts::FRAC_PI_4); // 45 degrees
        let scale = Mat3::from_diagonal(Vec3::new(2.0, 3.0, 0.5));
        let j = rot * scale;

        let (r, s) = higham_polar_decomposition(j);

        assert!(is_orthogonal(r, 1e-4));
        assert!((r.determinant() - 1.0).abs() < 1e-4);
        assert!(r.abs_diff_eq(rot, 1e-4));
        assert!(s.abs_diff_eq(scale, 1e-4));
    }

    #[test]
    fn test_polar_inverted_reflection() {
        // Reflection across z-axis combined with scale
        let j = Mat3::from_diagonal(Vec3::new(2.0, 2.0, -2.0));
        assert!(j.determinant() < 0.0);

        let (r, _) = higham_polar_decomposition(j);

        assert!(is_orthogonal(r, 1e-5));
        // Enforced det(R) = +1 by column 2 negation
        assert!((r.determinant() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_polar_near_singular_matrix() {
        let j = Mat3::ZERO;
        let (r, _) = higham_polar_decomposition(j);

        assert_eq!(r, Mat3::IDENTITY);
    }
}
