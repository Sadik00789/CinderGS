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
