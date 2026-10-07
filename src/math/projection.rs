use glam::{Mat3, Mat4, Vec3};

/// Constant low-pass filter variance added to 2D screen-space covariance diagonal.
pub const LOW_PASS_FILTER_VARIANCE: f32 = 0.3;

/// Determinant threshold below which a 2D Gaussian is considered degenerate and culled.
pub const MIN_DET_THRESHOLD: f32 = 1e-6;

/// Default near plane distance in camera space.
pub const DEFAULT_NEAR_PLANE: f32 = 0.2;

/// Hard minimum clamped radius for splats.
pub const MIN_SPLAT_RADIUS: f32 = 1.0;

/// Hard maximum clamped radius for splats (in pixels).
pub const MAX_SPLAT_RADIUS: f32 = 1024.0;

/// Output of projecting a single 3D Gaussian Splat to 2D screen space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectedSplat {
    /// Screen-space 2D center [mu_x, mu_y] in pixels.
    pub point_xy: [f32; 2],

    /// Camera-space depth t_z.
    pub depth: f32,

    /// Pixel radius r clamped to [1, 1024], or 0 if culled.
    pub radius: i32,

    /// Conic coefficients [a, b, c] defining inverse 2D covariance.
    pub conic: [f32; 3],
}

impl ProjectedSplat {
    /// Returns true if this splat was culled (radius == 0).
    #[inline]
    pub fn is_culled(&self) -> bool {
        self.radius <= 0
    }

    /// Culled representation for a Gaussian.
    #[inline]
    pub fn culled(depth: f32) -> Self {
        Self {
            point_xy: [0.0, 0.0],
            depth,
            radius: 0,
            conic: [0.0, 0.0, 0.0],
        }
    }
}

/// Perspective projection Jacobian elements:
///
/// $$J = \begin{bmatrix} \frac{f_x}{t_z} & 0 & -\frac{f_x t_x}{t_z^2} \\ 0 & \frac{f_y}{t_z} & -\frac{f_y t_y}{t_z^2} \end{bmatrix}$$
///
/// Returns `[j00, j02, j11, j12]`.
#[inline]
pub fn compute_jacobian(focal_x: f32, focal_y: f32, t_x: f32, t_y: f32, t_z: f32) -> [f32; 4] {
    let inv_tz = 1.0 / t_z;
    let inv_tz2 = inv_tz * inv_tz;
    [
        focal_x * inv_tz,         // J00
        -focal_x * t_x * inv_tz2, // J02
        focal_y * inv_tz,         // J11
        -focal_y * t_y * inv_tz2, // J12
    ]
}

/// Transforms 3D covariance to camera space:
///
/// $$\Sigma_{\text{cam}} = W_{3 \times 3} \cdot \Sigma_{3D} \cdot W_{3 \times 3}^T$$
#[inline]
pub fn transform_covariance_to_cam(w3: Mat3, cov3d: [f32; 6]) -> [f32; 6] {
    let sigma_3d = Mat3::from_cols(
        Vec3::new(cov3d[0], cov3d[1], cov3d[2]),
        Vec3::new(cov3d[1], cov3d[3], cov3d[4]),
        Vec3::new(cov3d[2], cov3d[4], cov3d[5]),
    );

    let sigma_cam = w3 * sigma_3d * w3.transpose();

    [
        sigma_cam.x_axis.x, // xx
        sigma_cam.x_axis.y, // xy
        sigma_cam.x_axis.z, // xz
        sigma_cam.y_axis.y, // yy
        sigma_cam.y_axis.z, // yz
        sigma_cam.z_axis.z, // zz
    ]
}

/// Projects camera-space covariance to 2D screen space using Jacobian:
///
/// $$\Sigma_{2D} = J \cdot \Sigma_{\text{cam}} \cdot J^T$$
///
/// Returns symmetric 2D covariance `[xx, xy, yy]`.
#[inline]
pub fn project_covariance_2d(jacobian: [f32; 4], cov_cam: [f32; 6]) -> [f32; 3] {
    let j00 = jacobian[0];
    let j02 = jacobian[1];
    let j11 = jacobian[2];
    let j12 = jacobian[3];

    let c_xx = cov_cam[0];
    let c_xy = cov_cam[1];
    let c_xz = cov_cam[2];
    let c_yy = cov_cam[3];
    let c_yz = cov_cam[4];
    let c_zz = cov_cam[5];

    // M = J * Sigma_cam (2 x 3)
    let m00 = j00 * c_xx + j02 * c_xz;
    let m01 = j00 * c_xy + j02 * c_yz;
    let m02 = j00 * c_xz + j02 * c_zz;

    let m11 = j11 * c_yy + j12 * c_yz;
    let m12 = j11 * c_yz + j12 * c_zz;

    // Sigma_2D = M * J^T (2 x 2 symmetric)
    let sigma_2d_xx = m00 * j00 + m02 * j02;
    let sigma_2d_xy = m01 * j11 + m02 * j12;
    let sigma_2d_yy = m11 * j11 + m12 * j12;

    [sigma_2d_xx, sigma_2d_xy, sigma_2d_yy]
}

/// Evaluates conic coefficients and screen-space 3-sigma radius from regularized 2D covariance:
///
/// - Regularizes: $\Sigma'_{xx} = \Sigma_{xx} + 0.3, \Sigma'_{yy} = \Sigma_{yy} + 0.3$
/// - Determinant: $\det = \Sigma'_{xx} \Sigma'_{yy} - (\Sigma'_{xy})^2$
/// - Conic: $a = \Sigma'_{yy} / \det, b = -\Sigma'_{xy} / \det, c = \Sigma'_{xx} / \det$
/// - Eigenvalues: $\lambda_{\max} = \text{mid} + \sqrt{\max(0.1, \text{mid}^2 - \det)}$
/// - Radius: $r = \operatorname{clamp}(\lceil 3.0 \cdot \sqrt{\lambda_{\max}} \rceil, 1.0, 1024.0)$
///
/// Returns `Some((conic, radius))` if $\det > 1e-6$, or `None` if culled.
#[inline]
pub fn compute_conic_and_radius(
    sigma_2d_xx: f32,
    sigma_2d_xy: f32,
    sigma_2d_yy: f32,
) -> Option<([f32; 3], f32)> {
    // 1. Low-pass filter (0.3 variance)
    let s_xx = sigma_2d_xx + LOW_PASS_FILTER_VARIANCE;
    let s_xy = sigma_2d_xy;
    let s_yy = sigma_2d_yy + LOW_PASS_FILTER_VARIANCE;

    // 2. Determinant check
    let det = s_xx * s_yy - s_xy * s_xy;
    if det <= MIN_DET_THRESHOLD {
        return None;
    }

    // 3. Conic coefficients (inverse 2D covariance)
    let inv_det = 1.0 / det;
    let conic = [s_yy * inv_det, -s_xy * inv_det, s_xx * inv_det];

    // 4. Eigenvalue derivation for semi-major axis
    let mid = 0.5 * (s_xx + s_yy);
    let discr = (mid * mid - det).max(0.1).sqrt();
    let lambda1 = mid + discr;
    let lambda2 = (mid - discr).max(0.1);
    let lambda_max = lambda1.max(lambda2);

    // 5. 3-sigma radius with hard clamp
    let r_raw = (3.0 * lambda_max.sqrt()).ceil();
    let r_clamped = r_raw.clamp(MIN_SPLAT_RADIUS, MAX_SPLAT_RADIUS);

    Some((conic, r_clamped))
}

/// Evaluates screen-space pixel coordinates:
///
/// $$\mu_x = \frac{f_x t_x}{t_z} + c_x, \quad \mu_y = \frac{f_y t_y}{t_z} + c_y$$
#[inline]
pub fn screen_space_center(
    focal_x: f32,
    focal_y: f32,
    principal_x: f32,
    principal_y: f32,
    t_x: f32,
    t_y: f32,
    t_z: f32,
) -> [f32; 2] {
    let inv_tz = 1.0 / t_z;
    [
        focal_x * t_x * inv_tz + principal_x,
        focal_y * t_y * inv_tz + principal_y,
    ]
}

/// Determines whether the bounding box $[\mu \pm r]$ lies strictly outside the viewport.
#[inline]
pub fn is_frustum_culled(
    mu_x: f32,
    mu_y: f32,
    radius: f32,
    viewport_width: f32,
    viewport_height: f32,
) -> bool {
    let min_x = mu_x - radius;
    let max_x = mu_x + radius;
    let min_y = mu_y - radius;
    let max_y = mu_y + radius;

    max_x < 0.0 || min_x >= viewport_width || max_y < 0.0 || min_y >= viewport_height
}

/// Complete CPU EWA projection pipeline for a single 3D Gaussian Splat.
#[allow(clippy::too_many_arguments)]
pub fn project_ewa_splat(
    world_pos: Vec3,
    cov3d: [f32; 6],
    view_matrix: Mat4,
    focal_x: f32,
    focal_y: f32,
    principal_x: f32,
    principal_y: f32,
    viewport_width: f32,
    viewport_height: f32,
    near_plane: f32,
) -> ProjectedSplat {
    // 1. Transform position to camera space
    let t = view_matrix.transform_point3(world_pos);

    // 2. Near-plane culling
    if t.z <= near_plane {
        return ProjectedSplat::culled(t.z);
    }

    // 3. Perspective Jacobian
    let jacobian = compute_jacobian(focal_x, focal_y, t.x, t.y, t.z);

    // 4. Upper-left 3x3 rotation/view transform
    let w3 = Mat3::from_cols(
        view_matrix.x_axis.truncate(),
        view_matrix.y_axis.truncate(),
        view_matrix.z_axis.truncate(),
    );

    // 5. Transform 3D covariance to camera space
    let cov_cam = transform_covariance_to_cam(w3, cov3d);

    // 6. Project covariance to 2D
    let cov_2d = project_covariance_2d(jacobian, cov_cam);

    // 7. Regularization, conic inversion, and 3-sigma radius
    let (conic, radius) = match compute_conic_and_radius(cov_2d[0], cov_2d[1], cov_2d[2]) {
        Some(res) => res,
        None => return ProjectedSplat::culled(t.z),
    };

    // 8. Screen-space center
    let center = screen_space_center(
        focal_x,
        focal_y,
        principal_x,
        principal_y,
        t.x,
        t.y,
        t.z,
    );

    // 9. Frustum culling
    if is_frustum_culled(center[0], center[1], radius, viewport_width, viewport_height) {
        return ProjectedSplat {
            point_xy: center,
            depth: t.z,
            radius: 0,
            conic,
        };
    }

    ProjectedSplat {
        point_xy: center,
        depth: t.z,
        radius: radius as i32,
        conic,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_projection_manual_reference() {
        // Camera at origin looking down positive Z: identity view matrix
        let view_matrix = Mat4::IDENTITY;
        let pos = Vec3::new(0.0, 0.0, 2.0); // 2 meters in front of camera
        let cov3d = [0.04, 0.0, 0.0, 0.04, 0.0, 0.04]; // spherical splat: s = 0.2

        let focal_x = 1000.0;
        let focal_y = 1000.0;
        let principal_x = 500.0;
        let principal_y = 400.0;
        let width = 1000.0;
        let height = 800.0;

        let splat = project_ewa_splat(
            pos,
            cov3d,
            view_matrix,
            focal_x,
            focal_y,
            principal_x,
            principal_y,
            width,
            height,
            0.2,
        );

        assert!(!splat.is_culled());
        assert_eq!(splat.depth, 2.0);
        assert_eq!(splat.point_xy, [500.0, 400.0]);

        // J00 = 1000 / 2 = 500, J11 = 500, J02 = 0, J12 = 0
        // cov_2d_xx = 500 * 0.04 * 500 = 10000
        // cov'_xx = 10000 + 0.3 = 10000.3
        // lambda_max = 10000.3
        // r_raw = ceil(3 * sqrt(10000.3)) = ceil(3 * 100.0015) = 301
        assert_eq!(splat.radius, 301);

        // Check conic coefficients: a = c = 1 / 10000.3, b = 0
        assert!((splat.conic[0] - 1.0 / 10000.3).abs() < 1e-6);
        assert_eq!(splat.conic[1], 0.0);
        assert!((splat.conic[2] - 1.0 / 10000.3).abs() < 1e-6);
    }

    #[test]
    fn test_near_plane_culling() {
        let view_matrix = Mat4::IDENTITY;
        let cov3d = [0.01, 0.0, 0.0, 0.01, 0.0, 0.01];

        // Splat behind near plane (tz = 0.1 <= 0.2)
        let pos_near = Vec3::new(0.0, 0.0, 0.1);
        let splat = project_ewa_splat(
            pos_near, cov3d, view_matrix, 500.0, 500.0, 250.0, 250.0, 500.0, 500.0, 0.2,
        );
        assert!(splat.is_culled());
        assert_eq!(splat.radius, 0);

        // Splat exactly on near plane (tz = 0.2 <= 0.2)
        let pos_exact = Vec3::new(0.0, 0.0, 0.2);
        let splat_exact = project_ewa_splat(
            pos_exact, cov3d, view_matrix, 500.0, 500.0, 250.0, 250.0, 500.0, 500.0, 0.2,
        );
        assert!(splat_exact.is_culled());
        assert_eq!(splat_exact.radius, 0);

        // Splat behind camera (tz = -5.0)
        let pos_behind = Vec3::new(0.0, 0.0, -5.0);
        let splat_behind = project_ewa_splat(
            pos_behind, cov3d, view_matrix, 500.0, 500.0, 250.0, 250.0, 500.0, 500.0, 0.2,
        );
        assert!(splat_behind.is_culled());
        assert_eq!(splat_behind.radius, 0);
    }

    #[test]
    fn test_frustum_culling_outside_viewport() {
        let view_matrix = Mat4::IDENTITY;
        let cov3d = [0.01, 0.0, 0.0, 0.01, 0.0, 0.01];

        // Center way off to the right: x = 100m, z = 2m => mu_x = 500 * 50 + 250 = 25250 >> 500
        let pos_offscreen = Vec3::new(100.0, 0.0, 2.0);
        let splat = project_ewa_splat(
            pos_offscreen, cov3d, view_matrix, 500.0, 500.0, 250.0, 250.0, 500.0, 500.0, 0.2,
        );
        assert!(splat.is_culled());
        assert_eq!(splat.radius, 0);
    }

    #[test]
    fn test_conic_positive_definiteness_and_stability() {
        let cov3d_test = [
            [0.1, 0.02, 0.01, 0.08, 0.03, 0.15],
            [1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            [0.001, 0.0, 0.0, 0.001, 0.0, 0.001],
        ];

        let view_matrix = Mat4::IDENTITY;

        for cov in cov3d_test {
            let splat = project_ewa_splat(
                Vec3::new(0.0, 0.0, 3.0),
                cov,
                view_matrix,
                800.0,
                800.0,
                400.0,
                300.0,
                800.0,
                600.0,
                0.2,
            );

            assert!(!splat.is_culled());
            let a = splat.conic[0];
            let b = splat.conic[1];
            let c = splat.conic[2];

            // Sylvester's criterion on 2D conic matrix: a > 0, c > 0, det = a * c - b^2 > 0
            assert!(a > 0.0, "Conic 'a' must be positive, got {}", a);
            assert!(c > 0.0, "Conic 'c' must be positive, got {}", c);
            let conic_det = a * c - b * b;
            assert!(conic_det > 0.0, "Conic det must be positive, got {}", conic_det);
        }
    }

    #[test]
    fn test_radius_clamping_at_1024() {
        // Enormous 3D covariance
        let huge_cov3d = [1000.0, 0.0, 0.0, 1000.0, 0.0, 1000.0];
        let view_matrix = Mat4::IDENTITY;

        let splat = project_ewa_splat(
            Vec3::new(0.0, 0.0, 1.0),
            huge_cov3d,
            view_matrix,
            1000.0,
            1000.0,
            500.0,
            500.0,
            2000.0,
            2000.0,
            0.2,
        );

        assert_eq!(splat.radius, 1024);
    }
}
