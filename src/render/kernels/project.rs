use cubecl::prelude::*;

/// CubeCL compute kernel performing EWA 3D-to-2D projection, conic derivation, and frustum culling.
///
/// Each invocation processes one Gaussian Splat in parallel:
/// - Transforms position from world to camera space: $t = W \cdot p$
/// - Culls splats behind near plane ($t_z \le \text{near\_plane}$)
/// - Transforms 3D covariance to camera space: $\Sigma_{\text{cam}} = W_{3 \times 3} \Sigma_{3D} W_{3 \times 3}^T$
/// - Computes perspective projection Jacobian $J$ and projects to 2D screen covariance $\Sigma_{2D}$
/// - Applies 0.3 pixel variance low-pass filter
/// - Culls degenerate splats ($\det \le 1e-6$)
/// - Inverts 2D covariance to produce conic coefficients $(a, b, c)$
/// - Computes semi-major axis eigenvalue $\lambda_{\max}$ and 3-sigma radius clamped to $[1, 1024]$
/// - Culls splats lying outside the screen viewport frustum
#[cube(launch)]
pub fn project_gaussians_kernel(
    positions: &[f32],
    covariances: &[f32],
    camera: &[f32],
    points_xy: &mut [f32],
    depths: &mut [f32],
    radii: &mut [i32],
    conics: &mut [f32],
    num_gaussians: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= num_gaussians as usize {
        terminate!();
    }

    // World position (px, py, pz)
    let p_base = idx * 3;
    let px = positions[p_base];
    let py = positions[p_base + 1];
    let pz = positions[p_base + 2];

    // Upper 3x4 view matrix from camera uniform (column-major)
    // Column 0
    let w00 = camera[0];
    let w10 = camera[1];
    let w20 = camera[2];
    // Column 1
    let w01 = camera[4];
    let w11 = camera[5];
    let w21 = camera[6];
    // Column 2
    let w02 = camera[8];
    let w12 = camera[9];
    let w22 = camera[10];
    // Column 3 (translation)
    let w03 = camera[12];
    let w13 = camera[13];
    let w23 = camera[14];

    // Camera parameters
    let focal_x = camera[16];
    let focal_y = camera[17];
    let principal_x = camera[18];
    let principal_y = camera[19];
    let viewport_width = camera[20];
    let viewport_height = camera[21];
    let near_plane = camera[22];

    // Camera space position: t = W * p
    let tx = w00 * px + w01 * py + w02 * pz + w03;
    let ty = w10 * px + w11 * py + w12 * pz + w13;
    let tz = w20 * px + w21 * py + w22 * pz + w23;

    let pt_base = idx * 2;
    let conic_base = idx * 3;

    // 1. Near-plane culling (tz <= near_plane)
    if tz <= near_plane {
        points_xy[pt_base] = 0.0;
        points_xy[pt_base + 1] = 0.0;
        depths[idx] = tz;
        radii[idx] = 0;
        conics[conic_base] = 0.0;
        conics[conic_base + 1] = 0.0;
        conics[conic_base + 2] = 0.0;
        terminate!();
    }

    // 2. 3D Covariance (symmetric upper-triangular: xx, xy, xz, yy, yz, zz)
    let cov_base = idx * 6;
    let v_xx = covariances[cov_base];
    let v_xy = covariances[cov_base + 1];
    let v_xz = covariances[cov_base + 2];
    let v_yy = covariances[cov_base + 3];
    let v_yz = covariances[cov_base + 4];
    let v_zz = covariances[cov_base + 5];

    // Transform 3D covariance to camera space: Sigma_cam = W3 * Sigma3D * W3^T
    // Step 2a: K = W3 * Sigma3D (3x3)
    let k00 = w00 * v_xx + w01 * v_xy + w02 * v_xz;
    let k01 = w00 * v_xy + w01 * v_yy + w02 * v_yz;
    let k02 = w00 * v_xz + w01 * v_yz + w02 * v_zz;

    let k10 = w10 * v_xx + w11 * v_xy + w12 * v_xz;
    let k11 = w10 * v_xy + w11 * v_yy + w12 * v_yz;
    let k12 = w10 * v_xz + w11 * v_yz + w12 * v_zz;

    let k20 = w20 * v_xx + w21 * v_xy + w22 * v_xz;
    let k21 = w20 * v_xy + w21 * v_yy + w22 * v_yz;
    let k22 = w20 * v_xz + w21 * v_yz + w22 * v_zz;

    // Step 2b: Sigma_cam = K * W3^T (3x3 symmetric)
    let cam_xx = k00 * w00 + k01 * w01 + k02 * w02;
    let cam_xy = k00 * w10 + k01 * w11 + k02 * w12;
    let cam_xz = k00 * w20 + k01 * w21 + k02 * w22;
    let cam_yy = k10 * w10 + k11 * w11 + k12 * w12;
    let cam_yz = k10 * w20 + k11 * w21 + k12 * w22;
    let cam_zz = k20 * w20 + k21 * w21 + k22 * w22;

    // 3. Perspective Jacobian J_proj
    let inv_tz = 1.0 / tz;
    let inv_tz2 = inv_tz * inv_tz;
    let j00 = focal_x * inv_tz;
    let j02 = -focal_x * tx * inv_tz2;
    let j11 = focal_y * inv_tz;
    let j12 = -focal_y * ty * inv_tz2;

    // 4. Screen covariance: Sigma_2D = J * Sigma_cam * J^T
    // M = J * Sigma_cam (2x3)
    let m00 = j00 * cam_xx + j02 * cam_xz;
    let m01 = j00 * cam_xy + j02 * cam_yz;
    let m02 = j00 * cam_xz + j02 * cam_zz;

    let m11 = j11 * cam_yy + j12 * cam_yz;
    let m12 = j11 * cam_yz + j12 * cam_zz;

    // Sigma_2D = M * J^T (2x2 symmetric)
    let sig2d_xx = m00 * j00 + m02 * j02;
    let sig2d_xy = m01 * j11 + m02 * j12;
    let sig2d_yy = m11 * j11 + m12 * j12;

    // 5. Low-pass filter (0.3 variance)
    let sig_prime_xx = sig2d_xx + 0.3;
    let sig_prime_xy = sig2d_xy;
    let sig_prime_yy = sig2d_yy + 0.3;

    // 6. Determinant and degeneracy check
    let det = sig_prime_xx * sig_prime_yy - sig_prime_xy * sig_prime_xy;
    if det <= 1e-6 {
        points_xy[pt_base] = 0.0;
        points_xy[pt_base + 1] = 0.0;
        depths[idx] = tz;
        radii[idx] = 0;
        conics[conic_base] = 0.0;
        conics[conic_base + 1] = 0.0;
        conics[conic_base + 2] = 0.0;
        terminate!();
    }

    // 7. Inverse covariance for conic coefficients: a = yy/det, b = -xy/det, c = xx/det
    let inv_det = 1.0 / det;
    let a = sig_prime_yy * inv_det;
    let b = -sig_prime_xy * inv_det;
    let c = sig_prime_xx * inv_det;

    // 8. Eigenvalue derivation for semi-major axis
    let mid = 0.5 * (sig_prime_xx + sig_prime_yy);
    let term = f32::max(0.1, mid * mid - det);
    let delta = f32::sqrt(term);
    let lambda1 = mid + delta;
    let lambda2 = f32::max(0.1, mid - delta);
    let lambda_max = f32::max(lambda1, lambda2);

    // 9. 3-sigma radius with hard clamp to [1, 1024]
    let r_raw = f32::ceil(3.0 * f32::sqrt(lambda_max));
    let radius = f32::clamp(r_raw, 1.0, 1024.0);

    // 10. Screen-space pixel coordinates
    let mu_x = focal_x * tx * inv_tz + principal_x;
    let mu_y = focal_y * ty * inv_tz + principal_y;

    // 11. Viewport frustum culling
    let min_x = mu_x - radius;
    let max_x = mu_x + radius;
    let min_y = mu_y - radius;
    let max_y = mu_y + radius;

    if max_x < 0.0
        || min_x >= viewport_width
        || max_y < 0.0
        || min_y >= viewport_height
    {
        points_xy[pt_base] = mu_x;
        points_xy[pt_base + 1] = mu_y;
        depths[idx] = tz;
        radii[idx] = 0;
        conics[conic_base] = a;
        conics[conic_base + 1] = b;
        conics[conic_base + 2] = c;
        terminate!();
    }

    // Surviving splat: write output attributes
    points_xy[pt_base] = mu_x;
    points_xy[pt_base + 1] = mu_y;
    depths[idx] = tz;
    radii[idx] = radius as i32;
    conics[conic_base] = a;
    conics[conic_base + 1] = b;
    conics[conic_base + 2] = c;
}
