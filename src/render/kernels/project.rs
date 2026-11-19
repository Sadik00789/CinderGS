#![allow(clippy::excessive_precision)]

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

    if max_x < 0.0 || min_x >= viewport_width || max_y < 0.0 || min_y >= viewport_height {
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

/// Decodes IEEE-754 binary16 low 16 bits from a `u32` into `f32` via pure integer
/// bit manipulation (no floating-point `powf` transcendentals).
#[cube]
pub fn unpack_f16(bits: u32) -> f32 {
    let h = bits & 0xFFFFu32;
    let sign = (h & 0x8000u32) << 16u32;
    let exp = (h >> 10u32) & 0x1Fu32;
    let mant = h & 0x3FFu32;

    if exp == 0u32 {
        if mant == 0u32 {
            f32::reinterpret(sign)
        } else {
            let f_sub = (mant as f32) * 5.9604645e-8f32;
            let bits_sub = u32::reinterpret(f_sub) | sign;
            f32::reinterpret(bits_sub)
        }
    } else if exp == 31u32 {
        if (h & 0x8000u32) != 0u32 {
            -65504.0f32
        } else {
            65504.0f32
        }
    } else {
        let f32_bits = sign | ((exp + 112u32) << 23u32) | (mant << 13u32);
        f32::reinterpret(f32_bits)
    }
}

/// GPU projection kernel with 3D Mip-Splatting variance regularization, 2D EWA opacity
/// compensation, opacity/sub-pixel/frustum culling, and `visibility_flags` emission
/// driving indirect tile-binning dispatch.
#[allow(clippy::too_many_arguments, clippy::unnecessary_cast)]
#[cube(launch)]
pub fn project_gaussians_cull_kernel(
    positions: &[f32],
    covariances: &[f32],
    opacities: &[f32],
    camera: &[f32],
    mip_filter_3d_var: f32,
    min_opacity: f32,
    min_radius_px: f32,
    points_xy: &mut [f32],
    depths: &mut [f32],
    radii: &mut [i32],
    conics: &mut [f32],
    out_opacities: &mut [f32],
    visibility_flags: &mut [u32],
    num_gaussians: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= num_gaussians as usize {
        terminate!();
    }

    visibility_flags[idx] = 0u32;
    radii[idx] = 0;
    out_opacities[idx] = 0.0f32;

    let pt_base = idx * 2usize;
    let conic_base = idx * 3usize;
    points_xy[pt_base] = 0.0f32;
    points_xy[pt_base + 1usize] = 0.0f32;
    conics[conic_base] = 0.0f32;
    conics[conic_base + 1usize] = 0.0f32;
    conics[conic_base + 2usize] = 0.0f32;

    let p_base = idx * 3usize;
    let px = positions[p_base];
    let py = positions[p_base + 1usize];
    let pz = positions[p_base + 2usize];

    let w00 = camera[0usize];
    let w10 = camera[1usize];
    let w20 = camera[2usize];
    let w01 = camera[4usize];
    let w11 = camera[5usize];
    let w21 = camera[6usize];
    let w02 = camera[8usize];
    let w12 = camera[9usize];
    let w22 = camera[10usize];
    let w03 = camera[12usize];
    let w13 = camera[13usize];
    let w23 = camera[14usize];

    let focal_x = camera[16usize];
    let focal_y = camera[17usize];
    let principal_x = camera[18usize];
    let principal_y = camera[19usize];
    let viewport_width = camera[20usize];
    let viewport_height = camera[21usize];
    let near_plane = camera[22usize];

    let tx = w00 * px + w01 * py + w02 * pz + w03;
    let ty = w10 * px + w11 * py + w12 * pz + w13;
    let tz = w20 * px + w21 * py + w22 * pz + w23;
    depths[idx] = tz;

    if tz <= near_plane {
        terminate!();
    }

    let op = opacities[idx];
    if op < min_opacity {
        terminate!();
    }

    let cov_base = idx * 6usize;
    let mut v_xx = covariances[cov_base];
    let v_xy = covariances[cov_base + 1usize];
    let v_xz = covariances[cov_base + 2usize];
    let mut v_yy = covariances[cov_base + 3usize];
    let v_yz = covariances[cov_base + 4usize];
    let mut v_zz = covariances[cov_base + 5usize];

    let mut alpha_3d = op;
    if mip_filter_3d_var > 0.0f32 {
        let det_orig = f32::max(
            v_xx * (v_yy * v_zz - v_yz * v_yz) - v_xy * (v_xy * v_zz - v_xz * v_yz)
                + v_xz * (v_xy * v_yz - v_xz * v_yy),
            1e-24f32,
        );
        v_xx += mip_filter_3d_var;
        v_yy += mip_filter_3d_var;
        v_zz += mip_filter_3d_var;
        let det_reg = f32::max(
            v_xx * (v_yy * v_zz - v_yz * v_yz) - v_xy * (v_xy * v_zz - v_xz * v_yz)
                + v_xz * (v_xy * v_yz - v_xz * v_yy),
            det_orig,
        );
        let comp = f32::sqrt(f32::clamp(det_orig / det_reg, 0.0f32, 1.0f32));
        alpha_3d = f32::clamp(op * comp, 0.0f32, 1.0f32);
        if alpha_3d < min_opacity {
            terminate!();
        }
    }

    let k00 = w00 * v_xx + w01 * v_xy + w02 * v_xz;
    let k01 = w00 * v_xy + w01 * v_yy + w02 * v_yz;
    let k02 = w00 * v_xz + w01 * v_yz + w02 * v_zz;
    let k10 = w10 * v_xx + w11 * v_xy + w12 * v_xz;
    let k11 = w10 * v_xy + w11 * v_yy + w12 * v_yz;
    let k12 = w10 * v_xz + w11 * v_yz + w12 * v_zz;
    let k20 = w20 * v_xx + w21 * v_xy + w22 * v_xz;
    let k21 = w20 * v_xy + w21 * v_yy + w22 * v_yz;
    let k22 = w20 * v_xz + w21 * v_yz + w22 * v_zz;

    let cam_xx = k00 * w00 + k01 * w01 + k02 * w02;
    let cam_xy = k00 * w10 + k01 * w11 + k02 * w12;
    let cam_xz = k00 * w20 + k01 * w21 + k02 * w22;
    let cam_yy = k10 * w10 + k11 * w11 + k12 * w12;
    let cam_yz = k10 * w20 + k11 * w21 + k12 * w22;
    let cam_zz = k20 * w20 + k21 * w21 + k22 * w22;

    let inv_tz = 1.0f32 / tz;
    let inv_tz2 = inv_tz * inv_tz;
    let j00 = focal_x * inv_tz;
    let j02 = -focal_x * tx * inv_tz2;
    let j11 = focal_y * inv_tz;
    let j12 = -focal_y * ty * inv_tz2;

    let m00 = j00 * cam_xx + j02 * cam_xz;
    let m01 = j00 * cam_xy + j02 * cam_yz;
    let m02 = j00 * cam_xz + j02 * cam_zz;
    let m11 = j11 * cam_yy + j12 * cam_yz;
    let m12 = j11 * cam_yz + j12 * cam_zz;

    let sig2d_xx = m00 * j00 + m02 * j02;
    let sig2d_xy = m01 * j11 + m02 * j12;
    let sig2d_yy = m11 * j11 + m12 * j12;

    // Pre-filter screen-space 3-sigma radius culling
    let raw_mid = 0.5f32 * (sig2d_xx + sig2d_yy);
    let raw_det = f32::max(sig2d_xx * sig2d_yy - sig2d_xy * sig2d_xy, 0.0f32);
    let raw_discr = f32::sqrt(f32::max(raw_mid * raw_mid - raw_det, 0.0f32));
    let raw_lambda_max = f32::max(raw_mid + raw_discr, 0.0f32);
    if 3.0f32 * f32::sqrt(raw_lambda_max) < min_radius_px {
        terminate!();
    }

    // 2D EWA low-pass filter with area-conserving opacity compensation
    let sig_prime_xx = sig2d_xx + 0.3f32;
    let sig_prime_xy = sig2d_xy;
    let sig_prime_yy = sig2d_yy + 0.3f32;
    let det = sig_prime_xx * sig_prime_yy - sig_prime_xy * sig_prime_xy;
    if det <= 1e-6f32 {
        terminate!();
    }

    let comp_2d = f32::sqrt(f32::clamp(raw_det / det, 0.0f32, 1.0f32));
    let alpha_2d = f32::clamp(alpha_3d * comp_2d, 0.0f32, 1.0f32);
    if alpha_2d < min_opacity {
        terminate!();
    }

    let inv_det = 1.0f32 / det;
    let a = sig_prime_yy * inv_det;
    let b = -sig_prime_xy * inv_det;
    let c = sig_prime_xx * inv_det;

    let mid = 0.5f32 * (sig_prime_xx + sig_prime_yy);
    let term = f32::max(0.1f32, mid * mid - det);
    let delta = f32::sqrt(term);
    let lambda1 = mid + delta;
    let lambda2 = f32::max(0.1f32, mid - delta);
    let lambda_max = f32::max(lambda1, lambda2);
    let radius = f32::clamp(f32::ceil(3.0f32 * f32::sqrt(lambda_max)), 1.0f32, 1024.0f32);

    let mu_x = focal_x * tx * inv_tz + principal_x;
    let mu_y = focal_y * ty * inv_tz + principal_y;
    points_xy[pt_base] = mu_x;
    points_xy[pt_base + 1usize] = mu_y;
    conics[conic_base] = a;
    conics[conic_base + 1usize] = b;
    conics[conic_base + 2usize] = c;

    if mu_x + radius < 0.0f32
        || mu_x - radius >= viewport_width
        || mu_y + radius < 0.0f32
        || mu_y - radius >= viewport_height
    {
        terminate!();
    }

    radii[idx] = radius as i32;
    out_opacities[idx] = alpha_2d;
    visibility_flags[idx] = 1u32;
}

/// GPU projection kernel reading packed `f16` 3D covariances (`3 x u32` words per splat)
/// with 3D/2D Mip-Splatting filtering and frustum/opacity/radius culling.
#[allow(clippy::too_many_arguments, clippy::unnecessary_cast)]
#[cube(launch)]
pub fn project_gaussians_quantized_kernel(
    positions: &[f32],
    covariances_f16_packed: &[u32],
    opacities: &[f32],
    camera: &[f32],
    mip_filter_3d_var: f32,
    min_opacity: f32,
    min_radius_px: f32,
    points_xy: &mut [f32],
    depths: &mut [f32],
    radii: &mut [i32],
    conics: &mut [f32],
    out_opacities: &mut [f32],
    visibility_flags: &mut [u32],
    num_gaussians: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= num_gaussians as usize {
        terminate!();
    }

    visibility_flags[idx] = 0u32;
    radii[idx] = 0;
    out_opacities[idx] = 0.0f32;

    let pt_base = idx * 2usize;
    let conic_base = idx * 3usize;
    points_xy[pt_base] = 0.0f32;
    points_xy[pt_base + 1usize] = 0.0f32;
    conics[conic_base] = 0.0f32;
    conics[conic_base + 1usize] = 0.0f32;
    conics[conic_base + 2usize] = 0.0f32;

    let p_base = idx * 3usize;
    let px = positions[p_base];
    let py = positions[p_base + 1usize];
    let pz = positions[p_base + 2usize];

    let w00 = camera[0usize];
    let w10 = camera[1usize];
    let w20 = camera[2usize];
    let w01 = camera[4usize];
    let w11 = camera[5usize];
    let w21 = camera[6usize];
    let w02 = camera[8usize];
    let w12 = camera[9usize];
    let w22 = camera[10usize];
    let w03 = camera[12usize];
    let w13 = camera[13usize];
    let w23 = camera[14usize];

    let focal_x = camera[16usize];
    let focal_y = camera[17usize];
    let principal_x = camera[18usize];
    let principal_y = camera[19usize];
    let viewport_width = camera[20usize];
    let viewport_height = camera[21usize];
    let near_plane = camera[22usize];

    let tx = w00 * px + w01 * py + w02 * pz + w03;
    let ty = w10 * px + w11 * py + w12 * pz + w13;
    let tz = w20 * px + w21 * py + w22 * pz + w23;
    depths[idx] = tz;

    if tz <= near_plane {
        terminate!();
    }

    let op = opacities[idx];
    if op < min_opacity {
        terminate!();
    }

    let cov_base = idx * 3usize;
    let c_w0 = covariances_f16_packed[cov_base];
    let c_w1 = covariances_f16_packed[cov_base + 1usize];
    let c_w2 = covariances_f16_packed[cov_base + 2usize];
    let mut v_xx = unpack_f16(c_w0);
    let v_xy = unpack_f16(c_w0 >> 16u32);
    let v_xz = unpack_f16(c_w1);
    let mut v_yy = unpack_f16(c_w1 >> 16u32);
    let v_yz = unpack_f16(c_w2);
    let mut v_zz = unpack_f16(c_w2 >> 16u32);

    let mut alpha_3d = op;
    if mip_filter_3d_var > 0.0f32 {
        let det_orig = f32::max(
            v_xx * (v_yy * v_zz - v_yz * v_yz) - v_xy * (v_xy * v_zz - v_xz * v_yz)
                + v_xz * (v_xy * v_yz - v_xz * v_yy),
            1e-24f32,
        );
        v_xx += mip_filter_3d_var;
        v_yy += mip_filter_3d_var;
        v_zz += mip_filter_3d_var;
        let det_reg = f32::max(
            v_xx * (v_yy * v_zz - v_yz * v_yz) - v_xy * (v_xy * v_zz - v_xz * v_yz)
                + v_xz * (v_xy * v_yz - v_xz * v_yy),
            det_orig,
        );
        let comp = f32::sqrt(f32::clamp(det_orig / det_reg, 0.0f32, 1.0f32));
        alpha_3d = f32::clamp(op * comp, 0.0f32, 1.0f32);
        if alpha_3d < min_opacity {
            terminate!();
        }
    }

    let k00 = w00 * v_xx + w01 * v_xy + w02 * v_xz;
    let k01 = w00 * v_xy + w01 * v_yy + w02 * v_yz;
    let k02 = w00 * v_xz + w01 * v_yz + w02 * v_zz;
    let k10 = w10 * v_xx + w11 * v_xy + w12 * v_xz;
    let k11 = w10 * v_xy + w11 * v_yy + w12 * v_yz;
    let k12 = w10 * v_xz + w11 * v_yz + w12 * v_zz;
    let k20 = w20 * v_xx + w21 * v_xy + w22 * v_xz;
    let k21 = w20 * v_xy + w21 * v_yy + w22 * v_yz;
    let k22 = w20 * v_xz + w21 * v_yz + w22 * v_zz;

    let cam_xx = k00 * w00 + k01 * w01 + k02 * w02;
    let cam_xy = k00 * w10 + k01 * w11 + k02 * w12;
    let cam_xz = k00 * w20 + k01 * w21 + k02 * w22;
    let cam_yy = k10 * w10 + k11 * w11 + k12 * w12;
    let cam_yz = k10 * w20 + k11 * w21 + k12 * w22;
    let cam_zz = k20 * w20 + k21 * w21 + k22 * w22;

    let inv_tz = 1.0f32 / tz;
    let inv_tz2 = inv_tz * inv_tz;
    let j00 = focal_x * inv_tz;
    let j02 = -focal_x * tx * inv_tz2;
    let j11 = focal_y * inv_tz;
    let j12 = -focal_y * ty * inv_tz2;

    let m00 = j00 * cam_xx + j02 * cam_xz;
    let m01 = j00 * cam_xy + j02 * cam_yz;
    let m02 = j00 * cam_xz + j02 * cam_zz;
    let m11 = j11 * cam_yy + j12 * cam_yz;
    let m12 = j11 * cam_yz + j12 * cam_zz;

    let sig2d_xx = m00 * j00 + m02 * j02;
    let sig2d_xy = m01 * j11 + m02 * j12;
    let sig2d_yy = m11 * j11 + m12 * j12;

    let raw_mid = 0.5f32 * (sig2d_xx + sig2d_yy);
    let raw_det = f32::max(sig2d_xx * sig2d_yy - sig2d_xy * sig2d_xy, 0.0f32);
    let raw_discr = f32::sqrt(f32::max(raw_mid * raw_mid - raw_det, 0.0f32));
    let raw_lambda_max = f32::max(raw_mid + raw_discr, 0.0f32);
    if 3.0f32 * f32::sqrt(raw_lambda_max) < min_radius_px {
        terminate!();
    }

    let sig_prime_xx = sig2d_xx + 0.3f32;
    let sig_prime_xy = sig2d_xy;
    let sig_prime_yy = sig2d_yy + 0.3f32;
    let det = sig_prime_xx * sig_prime_yy - sig_prime_xy * sig_prime_xy;
    if det <= 1e-6f32 {
        terminate!();
    }

    let comp_2d = f32::sqrt(f32::clamp(raw_det / det, 0.0f32, 1.0f32));
    let alpha_2d = f32::clamp(alpha_3d * comp_2d, 0.0f32, 1.0f32);
    if alpha_2d < min_opacity {
        terminate!();
    }

    let inv_det = 1.0f32 / det;
    let a = sig_prime_yy * inv_det;
    let b = -sig_prime_xy * inv_det;
    let c = sig_prime_xx * inv_det;

    let mid = 0.5f32 * (sig_prime_xx + sig_prime_yy);
    let term = f32::max(0.1f32, mid * mid - det);
    let delta = f32::sqrt(term);
    let lambda1 = mid + delta;
    let lambda2 = f32::max(0.1f32, mid - delta);
    let lambda_max = f32::max(lambda1, lambda2);
    let radius = f32::clamp(f32::ceil(3.0f32 * f32::sqrt(lambda_max)), 1.0f32, 1024.0f32);

    let mu_x = focal_x * tx * inv_tz + principal_x;
    let mu_y = focal_y * ty * inv_tz + principal_y;
    points_xy[pt_base] = mu_x;
    points_xy[pt_base + 1usize] = mu_y;
    conics[conic_base] = a;
    conics[conic_base + 1usize] = b;
    conics[conic_base + 2usize] = c;

    if mu_x + radius < 0.0f32
        || mu_x - radius >= viewport_width
        || mu_y + radius < 0.0f32
        || mu_y - radius >= viewport_height
    {
        terminate!();
    }

    radii[idx] = radius as i32;
    out_opacities[idx] = alpha_2d;
    visibility_flags[idx] = 1u32;
}

/// Reduces per-Gaussian `visibility_flags` inside a single 256-thread workgroup to produce
/// `[workgroups_x, workgroups_y, workgroups_z, surviving_count]` for indirect GPU dispatch.
#[allow(clippy::unnecessary_cast, clippy::manual_div_ceil)]
#[cube(launch)]
pub fn build_indirect_dispatch_kernel(
    visibility_flags: &[u32],
    indirect_args: &mut [u32],
    num_gaussians: u32,
    dispatch_workgroup_size: u32,
) {
    let lid_u32 = UNIT_POS;
    let lid = lid_u32 as usize;
    let mut partial = Shared::<[u32]>::new_slice(256usize);

    let mut local_sum = 0u32;
    let mut i = lid;
    while i < num_gaussians as usize {
        if visibility_flags[i] != 0u32 {
            local_sum += 1u32;
        }
        i += 256usize;
    }

    partial[lid] = local_sum;
    sync_cube();

    if lid_u32 < 128u32 {
        partial[lid] += partial[lid + 128usize];
    }
    sync_cube();
    if lid_u32 < 64u32 {
        partial[lid] += partial[lid + 64usize];
    }
    sync_cube();
    if lid_u32 < 32u32 {
        partial[lid] += partial[lid + 32usize];
    }
    sync_cube();
    if lid_u32 < 16u32 {
        partial[lid] += partial[lid + 16usize];
    }
    sync_cube();
    if lid_u32 < 8u32 {
        partial[lid] += partial[lid + 8usize];
    }
    sync_cube();
    if lid_u32 < 4u32 {
        partial[lid] += partial[lid + 4usize];
    }
    sync_cube();
    if lid_u32 < 2u32 {
        partial[lid] += partial[lid + 2usize];
    }
    sync_cube();
    if lid_u32 < 1u32 {
        partial[lid] += partial[lid + 1usize];
    }
    sync_cube();

    if lid_u32 == 0u32 {
        let surviving = partial[0usize];
        let wg = u32::max(dispatch_workgroup_size, 1u32);
        indirect_args[0usize] = if surviving == 0u32 {
            0u32
        } else {
            (surviving + wg - 1u32) / wg
        };
        indirect_args[1usize] = 1u32;
        indirect_args[2usize] = 1u32;
        indirect_args[3usize] = surviving;
    }
}

/// Multi-workgroup atomic counter reduction over `visibility_flags`.
///
/// `out_counter[0]` must be initialized to `0` prior to launch. Each 256-thread
/// workgroup reduces in shared memory and atomically adds its subtotal once.
#[allow(clippy::unnecessary_cast)]
#[cube(launch)]
pub fn count_visibility_atomic_kernel(
    visibility_flags: &[u32],
    out_counter: &mut [Atomic<u32>],
    num_gaussians: u32,
) {
    let idx = ABSOLUTE_POS;
    let lid_u32 = UNIT_POS;
    let lid = lid_u32 as usize;
    let mut partial = Shared::<[u32]>::new_slice(256usize);

    partial[lid] = if idx < num_gaussians as usize && visibility_flags[idx] != 0u32 {
        1u32
    } else {
        0u32
    };
    sync_cube();

    if lid_u32 < 128u32 {
        partial[lid] += partial[lid + 128usize];
    }
    sync_cube();
    if lid_u32 < 64u32 {
        partial[lid] += partial[lid + 64usize];
    }
    sync_cube();
    if lid_u32 < 32u32 {
        partial[lid] += partial[lid + 32usize];
    }
    sync_cube();
    if lid_u32 < 16u32 {
        partial[lid] += partial[lid + 16usize];
    }
    sync_cube();
    if lid_u32 < 8u32 {
        partial[lid] += partial[lid + 8usize];
    }
    sync_cube();
    if lid_u32 < 4u32 {
        partial[lid] += partial[lid + 4usize];
    }
    sync_cube();
    if lid_u32 < 2u32 {
        partial[lid] += partial[lid + 2usize];
    }
    sync_cube();
    if lid_u32 < 1u32 {
        partial[lid] += partial[lid + 1usize];
    }
    sync_cube();

    if lid_u32 == 0u32 {
        let wg_sum = partial[0usize];
        if wg_sum > 0u32 {
            out_counter[0usize].fetch_add(wg_sum);
        }
    }
}

/// Writes `[workgroups_x, 1, 1, surviving_count]` from the atomic counter buffer.
#[allow(clippy::manual_div_ceil)]
#[cube(launch)]
pub fn finalize_indirect_dispatch_kernel(
    counter: &[u32],
    indirect_args: &mut [u32],
    dispatch_workgroup_size: u32,
) {
    if ABSOLUTE_POS != 0usize {
        terminate!();
    }
    let surviving = counter[0usize];
    let wg = u32::max(dispatch_workgroup_size, 1u32);
    indirect_args[0usize] = if surviving == 0u32 {
        0u32
    } else {
        (surviving + wg - 1u32) / wg
    };
    indirect_args[1usize] = 1u32;
    indirect_args[2usize] = 1u32;
    indirect_args[3usize] = surviving;
}

/// Evaluates degrees 0–3 spherical harmonics from 48-float rest buffers matching
/// [`crate::render::rasterizer::sh::evaluate_sh`].
#[cube]
pub fn eval_sh_f32(
    sh_coeffs: &[f32],
    sh_base: usize,
    dir_x: f32,
    dir_y: f32,
    dir_z: f32,
    sh_degree: u32,
) -> (f32, f32, f32) {
    let mut color_r = 0.28209479177387814f32 * sh_coeffs[sh_base];
    let mut color_g = 0.28209479177387814f32 * sh_coeffs[sh_base + 1usize];
    let mut color_b = 0.28209479177387814f32 * sh_coeffs[sh_base + 2usize];

    if sh_degree >= 1u32 {
        let y1_m1 = -0.4886025119029199f32 * dir_y;
        let y1_0 = 0.4886025119029199f32 * dir_z;
        let y1_p1 = -0.4886025119029199f32 * dir_x;

        color_r += y1_m1 * sh_coeffs[sh_base + 3usize]
            + y1_0 * sh_coeffs[sh_base + 4usize]
            + y1_p1 * sh_coeffs[sh_base + 5usize];
        color_g += y1_m1 * sh_coeffs[sh_base + 18usize]
            + y1_0 * sh_coeffs[sh_base + 19usize]
            + y1_p1 * sh_coeffs[sh_base + 20usize];
        color_b += y1_m1 * sh_coeffs[sh_base + 33usize]
            + y1_0 * sh_coeffs[sh_base + 34usize]
            + y1_p1 * sh_coeffs[sh_base + 35usize];

        if sh_degree >= 2u32 {
            let xx = dir_x * dir_x;
            let yy = dir_y * dir_y;
            let zz = dir_z * dir_z;
            let xy = dir_x * dir_y;
            let yz = dir_y * dir_z;
            let xz = dir_x * dir_z;

            let y2_m2 = 1.0925484305920792f32 * xy;
            let y2_m1 = -1.0925484305920792f32 * yz;
            let y2_0 = 0.31539156525252005f32 * (2.0f32 * zz - xx - yy);
            let y2_p1 = -1.0925484305920792f32 * xz;
            let y2_p2 = 0.5462742152960396f32 * (xx - yy);

            color_r += y2_m2 * sh_coeffs[sh_base + 6usize]
                + y2_m1 * sh_coeffs[sh_base + 7usize]
                + y2_0 * sh_coeffs[sh_base + 8usize]
                + y2_p1 * sh_coeffs[sh_base + 9usize]
                + y2_p2 * sh_coeffs[sh_base + 10usize];
            color_g += y2_m2 * sh_coeffs[sh_base + 21usize]
                + y2_m1 * sh_coeffs[sh_base + 22usize]
                + y2_0 * sh_coeffs[sh_base + 23usize]
                + y2_p1 * sh_coeffs[sh_base + 24usize]
                + y2_p2 * sh_coeffs[sh_base + 25usize];
            color_b += y2_m2 * sh_coeffs[sh_base + 36usize]
                + y2_m1 * sh_coeffs[sh_base + 37usize]
                + y2_0 * sh_coeffs[sh_base + 38usize]
                + y2_p1 * sh_coeffs[sh_base + 39usize]
                + y2_p2 * sh_coeffs[sh_base + 40usize];

            if sh_degree >= 3u32 {
                let y3_m3 = -0.5900435899266435f32 * dir_y * (3.0f32 * xx - yy);
                let y3_m2 = 2.890611442640554f32 * xy * dir_z;
                let y3_m1 = -0.4570457994644658f32 * dir_y * (4.0f32 * zz - xx - yy);
                let y3_0 =
                    0.3731763325901154f32 * dir_z * (2.0f32 * zz - 3.0f32 * xx - 3.0f32 * yy);
                let y3_p1 = -0.4570457994644658f32 * dir_x * (4.0f32 * zz - xx - yy);
                let y3_p2 = 1.445305721320277f32 * dir_z * (xx - yy);
                let y3_p3 = -0.5900435899266435f32 * dir_x * (xx - 3.0f32 * yy);

                color_r += y3_m3 * sh_coeffs[sh_base + 11usize]
                    + y3_m2 * sh_coeffs[sh_base + 12usize]
                    + y3_m1 * sh_coeffs[sh_base + 13usize]
                    + y3_0 * sh_coeffs[sh_base + 14usize]
                    + y3_p1 * sh_coeffs[sh_base + 15usize]
                    + y3_p2 * sh_coeffs[sh_base + 16usize]
                    + y3_p3 * sh_coeffs[sh_base + 17usize];
                color_g += y3_m3 * sh_coeffs[sh_base + 26usize]
                    + y3_m2 * sh_coeffs[sh_base + 27usize]
                    + y3_m1 * sh_coeffs[sh_base + 28usize]
                    + y3_0 * sh_coeffs[sh_base + 29usize]
                    + y3_p1 * sh_coeffs[sh_base + 30usize]
                    + y3_p2 * sh_coeffs[sh_base + 31usize]
                    + y3_p3 * sh_coeffs[sh_base + 32usize];
                color_b += y3_m3 * sh_coeffs[sh_base + 41usize]
                    + y3_m2 * sh_coeffs[sh_base + 42usize]
                    + y3_m1 * sh_coeffs[sh_base + 43usize]
                    + y3_0 * sh_coeffs[sh_base + 44usize]
                    + y3_p1 * sh_coeffs[sh_base + 45usize]
                    + y3_p2 * sh_coeffs[sh_base + 46usize]
                    + y3_p3 * sh_coeffs[sh_base + 47usize];
            }
        }
    }

    (
        f32::clamp(color_r + 0.5f32, 0.0f32, 1.0f32),
        f32::clamp(color_g + 0.5f32, 0.0f32, 1.0f32),
        f32::clamp(color_b + 0.5f32, 0.0f32, 1.0f32),
    )
}

/// Native IEEE-754 `f16` variant of [`eval_sh_f32`] reading tightly packed `half::f16`
/// SH storage (`sh_base = splat_idx * sh_words_per_splat * 2`).
#[cube]
pub fn eval_sh_native_f16(
    sh_coeffs: &[half::f16],
    sh_base: usize,
    dir_x: f32,
    dir_y: f32,
    dir_z: f32,
    sh_degree: u32,
) -> (f32, f32, f32) {
    let mut color_r = 0.28209479177387814f32 * f32::cast_from(sh_coeffs[sh_base]);
    let mut color_g = 0.28209479177387814f32 * f32::cast_from(sh_coeffs[sh_base + 1usize]);
    let mut color_b = 0.28209479177387814f32 * f32::cast_from(sh_coeffs[sh_base + 2usize]);

    if sh_degree >= 1u32 {
        let bases_per_ch = if sh_degree == 1u32 {
            4usize
        } else if sh_degree == 2u32 {
            9usize
        } else {
            16usize
        };
        let rest_per_ch = bases_per_ch - 1usize;
        let r_rest = sh_base + 3usize;
        let g_rest = sh_base + 3usize + rest_per_ch;
        let b_rest = sh_base + 3usize + rest_per_ch * 2usize;

        let y1_m1 = -0.4886025119029199f32 * dir_y;
        let y1_0 = 0.4886025119029199f32 * dir_z;
        let y1_p1 = -0.4886025119029199f32 * dir_x;

        color_r += y1_m1 * f32::cast_from(sh_coeffs[r_rest])
            + y1_0 * f32::cast_from(sh_coeffs[r_rest + 1usize])
            + y1_p1 * f32::cast_from(sh_coeffs[r_rest + 2usize]);
        color_g += y1_m1 * f32::cast_from(sh_coeffs[g_rest])
            + y1_0 * f32::cast_from(sh_coeffs[g_rest + 1usize])
            + y1_p1 * f32::cast_from(sh_coeffs[g_rest + 2usize]);
        color_b += y1_m1 * f32::cast_from(sh_coeffs[b_rest])
            + y1_0 * f32::cast_from(sh_coeffs[b_rest + 1usize])
            + y1_p1 * f32::cast_from(sh_coeffs[b_rest + 2usize]);

        if sh_degree >= 2u32 {
            let xx = dir_x * dir_x;
            let yy = dir_y * dir_y;
            let zz = dir_z * dir_z;
            let xy = dir_x * dir_y;
            let yz = dir_y * dir_z;
            let xz = dir_x * dir_z;

            let y2_m2 = 1.0925484305920792f32 * xy;
            let y2_m1 = -1.0925484305920792f32 * yz;
            let y2_0 = 0.31539156525252005f32 * (2.0f32 * zz - xx - yy);
            let y2_p1 = -1.0925484305920792f32 * xz;
            let y2_p2 = 0.5462742152960396f32 * (xx - yy);

            color_r += y2_m2 * f32::cast_from(sh_coeffs[r_rest + 3usize])
                + y2_m1 * f32::cast_from(sh_coeffs[r_rest + 4usize])
                + y2_0 * f32::cast_from(sh_coeffs[r_rest + 5usize])
                + y2_p1 * f32::cast_from(sh_coeffs[r_rest + 6usize])
                + y2_p2 * f32::cast_from(sh_coeffs[r_rest + 7usize]);
            color_g += y2_m2 * f32::cast_from(sh_coeffs[g_rest + 3usize])
                + y2_m1 * f32::cast_from(sh_coeffs[g_rest + 4usize])
                + y2_0 * f32::cast_from(sh_coeffs[g_rest + 5usize])
                + y2_p1 * f32::cast_from(sh_coeffs[g_rest + 6usize])
                + y2_p2 * f32::cast_from(sh_coeffs[g_rest + 7usize]);
            color_b += y2_m2 * f32::cast_from(sh_coeffs[b_rest + 3usize])
                + y2_m1 * f32::cast_from(sh_coeffs[b_rest + 4usize])
                + y2_0 * f32::cast_from(sh_coeffs[b_rest + 5usize])
                + y2_p1 * f32::cast_from(sh_coeffs[b_rest + 6usize])
                + y2_p2 * f32::cast_from(sh_coeffs[b_rest + 7usize]);

            if sh_degree >= 3u32 {
                let y3_m3 = -0.5900435899266435f32 * dir_y * (3.0f32 * xx - yy);
                let y3_m2 = 2.890611442640554f32 * xy * dir_z;
                let y3_m1 = -0.4570457994644658f32 * dir_y * (4.0f32 * zz - xx - yy);
                let y3_0 =
                    0.3731763325901154f32 * dir_z * (2.0f32 * zz - 3.0f32 * xx - 3.0f32 * yy);
                let y3_p1 = -0.4570457994644658f32 * dir_x * (4.0f32 * zz - xx - yy);
                let y3_p2 = 1.445305721320277f32 * dir_z * (xx - yy);
                let y3_p3 = -0.5900435899266435f32 * dir_x * (xx - 3.0f32 * yy);

                color_r += y3_m3 * f32::cast_from(sh_coeffs[r_rest + 8usize])
                    + y3_m2 * f32::cast_from(sh_coeffs[r_rest + 9usize])
                    + y3_m1 * f32::cast_from(sh_coeffs[r_rest + 10usize])
                    + y3_0 * f32::cast_from(sh_coeffs[r_rest + 11usize])
                    + y3_p1 * f32::cast_from(sh_coeffs[r_rest + 12usize])
                    + y3_p2 * f32::cast_from(sh_coeffs[r_rest + 13usize])
                    + y3_p3 * f32::cast_from(sh_coeffs[r_rest + 14usize]);
                color_g += y3_m3 * f32::cast_from(sh_coeffs[g_rest + 8usize])
                    + y3_m2 * f32::cast_from(sh_coeffs[g_rest + 9usize])
                    + y3_m1 * f32::cast_from(sh_coeffs[g_rest + 10usize])
                    + y3_0 * f32::cast_from(sh_coeffs[g_rest + 11usize])
                    + y3_p1 * f32::cast_from(sh_coeffs[g_rest + 12usize])
                    + y3_p2 * f32::cast_from(sh_coeffs[g_rest + 13usize])
                    + y3_p3 * f32::cast_from(sh_coeffs[g_rest + 14usize]);
                color_b += y3_m3 * f32::cast_from(sh_coeffs[b_rest + 8usize])
                    + y3_m2 * f32::cast_from(sh_coeffs[b_rest + 9usize])
                    + y3_m1 * f32::cast_from(sh_coeffs[b_rest + 10usize])
                    + y3_0 * f32::cast_from(sh_coeffs[b_rest + 11usize])
                    + y3_p1 * f32::cast_from(sh_coeffs[b_rest + 12usize])
                    + y3_p2 * f32::cast_from(sh_coeffs[b_rest + 13usize])
                    + y3_p3 * f32::cast_from(sh_coeffs[b_rest + 14usize]);
            }
        }
    }

    (
        f32::clamp(color_r + 0.5f32, 0.0f32, 1.0f32),
        f32::clamp(color_g + 0.5f32, 0.0f32, 1.0f32),
        f32::clamp(color_b + 0.5f32, 0.0f32, 1.0f32),
    )
}

/// Fused Stage 2 GPU kernel performing EWA 3D-to-2D projection, tight opacity-aware
/// ellipse AABB computation, exact $R^T d$ inverse-ray view-dependent SH evaluation,
/// and workgroup-aggregated atomic stream compaction.
///
/// Writes 10 floats per splat to `splat_attrs`:
/// `[mu_x, mu_y, a, b, c, alpha_2d, rgb_r, rgb_g, rgb_b, tz]`, and packed
/// `(ext_x & 0xFFFF) | ((ext_y & 0xFFFF) << 16)` to `extents_packed` (`0` when culled).
#[allow(clippy::too_many_arguments, clippy::unnecessary_cast)]
#[cube(launch)]
pub fn project_gaussians_v2_kernel(
    positions: &[f32],
    covariances: &[f32],
    opacities: &[f32],
    sh_coeffs: &[f32],
    bindings_tet: &[u32],
    tet_rotations: &[f32],
    split_meta: &[u32],
    split_counter: &[u32],
    camera: &[f32],
    splat_attrs: &mut [f32],
    extents_packed: &mut [u32],
    surviving_ids: &mut [u32],
    surviving_counter: &mut [Atomic<u32>],
    cam_pos_x: f32,
    cam_pos_y: f32,
    cam_pos_z: f32,
    mip_filter_3d_var: f32,
    min_opacity: f32,
    min_radius_px: f32,
    sh_degree: u32,
    has_deformation: u32,
    num_tets: u32,
    num_rest_gaussians: u32,
) {
    let idx = ABSOLUTE_POS;
    let lid = UNIT_POS as usize;
    let active_gaussians = num_rest_gaussians + split_counter[0usize];
    let mut alive = idx < active_gaussians as usize;

    let mut px = 0.0f32;
    let mut py = 0.0f32;
    let mut pz = 0.0f32;
    let mut tx = 0.0f32;
    let mut ty = 0.0f32;
    let mut tz = 0.0f32;
    let mut alpha_3d = 0.0f32;
    let mut v_xx = 0.0f32;
    let mut v_xy = 0.0f32;
    let mut v_xz = 0.0f32;
    let mut v_yy = 0.0f32;
    let mut v_yz = 0.0f32;
    let mut v_zz = 0.0f32;

    let w00 = camera[0usize];
    let w10 = camera[1usize];
    let w20 = camera[2usize];
    let w01 = camera[4usize];
    let w11 = camera[5usize];
    let w21 = camera[6usize];
    let w02 = camera[8usize];
    let w12 = camera[9usize];
    let w22 = camera[10usize];
    let w03 = camera[12usize];
    let w13 = camera[13usize];
    let w23 = camera[14usize];
    let focal_x = camera[16usize];
    let focal_y = camera[17usize];
    let principal_x = camera[18usize];
    let principal_y = camera[19usize];
    let viewport_width = camera[20usize];
    let viewport_height = camera[21usize];
    let near_plane = camera[22usize];

    if alive {
        extents_packed[idx] = 0u32;
        let p_base = idx * 3usize;
        px = positions[p_base];
        py = positions[p_base + 1usize];
        pz = positions[p_base + 2usize];

        tx = w00 * px + w01 * py + w02 * pz + w03;
        ty = w10 * px + w11 * py + w12 * pz + w13;
        tz = w20 * px + w21 * py + w22 * pz + w23;

        if tz <= near_plane {
            alive = false;
        }
    }

    if alive {
        let op = opacities[idx];
        if op < min_opacity {
            alive = false;
        } else {
            let cov_base = idx * 6usize;
            v_xx = covariances[cov_base];
            v_xy = covariances[cov_base + 1usize];
            v_xz = covariances[cov_base + 2usize];
            v_yy = covariances[cov_base + 3usize];
            v_yz = covariances[cov_base + 4usize];
            v_zz = covariances[cov_base + 5usize];
            alpha_3d = op;

            if mip_filter_3d_var > 0.0f32 {
                let det_orig = f32::max(
                    v_xx * (v_yy * v_zz - v_yz * v_yz) - v_xy * (v_xy * v_zz - v_xz * v_yz)
                        + v_xz * (v_xy * v_yz - v_xz * v_yy),
                    1e-24f32,
                );
                v_xx += mip_filter_3d_var;
                v_yy += mip_filter_3d_var;
                v_zz += mip_filter_3d_var;
                let det_reg = f32::max(
                    v_xx * (v_yy * v_zz - v_yz * v_yz) - v_xy * (v_xy * v_zz - v_xz * v_yz)
                        + v_xz * (v_xy * v_yz - v_xz * v_yy),
                    det_orig,
                );
                let comp = f32::sqrt(f32::clamp(det_orig / det_reg, 0.0f32, 1.0f32));
                alpha_3d = f32::clamp(op * comp, 0.0f32, 1.0f32);
                if alpha_3d < min_opacity {
                    alive = false;
                }
            }
        }
    }

    if alive {
        let k00 = w00 * v_xx + w01 * v_xy + w02 * v_xz;
        let k01 = w00 * v_xy + w01 * v_yy + w02 * v_yz;
        let k02 = w00 * v_xz + w01 * v_yz + w02 * v_zz;
        let k10 = w10 * v_xx + w11 * v_xy + w12 * v_xz;
        let k11 = w10 * v_xy + w11 * v_yy + w12 * v_yz;
        let k12 = w10 * v_xz + w11 * v_yz + w12 * v_zz;
        let k20 = w20 * v_xx + w21 * v_xy + w22 * v_xz;
        let k21 = w20 * v_xy + w21 * v_yy + w22 * v_yz;
        let k22 = w20 * v_xz + w21 * v_yz + w22 * v_zz;

        let cam_xx = k00 * w00 + k01 * w01 + k02 * w02;
        let cam_xy = k00 * w10 + k01 * w11 + k02 * w12;
        let cam_xz = k00 * w20 + k01 * w21 + k02 * w22;
        let cam_yy = k10 * w10 + k11 * w11 + k12 * w12;
        let cam_yz = k10 * w20 + k11 * w21 + k12 * w22;
        let cam_zz = k20 * w20 + k21 * w21 + k22 * w22;

        let inv_tz = 1.0f32 / tz;
        let inv_tz2 = inv_tz * inv_tz;
        let j00 = focal_x * inv_tz;
        let j02 = -focal_x * tx * inv_tz2;
        let j11 = focal_y * inv_tz;
        let j12 = -focal_y * ty * inv_tz2;

        let m00 = j00 * cam_xx + j02 * cam_xz;
        let m01 = j00 * cam_xy + j02 * cam_yz;
        let m02 = j00 * cam_xz + j02 * cam_zz;
        let m11 = j11 * cam_yy + j12 * cam_yz;
        let m12 = j11 * cam_yz + j12 * cam_zz;

        let sig2d_xx = m00 * j00 + m02 * j02;
        let sig2d_xy = m01 * j11 + m02 * j12;
        let sig2d_yy = m11 * j11 + m12 * j12;

        let raw_mid = 0.5f32 * (sig2d_xx + sig2d_yy);
        let raw_det = f32::max(sig2d_xx * sig2d_yy - sig2d_xy * sig2d_xy, 0.0f32);
        let raw_discr = f32::sqrt(f32::max(raw_mid * raw_mid - raw_det, 0.0f32));
        let raw_lambda_max = f32::max(raw_mid + raw_discr, 0.0f32);

        if 3.0f32 * f32::sqrt(raw_lambda_max) < min_radius_px {
            alive = false;
        } else {
            let sig_prime_xx = sig2d_xx + 0.3f32;
            let sig_prime_xy = sig2d_xy;
            let sig_prime_yy = sig2d_yy + 0.3f32;
            let det = sig_prime_xx * sig_prime_yy - sig_prime_xy * sig_prime_xy;

            if det <= 1e-6f32 {
                alive = false;
            } else {
                let comp_2d = f32::sqrt(f32::clamp(raw_det / det, 0.0f32, 1.0f32));
                let alpha_2d = f32::clamp(alpha_3d * comp_2d, 0.0f32, 1.0f32);
                if alpha_2d < min_opacity || alpha_2d <= (1.0f32 / 255.0f32) {
                    alive = false;
                } else {
                    let q_max = f32::clamp(2.0f32 * f32::ln(255.0f32 * alpha_2d), 0.0f32, 8.0f32);
                    let sigma_cut = f32::sqrt(q_max);
                    let ext_x = f32::clamp(
                        f32::ceil(sigma_cut * f32::sqrt(f32::max(sig_prime_xx, 0.0f32))),
                        1.0f32,
                        1024.0f32,
                    );
                    let ext_y = f32::clamp(
                        f32::ceil(sigma_cut * f32::sqrt(f32::max(sig_prime_yy, 0.0f32))),
                        1.0f32,
                        1024.0f32,
                    );
                    let mu_x = focal_x * tx * inv_tz + principal_x;
                    let mu_y = focal_y * ty * inv_tz + principal_y;

                    if mu_x + ext_x < 0.0f32
                        || mu_x - ext_x >= viewport_width
                        || mu_y + ext_y < 0.0f32
                        || mu_y - ext_y >= viewport_height
                    {
                        alive = false;
                    } else {
                        let inv_det = 1.0f32 / det;
                        let a = sig_prime_yy * inv_det;
                        let b = -sig_prime_xy * inv_det;
                        let c = sig_prime_xx * inv_det;

                        let vx = px - cam_pos_x;
                        let vy = py - cam_pos_y;
                        let vz = pz - cam_pos_z;
                        let inv_len =
                            1.0f32 / f32::max(0.0001f32, f32::sqrt(vx * vx + vy * vy + vz * vz));
                        let mut dir_x = vx * inv_len;
                        let mut dir_y = vy * inv_len;
                        let mut dir_z = vz * inv_len;

                        let mut sh_idx = idx;
                        let mut tet_raw = num_tets;
                        if idx < num_rest_gaussians as usize {
                            if has_deformation != 0u32 {
                                tet_raw = bindings_tet[idx];
                            }
                        } else {
                            let child_off = idx - num_rest_gaussians as usize;
                            let meta_base = child_off * 2usize;
                            sh_idx = split_meta[meta_base] as usize;
                            tet_raw = split_meta[meta_base + 1usize];
                        }

                        if has_deformation != 0u32 && tet_raw < num_tets {
                            let r_base = (tet_raw as usize) * 9usize;
                            let r00 = tet_rotations[r_base];
                            let r10 = tet_rotations[r_base + 1usize];
                            let r20 = tet_rotations[r_base + 2usize];
                            let r01 = tet_rotations[r_base + 3usize];
                            let r11 = tet_rotations[r_base + 4usize];
                            let r21 = tet_rotations[r_base + 5usize];
                            let r02 = tet_rotations[r_base + 6usize];
                            let r12 = tet_rotations[r_base + 7usize];
                            let r22 = tet_rotations[r_base + 8usize];

                            let ldx = r00 * dir_x + r10 * dir_y + r20 * dir_z;
                            let ldy = r01 * dir_x + r11 * dir_y + r21 * dir_z;
                            let ldz = r02 * dir_x + r12 * dir_y + r22 * dir_z;
                            dir_x = ldx;
                            dir_y = ldy;
                            dir_z = ldz;
                        }

                        let (rgb_r, rgb_g, rgb_b) = eval_sh_f32(
                            sh_coeffs,
                            sh_idx * 48usize,
                            dir_x,
                            dir_y,
                            dir_z,
                            sh_degree,
                        );

                        let attr_base = idx * 10usize;
                        splat_attrs[attr_base] = mu_x;
                        splat_attrs[attr_base + 1usize] = mu_y;
                        splat_attrs[attr_base + 2usize] = a;
                        splat_attrs[attr_base + 3usize] = b;
                        splat_attrs[attr_base + 4usize] = c;
                        splat_attrs[attr_base + 5usize] = alpha_2d;
                        splat_attrs[attr_base + 6usize] = rgb_r;
                        splat_attrs[attr_base + 7usize] = rgb_g;
                        splat_attrs[attr_base + 8usize] = rgb_b;
                        splat_attrs[attr_base + 9usize] = tz;

                        extents_packed[idx] =
                            ((ext_x as u32) & 0xFFFFu32) | (((ext_y as u32) & 0xFFFFu32) << 16u32);
                    }
                }
            }
        }
    }

    // Workgroup-aggregated Blelloch exclusive scan over surviving flags (uniform control flow).
    let mut scan = Shared::<[u32]>::new_slice(256usize);
    let mut wg_base_shared = Shared::<[u32]>::new_slice(1usize);
    scan[lid] = if alive { 1u32 } else { 0u32 };
    sync_cube();

    let mut offset = 1usize;
    while offset < 256usize {
        let step = offset * 2usize;
        let active_pairs = 256usize / step;
        if lid < active_pairs {
            let ai = step * lid + offset - 1usize;
            let bi = step * lid + step - 1usize;
            scan[bi] += scan[ai];
        }
        sync_cube();
        offset = step;
    }

    if lid == 0usize {
        let wg_total = scan[255usize];
        let mut wg_base = 0u32;
        if wg_total > 0u32 {
            wg_base = surviving_counter[0usize].fetch_add(wg_total);
        }
        wg_base_shared[0usize] = wg_base;
        scan[255usize] = 0u32;
    }
    sync_cube();

    let mut down_offset = 128usize;
    while down_offset >= 1usize {
        let step = down_offset * 2usize;
        let active_pairs = 256usize / step;
        if lid < active_pairs {
            let ai = step * lid + down_offset - 1usize;
            let bi = step * lid + step - 1usize;
            let tmp = scan[ai];
            scan[ai] = scan[bi];
            scan[bi] += tmp;
        }
        sync_cube();
        down_offset /= 2usize;
    }

    if alive {
        let global_rank = wg_base_shared[0usize] + scan[lid];
        surviving_ids[global_rank as usize] = idx as u32;
    }
}

/// Native `half::f16` variant of [`project_gaussians_v2_kernel`] reading 16-bit
/// rest covariances (`6` `f16` per splat) and 16-bit packed SH coefficients.
#[allow(
    clippy::too_many_arguments,
    clippy::unnecessary_cast,
    clippy::manual_div_ceil
)]
#[cube(launch)]
pub fn project_gaussians_v2_f16_kernel(
    positions: &[f32],
    covariances_f16: &[half::f16],
    opacities: &[f32],
    sh_coeffs_f16: &[half::f16],
    bindings_tet: &[u32],
    tet_rotations: &[f32],
    split_meta: &[u32],
    split_counter: &[u32],
    camera: &[f32],
    splat_attrs: &mut [f32],
    extents_packed: &mut [u32],
    surviving_ids: &mut [u32],
    surviving_counter: &mut [Atomic<u32>],
    cam_pos_x: f32,
    cam_pos_y: f32,
    cam_pos_z: f32,
    mip_filter_3d_var: f32,
    min_opacity: f32,
    min_radius_px: f32,
    sh_degree: u32,
    has_deformation: u32,
    num_tets: u32,
    num_rest_gaussians: u32,
) {
    let idx = ABSOLUTE_POS;
    let lid = UNIT_POS as usize;
    let active_gaussians = num_rest_gaussians + split_counter[0usize];
    let mut alive = idx < active_gaussians as usize;

    let mut px = 0.0f32;
    let mut py = 0.0f32;
    let mut pz = 0.0f32;
    let mut tx = 0.0f32;
    let mut ty = 0.0f32;
    let mut tz = 0.0f32;
    let mut alpha_3d = 0.0f32;
    let mut v_xx = 0.0f32;
    let mut v_xy = 0.0f32;
    let mut v_xz = 0.0f32;
    let mut v_yy = 0.0f32;
    let mut v_yz = 0.0f32;
    let mut v_zz = 0.0f32;

    let w00 = camera[0usize];
    let w10 = camera[1usize];
    let w20 = camera[2usize];
    let w01 = camera[4usize];
    let w11 = camera[5usize];
    let w21 = camera[6usize];
    let w02 = camera[8usize];
    let w12 = camera[9usize];
    let w22 = camera[10usize];
    let w03 = camera[12usize];
    let w13 = camera[13usize];
    let w23 = camera[14usize];
    let focal_x = camera[16usize];
    let focal_y = camera[17usize];
    let principal_x = camera[18usize];
    let principal_y = camera[19usize];
    let viewport_width = camera[20usize];
    let viewport_height = camera[21usize];
    let near_plane = camera[22usize];

    if alive {
        extents_packed[idx] = 0u32;
        let p_base = idx * 3usize;
        px = positions[p_base];
        py = positions[p_base + 1usize];
        pz = positions[p_base + 2usize];

        tx = w00 * px + w01 * py + w02 * pz + w03;
        ty = w10 * px + w11 * py + w12 * pz + w13;
        tz = w20 * px + w21 * py + w22 * pz + w23;

        if tz <= near_plane {
            alive = false;
        }
    }

    if alive {
        let op = opacities[idx];
        if op < min_opacity {
            alive = false;
        } else {
            let cov_base = idx * 6usize;
            v_xx = f32::cast_from(covariances_f16[cov_base]);
            v_xy = f32::cast_from(covariances_f16[cov_base + 1usize]);
            v_xz = f32::cast_from(covariances_f16[cov_base + 2usize]);
            v_yy = f32::cast_from(covariances_f16[cov_base + 3usize]);
            v_yz = f32::cast_from(covariances_f16[cov_base + 4usize]);
            v_zz = f32::cast_from(covariances_f16[cov_base + 5usize]);
            alpha_3d = op;

            if mip_filter_3d_var > 0.0f32 {
                let det_orig = f32::max(
                    v_xx * (v_yy * v_zz - v_yz * v_yz) - v_xy * (v_xy * v_zz - v_xz * v_yz)
                        + v_xz * (v_xy * v_yz - v_xz * v_yy),
                    1e-24f32,
                );
                v_xx += mip_filter_3d_var;
                v_yy += mip_filter_3d_var;
                v_zz += mip_filter_3d_var;
                let det_reg = f32::max(
                    v_xx * (v_yy * v_zz - v_yz * v_yz) - v_xy * (v_xy * v_zz - v_xz * v_yz)
                        + v_xz * (v_xy * v_yz - v_xz * v_yy),
                    det_orig,
                );
                let comp = f32::sqrt(f32::clamp(det_orig / det_reg, 0.0f32, 1.0f32));
                alpha_3d = f32::clamp(op * comp, 0.0f32, 1.0f32);
                if alpha_3d < min_opacity {
                    alive = false;
                }
            }
        }
    }

    if alive {
        let k00 = w00 * v_xx + w01 * v_xy + w02 * v_xz;
        let k01 = w00 * v_xy + w01 * v_yy + w02 * v_yz;
        let k02 = w00 * v_xz + w01 * v_yz + w02 * v_zz;
        let k10 = w10 * v_xx + w11 * v_xy + w12 * v_xz;
        let k11 = w10 * v_xy + w11 * v_yy + w12 * v_yz;
        let k12 = w10 * v_xz + w11 * v_yz + w12 * v_zz;
        let k20 = w20 * v_xx + w21 * v_xy + w22 * v_xz;
        let k21 = w20 * v_xy + w21 * v_yy + w22 * v_yz;
        let k22 = w20 * v_xz + w21 * v_yz + w22 * v_zz;

        let cam_xx = k00 * w00 + k01 * w01 + k02 * w02;
        let cam_xy = k00 * w10 + k01 * w11 + k02 * w12;
        let cam_xz = k00 * w20 + k01 * w21 + k02 * w22;
        let cam_yy = k10 * w10 + k11 * w11 + k12 * w12;
        let cam_yz = k10 * w20 + k11 * w21 + k12 * w22;
        let cam_zz = k20 * w20 + k21 * w21 + k22 * w22;

        let inv_tz = 1.0f32 / tz;
        let inv_tz2 = inv_tz * inv_tz;
        let j00 = focal_x * inv_tz;
        let j02 = -focal_x * tx * inv_tz2;
        let j11 = focal_y * inv_tz;
        let j12 = -focal_y * ty * inv_tz2;

        let m00 = j00 * cam_xx + j02 * cam_xz;
        let m01 = j00 * cam_xy + j02 * cam_yz;
        let m02 = j00 * cam_xz + j02 * cam_zz;
        let m11 = j11 * cam_yy + j12 * cam_yz;
        let m12 = j11 * cam_yz + j12 * cam_zz;

        let sig2d_xx = m00 * j00 + m02 * j02;
        let sig2d_xy = m01 * j11 + m02 * j12;
        let sig2d_yy = m11 * j11 + m12 * j12;

        let raw_mid = 0.5f32 * (sig2d_xx + sig2d_yy);
        let raw_det = f32::max(sig2d_xx * sig2d_yy - sig2d_xy * sig2d_xy, 0.0f32);
        let raw_discr = f32::sqrt(f32::max(raw_mid * raw_mid - raw_det, 0.0f32));
        let raw_lambda_max = f32::max(raw_mid + raw_discr, 0.0f32);

        if 3.0f32 * f32::sqrt(raw_lambda_max) < min_radius_px {
            alive = false;
        } else {
            let sig_prime_xx = sig2d_xx + 0.3f32;
            let sig_prime_xy = sig2d_xy;
            let sig_prime_yy = sig2d_yy + 0.3f32;
            let det = sig_prime_xx * sig_prime_yy - sig_prime_xy * sig_prime_xy;

            if det <= 1e-6f32 {
                alive = false;
            } else {
                let comp_2d = f32::sqrt(f32::clamp(raw_det / det, 0.0f32, 1.0f32));
                let alpha_2d = f32::clamp(alpha_3d * comp_2d, 0.0f32, 1.0f32);
                if alpha_2d < min_opacity || alpha_2d <= (1.0f32 / 255.0f32) {
                    alive = false;
                } else {
                    let q_max = f32::clamp(2.0f32 * f32::ln(255.0f32 * alpha_2d), 0.0f32, 8.0f32);
                    let sigma_cut = f32::sqrt(q_max);
                    let ext_x = f32::clamp(
                        f32::ceil(sigma_cut * f32::sqrt(f32::max(sig_prime_xx, 0.0f32))),
                        1.0f32,
                        1024.0f32,
                    );
                    let ext_y = f32::clamp(
                        f32::ceil(sigma_cut * f32::sqrt(f32::max(sig_prime_yy, 0.0f32))),
                        1.0f32,
                        1024.0f32,
                    );
                    let mu_x = focal_x * tx * inv_tz + principal_x;
                    let mu_y = focal_y * ty * inv_tz + principal_y;

                    if mu_x + ext_x < 0.0f32
                        || mu_x - ext_x >= viewport_width
                        || mu_y + ext_y < 0.0f32
                        || mu_y - ext_y >= viewport_height
                    {
                        alive = false;
                    } else {
                        let inv_det = 1.0f32 / det;
                        let a = sig_prime_yy * inv_det;
                        let b = -sig_prime_xy * inv_det;
                        let c = sig_prime_xx * inv_det;

                        let vx = px - cam_pos_x;
                        let vy = py - cam_pos_y;
                        let vz = pz - cam_pos_z;
                        let inv_len =
                            1.0f32 / f32::max(0.0001f32, f32::sqrt(vx * vx + vy * vy + vz * vz));
                        let mut dir_x = vx * inv_len;
                        let mut dir_y = vy * inv_len;
                        let mut dir_z = vz * inv_len;

                        let mut sh_idx = idx;
                        let mut tet_raw = num_tets;
                        if idx < num_rest_gaussians as usize {
                            if has_deformation != 0u32 {
                                tet_raw = bindings_tet[idx];
                            }
                        } else {
                            let child_off = idx - num_rest_gaussians as usize;
                            let meta_base = child_off * 2usize;
                            sh_idx = split_meta[meta_base] as usize;
                            tet_raw = split_meta[meta_base + 1usize];
                        }

                        if has_deformation != 0u32 && tet_raw < num_tets {
                            let r_base = (tet_raw as usize) * 9usize;
                            let r00 = tet_rotations[r_base];
                            let r10 = tet_rotations[r_base + 1usize];
                            let r20 = tet_rotations[r_base + 2usize];
                            let r01 = tet_rotations[r_base + 3usize];
                            let r11 = tet_rotations[r_base + 4usize];
                            let r21 = tet_rotations[r_base + 5usize];
                            let r02 = tet_rotations[r_base + 6usize];
                            let r12 = tet_rotations[r_base + 7usize];
                            let r22 = tet_rotations[r_base + 8usize];

                            let ldx = r00 * dir_x + r10 * dir_y + r20 * dir_z;
                            let ldy = r01 * dir_x + r11 * dir_y + r21 * dir_z;
                            let ldz = r02 * dir_x + r12 * dir_y + r22 * dir_z;
                            dir_x = ldx;
                            dir_y = ldy;
                            dir_z = ldz;
                        }

                        let bases_per_ch = if sh_degree == 0u32 {
                            1usize
                        } else if sh_degree == 1u32 {
                            4usize
                        } else if sh_degree == 2u32 {
                            9usize
                        } else {
                            16usize
                        };
                        let sh_f16_stride = ((bases_per_ch * 3usize + 1usize) / 2usize) * 2usize;
                        let (rgb_r, rgb_g, rgb_b) = eval_sh_native_f16(
                            sh_coeffs_f16,
                            sh_idx * sh_f16_stride,
                            dir_x,
                            dir_y,
                            dir_z,
                            sh_degree,
                        );

                        let attr_base = idx * 10usize;
                        splat_attrs[attr_base] = mu_x;
                        splat_attrs[attr_base + 1usize] = mu_y;
                        splat_attrs[attr_base + 2usize] = a;
                        splat_attrs[attr_base + 3usize] = b;
                        splat_attrs[attr_base + 4usize] = c;
                        splat_attrs[attr_base + 5usize] = alpha_2d;
                        splat_attrs[attr_base + 6usize] = rgb_r;
                        splat_attrs[attr_base + 7usize] = rgb_g;
                        splat_attrs[attr_base + 8usize] = rgb_b;
                        splat_attrs[attr_base + 9usize] = tz;

                        extents_packed[idx] =
                            ((ext_x as u32) & 0xFFFFu32) | (((ext_y as u32) & 0xFFFFu32) << 16u32);
                    }
                }
            }
        }
    }

    let mut scan = Shared::<[u32]>::new_slice(256usize);
    let mut wg_base_shared = Shared::<[u32]>::new_slice(1usize);
    scan[lid] = if alive { 1u32 } else { 0u32 };
    sync_cube();

    let mut offset = 1usize;
    while offset < 256usize {
        let step = offset * 2usize;
        let active_pairs = 256usize / step;
        if lid < active_pairs {
            let ai = step * lid + offset - 1usize;
            let bi = step * lid + step - 1usize;
            scan[bi] += scan[ai];
        }
        sync_cube();
        offset = step;
    }

    if lid == 0usize {
        let wg_total = scan[255usize];
        let mut wg_base = 0u32;
        if wg_total > 0u32 {
            wg_base = surviving_counter[0usize].fetch_add(wg_total);
        }
        wg_base_shared[0usize] = wg_base;
        scan[255usize] = 0u32;
    }
    sync_cube();

    let mut down_offset = 128usize;
    while down_offset >= 1usize {
        let step = down_offset * 2usize;
        let active_pairs = 256usize / step;
        if lid < active_pairs {
            let ai = step * lid + down_offset - 1usize;
            let bi = step * lid + step - 1usize;
            let tmp = scan[ai];
            scan[ai] = scan[bi];
            scan[bi] += tmp;
        }
        sync_cube();
        down_offset /= 2usize;
    }

    if alive {
        let global_rank = wg_base_shared[0usize] + scan[lid];
        surviving_ids[global_rank as usize] = idx as u32;
    }
}

/// Native `half::f16` counterpart of [`project_gaussians_quantized_kernel`]
/// reading `6` hardware `f16` scalars per splat when `SHADER_F16` is available.
#[allow(clippy::too_many_arguments, clippy::unnecessary_cast)]
#[cube(launch)]
pub fn project_gaussians_quantized_f16_kernel(
    positions: &[f32],
    covariances_f16: &[half::f16],
    opacities: &[f32],
    camera: &[f32],
    mip_filter_3d_var: f32,
    min_opacity: f32,
    min_radius_px: f32,
    points_xy: &mut [f32],
    depths: &mut [f32],
    radii: &mut [i32],
    conics: &mut [f32],
    out_opacities: &mut [f32],
    visibility_flags: &mut [u32],
    num_gaussians: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= num_gaussians as usize {
        terminate!();
    }

    visibility_flags[idx] = 0u32;
    radii[idx] = 0;
    out_opacities[idx] = 0.0f32;

    let pt_base = idx * 2usize;
    let conic_base = idx * 3usize;
    points_xy[pt_base] = 0.0f32;
    points_xy[pt_base + 1usize] = 0.0f32;
    conics[conic_base] = 0.0f32;
    conics[conic_base + 1usize] = 0.0f32;
    conics[conic_base + 2usize] = 0.0f32;

    let p_base = idx * 3usize;
    let px = positions[p_base];
    let py = positions[p_base + 1usize];
    let pz = positions[p_base + 2usize];

    let w00 = camera[0usize];
    let w10 = camera[1usize];
    let w20 = camera[2usize];
    let w01 = camera[4usize];
    let w11 = camera[5usize];
    let w21 = camera[6usize];
    let w02 = camera[8usize];
    let w12 = camera[9usize];
    let w22 = camera[10usize];
    let w03 = camera[12usize];
    let w13 = camera[13usize];
    let w23 = camera[14usize];

    let focal_x = camera[16usize];
    let focal_y = camera[17usize];
    let principal_x = camera[18usize];
    let principal_y = camera[19usize];
    let viewport_width = camera[20usize];
    let viewport_height = camera[21usize];
    let near_plane = camera[22usize];

    let tx = w00 * px + w01 * py + w02 * pz + w03;
    let ty = w10 * px + w11 * py + w12 * pz + w13;
    let tz = w20 * px + w21 * py + w22 * pz + w23;
    depths[idx] = tz;

    if tz <= near_plane {
        terminate!();
    }

    let op = opacities[idx];
    if op < min_opacity {
        terminate!();
    }

    let cov_base = idx * 6usize;
    let mut v_xx = f32::cast_from(covariances_f16[cov_base]);
    let v_xy = f32::cast_from(covariances_f16[cov_base + 1usize]);
    let v_xz = f32::cast_from(covariances_f16[cov_base + 2usize]);
    let mut v_yy = f32::cast_from(covariances_f16[cov_base + 3usize]);
    let v_yz = f32::cast_from(covariances_f16[cov_base + 4usize]);
    let mut v_zz = f32::cast_from(covariances_f16[cov_base + 5usize]);

    let mut alpha_3d = op;
    if mip_filter_3d_var > 0.0f32 {
        let det_orig = f32::max(
            v_xx * (v_yy * v_zz - v_yz * v_yz) - v_xy * (v_xy * v_zz - v_xz * v_yz)
                + v_xz * (v_xy * v_yz - v_xz * v_yy),
            1e-24f32,
        );
        v_xx += mip_filter_3d_var;
        v_yy += mip_filter_3d_var;
        v_zz += mip_filter_3d_var;
        let det_reg = f32::max(
            v_xx * (v_yy * v_zz - v_yz * v_yz) - v_xy * (v_xy * v_zz - v_xz * v_yz)
                + v_xz * (v_xy * v_yz - v_xz * v_yy),
            det_orig,
        );
        let comp = f32::sqrt(f32::clamp(det_orig / det_reg, 0.0f32, 1.0f32));
        alpha_3d = f32::clamp(op * comp, 0.0f32, 1.0f32);
        if alpha_3d < min_opacity {
            terminate!();
        }
    }

    let k00 = w00 * v_xx + w01 * v_xy + w02 * v_xz;
    let k01 = w00 * v_xy + w01 * v_yy + w02 * v_yz;
    let k02 = w00 * v_xz + w01 * v_yz + w02 * v_zz;
    let k10 = w10 * v_xx + w11 * v_xy + w12 * v_xz;
    let k11 = w10 * v_xy + w11 * v_yy + w12 * v_yz;
    let k12 = w10 * v_xz + w11 * v_yz + w12 * v_zz;
    let k20 = w20 * v_xx + w21 * v_xy + w22 * v_xz;
    let k21 = w20 * v_xy + w21 * v_yy + w22 * v_yz;
    let k22 = w20 * v_xz + w21 * v_yz + w22 * v_zz;

    let cam_xx = k00 * w00 + k01 * w01 + k02 * w02;
    let cam_xy = k00 * w10 + k01 * w11 + k02 * w12;
    let cam_xz = k00 * w20 + k01 * w21 + k02 * w22;
    let cam_yy = k10 * w10 + k11 * w11 + k12 * w12;
    let cam_yz = k10 * w20 + k11 * w21 + k12 * w22;
    let cam_zz = k20 * w20 + k21 * w21 + k22 * w22;

    let inv_tz = 1.0f32 / tz;
    let inv_tz2 = inv_tz * inv_tz;
    let j00 = focal_x * inv_tz;
    let j02 = -focal_x * tx * inv_tz2;
    let j11 = focal_y * inv_tz;
    let j12 = -focal_y * ty * inv_tz2;

    let m00 = j00 * cam_xx + j02 * cam_xz;
    let m01 = j00 * cam_xy + j02 * cam_yz;
    let m02 = j00 * cam_xz + j02 * cam_zz;
    let m11 = j11 * cam_yy + j12 * cam_yz;
    let m12 = j11 * cam_yz + j12 * cam_zz;

    let sig2d_xx = m00 * j00 + m02 * j02;
    let sig2d_xy = m01 * j11 + m02 * j12;
    let sig2d_yy = m11 * j11 + m12 * j12;

    let raw_mid = 0.5f32 * (sig2d_xx + sig2d_yy);
    let raw_det = f32::max(sig2d_xx * sig2d_yy - sig2d_xy * sig2d_xy, 0.0f32);
    let raw_discr = f32::sqrt(f32::max(raw_mid * raw_mid - raw_det, 0.0f32));
    let raw_lambda_max = f32::max(raw_mid + raw_discr, 0.0f32);
    if 3.0f32 * f32::sqrt(raw_lambda_max) < min_radius_px {
        terminate!();
    }

    let sig_prime_xx = sig2d_xx + 0.3f32;
    let sig_prime_xy = sig2d_xy;
    let sig_prime_yy = sig2d_yy + 0.3f32;
    let det = sig_prime_xx * sig_prime_yy - sig_prime_xy * sig_prime_xy;
    if det <= 1e-6f32 {
        terminate!();
    }

    let comp_2d = f32::sqrt(f32::clamp(raw_det / det, 0.0f32, 1.0f32));
    let alpha_2d = f32::clamp(alpha_3d * comp_2d, 0.0f32, 1.0f32);
    if alpha_2d < min_opacity {
        terminate!();
    }

    let inv_det = 1.0f32 / det;
    let a = sig_prime_yy * inv_det;
    let b = -sig_prime_xy * inv_det;
    let c = sig_prime_xx * inv_det;

    let mid = 0.5f32 * (sig_prime_xx + sig_prime_yy);
    let term = f32::max(0.1f32, mid * mid - det);
    let delta = f32::sqrt(term);
    let lambda1 = mid + delta;
    let lambda2 = f32::max(0.1f32, mid - delta);
    let lambda_max = f32::max(lambda1, lambda2);
    let radius = f32::clamp(f32::ceil(3.0f32 * f32::sqrt(lambda_max)), 1.0f32, 1024.0f32);

    let mu_x = focal_x * tx * inv_tz + principal_x;
    let mu_y = focal_y * ty * inv_tz + principal_y;
    points_xy[pt_base] = mu_x;
    points_xy[pt_base + 1usize] = mu_y;
    conics[conic_base] = a;
    conics[conic_base + 1usize] = b;
    conics[conic_base + 2usize] = c;

    if mu_x + radius < 0.0f32
        || mu_x - radius >= viewport_width
        || mu_y + radius < 0.0f32
        || mu_y - radius >= viewport_height
    {
        terminate!();
    }

    radii[idx] = radius as i32;
    out_opacities[idx] = alpha_2d;
    visibility_flags[idx] = 1u32;
}
