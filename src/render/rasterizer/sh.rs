#![allow(clippy::excessive_precision)]

use glam::Vec3;

// Spherical Harmonics constants
pub const SH_C0: f32 = 0.28209479177387814;
pub const SH_C1: f32 = 0.4886025119029199;

pub const SH_C2_0: f32 = 1.0925484305920792;
pub const SH_C2_1: f32 = -1.0925484305920792;
pub const SH_C2_2: f32 = 0.31539156525252005;
pub const SH_C2_3: f32 = -1.0925484305920792;
pub const SH_C2_4: f32 = 0.5462742152960396;

pub const SH_C3_0: f32 = -0.5900435899266435;
pub const SH_C3_1: f32 = 2.890611442640554;
pub const SH_C3_2: f32 = -0.4570457994644658;
pub const SH_C3_3: f32 = 0.3731763325901154;
pub const SH_C3_4: f32 = -0.4570457994644658;
pub const SH_C3_5: f32 = 1.445305721320277;
pub const SH_C3_6: f32 = -0.5900435899266435;

/// Evaluates Spherical Harmonics color [R, G, B] in [0.0, 1.0] along view direction `dir`.
///
/// Parameters:
/// - `sh_coeffs`: 48 floats per Gaussian (3 DC + 45 Rest, grouped by channel).
/// - `dir`: Normalized direction vector from camera to Gaussian center (or Gaussian to camera).
/// - `degree`: Active degree in [0, 3].
pub fn evaluate_sh(sh_coeffs: &[f32], dir: Vec3, degree: u32) -> [f32; 3] {
    if sh_coeffs.is_empty() {
        return [0.5, 0.5, 0.5];
    }

    // Degree 0 (DC)
    let mut r = SH_C0 * sh_coeffs[0];
    let mut g = SH_C0 * sh_coeffs[1];
    let mut b = SH_C0 * sh_coeffs[2];

    if degree == 0 || sh_coeffs.len() < 48 {
        return [
            (r + 0.5).clamp(0.0, 1.0),
            (g + 0.5).clamp(0.0, 1.0),
            (b + 0.5).clamp(0.0, 1.0),
        ];
    }

    let x = dir.x;
    let y = dir.y;
    let z = dir.z;

    // Degree 1 (3 basis functions)
    let y1_m1 = -SH_C1 * y;
    let y1_0 = SH_C1 * z;
    let y1_p1 = -SH_C1 * x;

    // R channel rest starts at index 3
    r += y1_m1 * sh_coeffs[3] + y1_0 * sh_coeffs[4] + y1_p1 * sh_coeffs[5];
    // G channel rest starts at index 3 + 15 = 18
    g += y1_m1 * sh_coeffs[18] + y1_0 * sh_coeffs[19] + y1_p1 * sh_coeffs[20];
    // B channel rest starts at index 3 + 30 = 33
    b += y1_m1 * sh_coeffs[33] + y1_0 * sh_coeffs[34] + y1_p1 * sh_coeffs[35];

    if degree >= 2 {
        let xx = x * x;
        let yy = y * y;
        let zz = z * z;
        let xy = x * y;
        let yz = y * z;
        let xz = x * z;

        // Degree 2 (5 basis functions)
        let y2_m2 = SH_C2_0 * xy;
        let y2_m1 = SH_C2_1 * yz;
        let y2_0 = SH_C2_2 * (2.0 * zz - xx - yy);
        let y2_p1 = SH_C2_3 * xz;
        let y2_p2 = SH_C2_4 * (xx - yy);

        r += y2_m2 * sh_coeffs[6]
            + y2_m1 * sh_coeffs[7]
            + y2_0 * sh_coeffs[8]
            + y2_p1 * sh_coeffs[9]
            + y2_p2 * sh_coeffs[10];

        g += y2_m2 * sh_coeffs[21]
            + y2_m1 * sh_coeffs[22]
            + y2_0 * sh_coeffs[23]
            + y2_p1 * sh_coeffs[24]
            + y2_p2 * sh_coeffs[25];

        b += y2_m2 * sh_coeffs[36]
            + y2_m1 * sh_coeffs[37]
            + y2_0 * sh_coeffs[38]
            + y2_p1 * sh_coeffs[39]
            + y2_p2 * sh_coeffs[40];

        if degree >= 3 {
            // Degree 3 (7 basis functions)
            let y3_m3 = SH_C3_0 * y * (3.0 * xx - yy);
            let y3_m2 = SH_C3_1 * xy * z;
            let y3_m1 = SH_C3_2 * y * (4.0 * zz - xx - yy);
            let y3_0 = SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
            let y3_p1 = SH_C3_4 * x * (4.0 * zz - xx - yy);
            let y3_p2 = SH_C3_5 * z * (xx - yy);
            let y3_p3 = SH_C3_6 * x * (xx - 3.0 * yy);

            r += y3_m3 * sh_coeffs[11]
                + y3_m2 * sh_coeffs[12]
                + y3_m1 * sh_coeffs[13]
                + y3_0 * sh_coeffs[14]
                + y3_p1 * sh_coeffs[15]
                + y3_p2 * sh_coeffs[16]
                + y3_p3 * sh_coeffs[17];

            g += y3_m3 * sh_coeffs[26]
                + y3_m2 * sh_coeffs[27]
                + y3_m1 * sh_coeffs[28]
                + y3_0 * sh_coeffs[29]
                + y3_p1 * sh_coeffs[30]
                + y3_p2 * sh_coeffs[31]
                + y3_p3 * sh_coeffs[32];

            b += y3_m3 * sh_coeffs[41]
                + y3_m2 * sh_coeffs[42]
                + y3_m1 * sh_coeffs[43]
                + y3_0 * sh_coeffs[44]
                + y3_p1 * sh_coeffs[45]
                + y3_p2 * sh_coeffs[46]
                + y3_p3 * sh_coeffs[47];
        }
    }

    [
        (r + 0.5).clamp(0.0, 1.0),
        (g + 0.5).clamp(0.0, 1.0),
        (b + 0.5).clamp(0.0, 1.0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sh_degree_zero_eval() {
        let mut sh = [0.0f32; 48];
        // Zero SH gives mid-gray (0.5, 0.5, 0.5)
        let color_mid = evaluate_sh(&sh, Vec3::Z, 0);
        assert_eq!(color_mid, [0.5, 0.5, 0.5]);

        // Positive DC
        sh[0] = 1.0;
        let color_pos = evaluate_sh(&sh, Vec3::Z, 0);
        assert!((color_pos[0] - (0.5 + SH_C0)).abs() < 1e-6);
        assert_eq!(color_pos[1], 0.5);
        assert_eq!(color_pos[2], 0.5);
    }

    #[test]
    fn test_sh_clamping() {
        let mut sh = [0.0f32; 48];
        sh[0] = 100.0; // Huge positive
        sh[1] = -100.0; // Huge negative

        let color = evaluate_sh(&sh, Vec3::Z, 0);
        assert_eq!(color[0], 1.0);
        assert_eq!(color[1], 0.0);
    }

    #[test]
    fn test_sh_degree_one_direction() {
        let mut sh = [0.0f32; 48];
        // Degree 1 z component is index 4 for Red
        sh[4] = 1.0;
        let color_z = evaluate_sh(&sh, Vec3::Z, 1);
        let expected_r = (0.5 + SH_C1).clamp(0.0, 1.0);
        assert!((color_z[0] - expected_r).abs() < 1e-6);

        // Opposite direction -Z
        let color_neg_z = evaluate_sh(&sh, -Vec3::Z, 1);
        let expected_neg_r = (0.5 - SH_C1).clamp(0.0, 1.0);
        assert!((color_neg_z[0] - expected_neg_r).abs() < 1e-6);
    }
}
