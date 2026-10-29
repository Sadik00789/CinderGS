//! Exact real spherical-harmonics rotation for degrees 1-3.
//!
//! A Gaussian's SH coefficients are authored in the splat's local frame. When a
//! volumetric cage rotates a splat, its view-dependent radiance must rotate with
//! it. Instead of the discarded "inverse-ray" hack (which evaluated world-space
//! SH along a rotated direction and therefore produced incorrect results for
//! anything but degree 1 without re-authoring coefficients), we rotate the
//! coefficients themselves.
//!
//! # Method
//!
//! Both the real SH basis and its rotated counterpart are homogeneous
//! polynomials of degree `l` in the direction components `(x, y, z)`. Let
//! `v_l(d)` be the vector of degree-`l` monomials and `B_l` the constant matrix
//! with `Y_l(d) = B_l v_l(d)`. A rotation acts linearly on monomials,
//! `v_l(Rᵀd) = T_l(Rᵀ) v_l(d)`.
//!
//! Requiring `c' ᵀY_l(d) = cᵀY_l(Rᵀd)` for all directions yields
//!
//! ```text
//! c' = (B_l T_l(Rᵀ) B_l⁺)ᵀ c
//! ```
//!
//! where `B_l⁺` is the right pseudo-inverse of `B_l`. Since `B_l` is constant it
//! is computed once and cached; only `T_l` depends on the rotation, and it is a
//! small monomial expansion (5x5 for degree 2, 7x7 for degree 3).

use glam::Mat3;
use std::sync::OnceLock;

/// Number of SH coefficients per degree (degree 0..=3).
pub const SH_COEFFS_PER_DEGREE: [usize; 4] = [1, 3, 5, 7];

/// Number of "rest" (non-DC) coefficients per colour channel.
pub const SH_REST_PER_CHANNEL: usize = 15;

const SH_C1: f64 = 0.488_602_511_902_919_9;
const SH_C2_0: f64 = 1.092_548_430_592_079_2;
const SH_C2_2: f64 = 0.315_391_565_252_520_05;
const SH_C2_4: f64 = 0.546_274_215_296_039_6;
const SH_C3_0: f64 = -0.590_043_589_926_643_5;
const SH_C3_1: f64 = 2.890_611_442_640_554;
const SH_C3_2: f64 = -0.457_045_799_464_465_8;
const SH_C3_3: f64 = 0.373_176_332_590_115_4;
const SH_C3_5: f64 = 1.445_305_721_320_277;

/// Monomial exponents `[x, y, z]` spanning homogeneous polynomials of `degree`.
fn exps_of(degree: u32) -> Vec<[u32; 3]> {
    let mut out = Vec::with_capacity(((degree + 1) * (degree + 2) / 2) as usize);
    for x in (0..=degree).rev() {
        for y in (0..=(degree - x)).rev() {
            out.push([x, y, degree - x - y]);
        }
    }
    out
}

fn index_of(exps: &[[u32; 3]], e: [u32; 3]) -> usize {
    exps.iter()
        .position(|candidate| *candidate == e)
        .expect("monomial always belongs to its degree basis")
}

fn poly_mul(a: &[f64], deg_a: u32, b: &[f64], deg_b: u32) -> Vec<f64> {
    let ea = exps_of(deg_a);
    let eb = exps_of(deg_b);
    let ed = exps_of(deg_a + deg_b);
    let mut out = vec![0.0f64; ed.len()];
    for (ia, &pa) in ea.iter().enumerate() {
        if a[ia] == 0.0 {
            continue;
        }
        for (ib, &pb) in eb.iter().enumerate() {
            if b[ib] == 0.0 {
                continue;
            }
            let sum = [pa[0] + pb[0], pa[1] + pb[1], pa[2] + pb[2]];
            out[index_of(&ed, sum)] += a[ia] * b[ib];
        }
    }
    out
}

fn poly_add_scaled(dst: &mut [f64], src: &[f64], scale: f64) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += s * scale;
    }
}

fn lin(x: f64, y: f64, z: f64) -> Vec<f64> {
    vec![x, y, z]
}

fn scaled(v: &[f64], s: f64) -> Vec<f64> {
    v.iter().map(|c| c * s).collect()
}

/// `degree`-1 polynomial value `2z² - x² - y²`.
fn poly_2z2_minus_xx_yy(x: &[f64], y: &[f64], z: &[f64]) -> Vec<f64> {
    let zz = poly_mul(z, 1, z, 1);
    let xx = poly_mul(x, 1, x, 1);
    let yy = poly_mul(y, 1, y, 1);
    let mut out = scaled(&zz, 2.0);
    poly_add_scaled(&mut out, &xx, -1.0);
    poly_add_scaled(&mut out, &yy, -1.0);
    out
}

/// `degree`-2 polynomial value `4z² - x² - y²`.
fn poly_4z2_minus_xx_yy(x: &[f64], y: &[f64], z: &[f64]) -> Vec<f64> {
    let zz = poly_mul(z, 1, z, 1);
    let xx = poly_mul(x, 1, x, 1);
    let yy = poly_mul(y, 1, y, 1);
    let mut out = scaled(&zz, 4.0);
    poly_add_scaled(&mut out, &xx, -1.0);
    poly_add_scaled(&mut out, &yy, -1.0);
    out
}

fn poly_xx_minus_yy(x: &[f64], y: &[f64]) -> Vec<f64> {
    let xx = poly_mul(x, 1, x, 1);
    let yy = poly_mul(y, 1, y, 1);
    let mut out = xx;
    poly_add_scaled(&mut out, &yy, -1.0);
    out
}

/// Rows of the real SH basis, expressed in the monomial basis of `degree`.
///
/// The ordering matches [`crate::render::rasterizer::sh::evaluate_sh`]:
/// `(m = -l, -l+1, ..., +l)`.
fn basis_rows(degree: u32) -> Vec<Vec<f64>> {
    let x = lin(1.0, 0.0, 0.0);
    let y = lin(0.0, 1.0, 0.0);
    let z = lin(0.0, 0.0, 1.0);

    match degree {
        0 => vec![vec![1.0]],
        1 => vec![scaled(&y, -SH_C1), scaled(&z, SH_C1), scaled(&x, -SH_C1)],
        2 => {
            let xy = poly_mul(&x, 1, &y, 1);
            let yz = poly_mul(&y, 1, &z, 1);
            let xz = poly_mul(&x, 1, &z, 1);
            let y2_0 = poly_2z2_minus_xx_yy(&x, &y, &z);
            let y2_p2 = poly_xx_minus_yy(&x, &y);
            vec![
                scaled(&xy, SH_C2_0),
                scaled(&yz, -SH_C2_0),
                scaled(&y2_0, SH_C2_2),
                scaled(&xz, -SH_C2_0),
                scaled(&y2_p2, SH_C2_4),
            ]
        }
        3 => {
            let xx = poly_mul(&x, 1, &x, 1);
            let yy = poly_mul(&y, 1, &y, 1);
            let xy = poly_mul(&x, 1, &y, 1);

            // y * (3x² - y²)
            let mut p_m3 = scaled(&xx, 3.0);
            poly_add_scaled(&mut p_m3, &yy, -1.0);
            let p_m3 = poly_mul(&y, 1, &p_m3, 2);

            // x * y * z
            let p_m2 = poly_mul(&xy, 2, &z, 1);

            // y * (4z² - x² - y²)
            let q = poly_4z2_minus_xx_yy(&x, &y, &z);
            let p_m1 = poly_mul(&y, 1, &q, 2);

            // z * (2z² - 3x² - 3y²)
            let zz = poly_mul(&z, 1, &z, 1);
            let mut p_0 = scaled(&zz, 2.0);
            poly_add_scaled(&mut p_0, &xx, -3.0);
            poly_add_scaled(&mut p_0, &yy, -3.0);
            let p_0 = poly_mul(&z, 1, &p_0, 2);

            // x * (4z² - x² - y²)
            let p_p1 = poly_mul(&x, 1, &q, 2);

            // z * (x² - y²)
            let d = poly_xx_minus_yy(&x, &y);
            let p_p2 = poly_mul(&z, 1, &d, 2);

            // x * (x² - 3y²)
            let mut p_p3 = xx;
            poly_add_scaled(&mut p_p3, &yy, -3.0);
            let p_p3 = poly_mul(&x, 1, &p_p3, 2);

            vec![
                scaled(&p_m3, SH_C3_0),
                scaled(&p_m2, SH_C3_1),
                scaled(&p_m1, SH_C3_2),
                scaled(&p_0, SH_C3_3),
                scaled(&p_p1, SH_C3_2),
                scaled(&p_p2, SH_C3_5),
                scaled(&p_p3, SH_C3_0),
            ]
        }
        _ => unreachable!("SH rotation supports degrees 0..=3"),
    }
}

fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = a.len();
    let cols = b[0].len();
    let inner = b.len();
    let mut out = vec![vec![0.0f64; cols]; rows];
    for (i, row) in out.iter_mut().enumerate() {
        for k in 0..inner {
            let aik = a[i][k];
            if aik == 0.0 {
                continue;
            }
            for (j, slot) in row.iter_mut().enumerate() {
                *slot += aik * b[k][j];
            }
        }
    }
    out
}

fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = a.len();
    let cols = a[0].len();
    let mut out = vec![vec![0.0f64; rows]; cols];
    for (i, row) in a.iter().enumerate() {
        for (j, value) in row.iter().enumerate() {
            out[j][i] = *value;
        }
    }
    out
}

/// Gauss-Jordan inverse with partial pivoting. Returns `None` if singular.
fn invert(m: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = m.len();
    let mut aug = vec![vec![0.0f64; 2 * n]; n];
    for (i, row) in m.iter().enumerate() {
        aug[i][..n].copy_from_slice(row);
        aug[i][n + i] = 1.0;
    }

    for col in 0..n {
        let pivot = (col..n).max_by(|&a, &b| {
            aug[a][col]
                .abs()
                .partial_cmp(&aug[b][col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        if aug[pivot][col].abs() < 1e-12 {
            return None;
        }
        aug.swap(col, pivot);

        let inv_pivot = 1.0 / aug[col][col];
        for value in aug[col].iter_mut() {
            *value *= inv_pivot;
        }

        let pivot_row = aug[col].clone();
        for (row, values) in aug.iter_mut().enumerate() {
            if row == col {
                continue;
            }
            let factor = values[col];
            if factor == 0.0 {
                continue;
            }
            for (value, pivot_value) in values.iter_mut().zip(pivot_row.iter()) {
                *value -= factor * pivot_value;
            }
        }
    }

    Some(aug.into_iter().map(|row| row[n..].to_vec()).collect())
}

struct Basis {
    b: Vec<Vec<f64>>,
    pinv: Vec<Vec<f64>>,
}

fn basis_for(degree: u32) -> &'static Basis {
    static B1: OnceLock<Basis> = OnceLock::new();
    static B2: OnceLock<Basis> = OnceLock::new();
    static B3: OnceLock<Basis> = OnceLock::new();

    let slot = match degree {
        1 => &B1,
        2 => &B2,
        3 => &B3,
        _ => unreachable!("SH basis only requested for degrees 1..=3"),
    };

    slot.get_or_init(|| {
        let b = basis_rows(degree);
        let bt = transpose(&b);
        let bbt = mat_mul(&b, &bt);
        let inv = invert(&bbt).expect("SH basis Gram matrix must be invertible");
        let pinv = mat_mul(&bt, &inv);
        Basis { b, pinv }
    })
}

fn poly_mul_linear(a: &[f64], deg: u32, l: [f64; 3]) -> Vec<f64> {
    let ea = exps_of(deg);
    let ed = exps_of(deg + 1);
    let mut out = vec![0.0f64; ed.len()];
    for (ia, &e) in ea.iter().enumerate() {
        let coeff = a[ia];
        if coeff == 0.0 {
            continue;
        }
        for axis in 0..3 {
            if l[axis] == 0.0 {
                continue;
            }
            let mut ne = e;
            ne[axis] += 1;
            out[index_of(&ed, ne)] += coeff * l[axis];
        }
    }
    out
}

/// Monomial transform `T` with `v_l(Rᵀd) = T v_l(d)`.
fn monomial_transform(rotation: Mat3, degree: u32) -> Vec<Vec<f64>> {
    let exps = exps_of(degree);
    let size = exps.len();

    // Rows of Rᵀ are the columns of R.
    let linear_forms = [
        [
            rotation.x_axis.x as f64,
            rotation.x_axis.y as f64,
            rotation.x_axis.z as f64,
        ],
        [
            rotation.y_axis.x as f64,
            rotation.y_axis.y as f64,
            rotation.y_axis.z as f64,
        ],
        [
            rotation.z_axis.x as f64,
            rotation.z_axis.y as f64,
            rotation.z_axis.z as f64,
        ],
    ];

    let mut t = vec![vec![0.0f64; size]; size];
    for (row, exp) in exps.iter().enumerate() {
        let mut poly = vec![1.0f64];
        let mut deg = 0u32;
        for (axis, form) in linear_forms.iter().enumerate() {
            for _ in 0..exp[axis] {
                poly = poly_mul_linear(&poly, deg, *form);
                deg += 1;
            }
        }
        t[row] = poly;
    }
    t
}

/// Builds the `(2l+1) x (2l+1)` row-major SH rotation matrix for `degree`.
///
/// Applying this matrix to a coefficient block `c` produces `c'` such that
/// `evaluate_sh(c', d, l) == evaluate_sh(c, Rᵀd, l)`.
pub fn sh_rotation_matrix(rotation: Mat3, degree: u32) -> Vec<f32> {
    assert!(
        (1..=3).contains(&degree),
        "SH rotation matrix requested for unsupported degree {degree}"
    );
    let basis = basis_for(degree);
    let t = monomial_transform(rotation, degree);
    let c = mat_mul(&basis.b, &t);
    let d = mat_mul(&c, &basis.pinv);
    let m = transpose(&d);
    m.into_iter()
        .flat_map(|row| row.into_iter().map(|v| v as f32))
        .collect()
}

fn apply_block(coeffs: &mut [f32; 48], offset: usize, size: usize, matrix: &[f32]) {
    let mut rotated = [0.0f32; 7];
    for (row, slot) in rotated.iter_mut().enumerate().take(size) {
        let mut acc = 0.0f32;
        for col in 0..size {
            acc += matrix[row * size + col] * coeffs[offset + col];
        }
        *slot = acc;
    }
    coeffs[offset..offset + size].copy_from_slice(&rotated[..size]);
}

/// Precomputed SH rotation for one frame of one tetrahedron.
///
/// The matrices depend only on the rotation, so they are built once per
/// tetrahedron and reused for every Gaussian bound to it. Building them per
/// Gaussian would dominate the deformation stage.
#[derive(Clone, Debug)]
pub struct ShRotator {
    degree: u32,
    m1: Vec<f32>,
    m2: Option<Vec<f32>>,
    m3: Option<Vec<f32>>,
}

impl ShRotator {
    /// Builds the rotation matrices for `degree` (0..=3).
    pub fn new(rotation: Mat3, degree: u32) -> Self {
        let degree = degree.min(3);
        if degree == 0 {
            return Self {
                degree,
                m1: Vec::new(),
                m2: None,
                m3: None,
            };
        }

        Self {
            degree,
            m1: sh_rotation_matrix(rotation, 1),
            m2: (degree >= 2).then(|| sh_rotation_matrix(rotation, 2)),
            m3: (degree >= 3).then(|| sh_rotation_matrix(rotation, 3)),
        }
    }

    /// Rotates the coefficients in place.
    pub fn apply(&self, coeffs: &mut [f32; 48]) {
        if self.degree == 0 {
            return;
        }

        for channel in 0..3 {
            let rest = 3 + channel * SH_REST_PER_CHANNEL;
            apply_block(coeffs, rest, 3, &self.m1);
            if let Some(matrix) = &self.m2 {
                apply_block(coeffs, rest + 3, 5, matrix);
            }
            if let Some(matrix) = &self.m3 {
                apply_block(coeffs, rest + 8, 7, matrix);
            }
        }
    }
}

/// Rotates the SH coefficients of a single Gaussian by `rotation`.
///
/// `degree` is the active SH degree (0..=3). Degree 0 is a scalar and is left
/// untouched. The per-channel layout matches
/// [`crate::scene::soa::GaussianSceneSoa::sh_coeffs`]: three DC values followed by
/// three contiguous 15-coefficient rest blocks.
pub fn rotate_sh_coefficients(coeffs: &mut [f32; 48], rotation: Mat3, degree: u32) {
    ShRotator::new(rotation, degree).apply(coeffs);
}

/// Convenience wrapper that operates on a flat `N * 48` coefficient buffer.
pub fn rotate_sh_buffer(coeffs: &mut [f32], index: usize, rotation: Mat3, degree: u32) {
    let base = index * 48;
    if base + 48 > coeffs.len() {
        return;
    }
    let mut block = [0.0f32; 48];
    block.copy_from_slice(&coeffs[base..base + 48]);
    rotate_sh_coefficients(&mut block, rotation, degree);
    coeffs[base..base + 48].copy_from_slice(&block);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::rasterizer::sh::evaluate_sh;
    use glam::Vec3;

    fn rotation_from_axis_angle(axis: Vec3, angle: f32) -> Mat3 {
        Mat3::from_axis_angle(axis.normalize(), angle)
    }

    #[test]
    fn test_rotation_matrix_is_orthogonal() {
        let r = rotation_from_axis_angle(Vec3::new(0.3, 1.0, -0.2), 0.7);
        for degree in 1..=3 {
            let n = SH_COEFFS_PER_DEGREE[degree as usize];
            let m = sh_rotation_matrix(r, degree);
            assert_eq!(m.len(), n * n);

            // M Mᵀ must be the identity.
            for i in 0..n {
                for j in 0..n {
                    let mut acc = 0.0f32;
                    for k in 0..n {
                        acc += m[i * n + k] * m[j * n + k];
                    }
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (acc - expected).abs() < 1e-3,
                        "degree {degree}: M Mᵀ[{i},{j}] = {acc}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_rotated_sh_matches_rotated_direction() {
        let rotations = [
            Mat3::IDENTITY,
            Mat3::from_rotation_y(0.4),
            Mat3::from_rotation_x(-0.9) * Mat3::from_rotation_z(0.25),
            rotation_from_axis_angle(Vec3::new(-1.0, 0.4, 0.8), 2.1),
        ];
        let dirs = [
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.3, -0.5, 0.81).normalize(),
            Vec3::new(-0.7, 0.2, -0.68).normalize(),
        ];

        for r in rotations {
            for degree in 1..=3u32 {
                let mut sh = [0.0f32; 48];
                // Deterministic pseudo-random coefficients in every rest slot.
                for c in 0..3 {
                    let base = 3 + c * 15;
                    for k in 0..15 {
                        sh[base + k] = ((c as f32 + 1.0) * (k as f32 + 1.0) * 0.13).sin() * 0.8;
                    }
                    sh[c] = 0.2 * (c as f32 + 1.0);
                }

                let mut rotated = sh;
                rotate_sh_coefficients(&mut rotated, r, degree);

                for d in dirs {
                    let expected = evaluate_sh(&sh, r.transpose() * d, degree);
                    let actual = evaluate_sh(&rotated, d, degree);
                    for channel in 0..3 {
                        assert!(
                            (actual[channel] - expected[channel]).abs() < 2e-3,
                            "degree {degree}, dir {d:?}, channel {channel}: {} vs {}",
                            actual[channel],
                            expected[channel]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_identity_rotation_is_a_no_op() {
        let mut sh = [0.0f32; 48];
        for (i, value) in sh.iter_mut().enumerate() {
            *value = (i as f32 * 0.37).sin();
        }
        let mut rotated = sh;
        rotate_sh_coefficients(&mut rotated, Mat3::IDENTITY, 3);
        for (a, b) in rotated.iter().zip(sh.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn test_degree_zero_is_untouched() {
        let mut sh = [0.0f32; 48];
        sh[0] = 1.5;
        sh[45] = 0.25;
        let before = sh;
        rotate_sh_coefficients(&mut sh, Mat3::from_rotation_z(1.0), 0);
        assert_eq!(sh, before);
    }
}
