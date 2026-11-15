use crate::math::{activate_opacity, activate_rotation, activate_scale, compute_covariance_3d};
use crate::parallel;
use crate::scene::raw_vertex::RawPlyVertex;
use thiserror::Error;

/// Error returned when `GaussianSceneSoa` buffer lengths are inconsistent with `count`.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SoaValidationError {
    #[error("Positions buffer length {actual} does not match count * 3 ({expected})")]
    InvalidPositionsLength { expected: usize, actual: usize },

    #[error("Covariances 3D buffer length {actual} does not match count * 6 ({expected})")]
    InvalidCovariancesLength { expected: usize, actual: usize },

    #[error("Opacities buffer length {actual} does not match count ({expected})")]
    InvalidOpacitiesLength { expected: usize, actual: usize },

    #[error("SH coefficients buffer length {actual} does not match count * 48 ({expected})")]
    InvalidShLength { expected: usize, actual: usize },
}

/// Structure-of-Arrays (SoA) layout optimized for GPU compute kernels and buffer uploads.
///
/// Holds activated 3D Gaussian Splatting scene data:
/// - Positions: `[x, y, z]` flattened (`N * 3` floats)
/// - Covariances 3D: `[xx, xy, xz, yy, yz, zz]` flattened (`N * 6` floats)
/// - Opacities: activated sigmoidal alpha (`N` floats)
/// - SH Coefficients: degree 0 (3 floats) + degree 1-3 (45 floats) (`N * 48` floats)
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianSceneSoa {
    /// World-space positions: `[N * 3]` floats (x, y, z).
    pub positions: Vec<f32>,

    /// 3D Covariances: `[N * 6]` floats (xx, xy, xz, yy, yz, zz upper triangular).
    pub covariances_3d: Vec<f32>,

    /// Activated opacities in range (0, 1): `[N]` floats.
    pub opacities: Vec<f32>,

    /// Spherical Harmonics coefficients: `[N * 48]` floats (f_dc 0..2 followed by f_rest 0..44).
    pub sh_coeffs: Vec<f32>,

    /// Total number of Gaussians in the scene.
    pub count: usize,
}

impl GaussianSceneSoa {
    /// Chunk size used for parallel multi-threaded ingestion.
    pub const PARALLEL_CHUNK_SIZE: usize = 2048;

    /// Creates an empty `GaussianSceneSoa`.
    #[inline]
    pub fn new() -> Self {
        Self {
            positions: Vec::new(),
            covariances_3d: Vec::new(),
            opacities: Vec::new(),
            sh_coeffs: Vec::new(),
            count: 0,
        }
    }

    /// Pre-allocates memory for `capacity` Gaussians.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            positions: Vec::with_capacity(capacity * 3),
            covariances_3d: Vec::with_capacity(capacity * 6),
            opacities: Vec::with_capacity(capacity),
            sh_coeffs: Vec::with_capacity(capacity * 48),
            count: 0,
        }
    }

    /// Number of Gaussians stored in the scene.
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns true if the scene contains no Gaussians.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Clears all buffers while retaining allocated memory.
    #[inline]
    pub fn clear(&mut self) {
        self.positions.clear();
        self.covariances_3d.clear();
        self.opacities.clear();
        self.sh_coeffs.clear();
        self.count = 0;
    }

    /// Appends an already-activated Gaussian to the SoA structure.
    pub fn push(
        &mut self,
        position: [f32; 3],
        covariance_3d: [f32; 6],
        opacity: f32,
        sh_coeffs: &[f32; 48],
    ) {
        self.positions.extend_from_slice(&position);
        self.covariances_3d.extend_from_slice(&covariance_3d);
        self.opacities.push(opacity);
        self.sh_coeffs.extend_from_slice(sh_coeffs);
        self.count += 1;
    }

    /// Activates a raw PLY vertex and appends it to the SoA layout.
    pub fn push_raw(&mut self, raw: &RawPlyVertex) {
        let scale = activate_scale(raw.scale);
        let rot = activate_rotation(raw.rot);
        let cov = compute_covariance_3d(scale, rot);
        let opacity = activate_opacity(raw.opacity);

        self.positions.extend_from_slice(&raw.position);
        self.covariances_3d.extend_from_slice(&cov);
        self.opacities.push(opacity);
        self.sh_coeffs.extend_from_slice(&raw.f_dc);
        self.sh_coeffs.extend_from_slice(&raw.f_rest);
        self.count += 1;
    }

    /// Ingests a slice of `RawPlyVertex` sequentially into a new `GaussianSceneSoa`.
    pub fn from_raw_vertices(raw: &[RawPlyVertex]) -> Self {
        let mut soa = Self::with_capacity(raw.len());
        for v in raw {
            soa.push_raw(v);
        }
        soa
    }

    /// Ingests a slice of `RawPlyVertex` in parallel using Rayon work-stealing threads.
    pub fn from_raw_vertices_par(raw: &[RawPlyVertex]) -> Self {
        let count = raw.len();
        if count == 0 {
            return Self::new();
        }

        let mut positions = vec![0.0f32; count * 3];
        let mut covariances_3d = vec![0.0f32; count * 6];
        let mut opacities = vec![0.0f32; count];
        let mut sh_coeffs = vec![0.0f32; count * 48];

        let chunk_size = Self::PARALLEL_CHUNK_SIZE;

        parallel::zip_fill_4(
            &mut positions,
            &mut covariances_3d,
            &mut opacities,
            &mut sh_coeffs,
            [chunk_size * 3, chunk_size * 6, chunk_size, chunk_size * 48],
            |base, pos_chunk, cov_chunk, op_chunk, sh_chunk| {
                let raw_chunk = &raw[base..(base + op_chunk.len()).min(count)];
                for (i, v) in raw_chunk.iter().enumerate() {
                    // 1. Position
                    let p_idx = i * 3;
                    pos_chunk[p_idx..p_idx + 3].copy_from_slice(&v.position);

                    // 2. 3D Covariance
                    let scale = activate_scale(v.scale);
                    let rot = activate_rotation(v.rot);
                    let cov = compute_covariance_3d(scale, rot);
                    let c_idx = i * 6;
                    cov_chunk[c_idx..c_idx + 6].copy_from_slice(&cov);

                    // 3. Opacity (Sigmoid)
                    op_chunk[i] = activate_opacity(v.opacity);

                    // 4. SH Coefficients: DC (3 floats) + Rest (45 floats) = 48 floats
                    let sh_idx = i * 48;
                    sh_chunk[sh_idx..sh_idx + 3].copy_from_slice(&v.f_dc);
                    sh_chunk[sh_idx + 3..sh_idx + 48].copy_from_slice(&v.f_rest);
                }
            },
        );

        Self {
            positions,
            covariances_3d,
            opacities,
            sh_coeffs,
            count,
        }
    }

    /// Validates that all buffer lengths match the declared Gaussian count.
    pub fn validate(&self) -> Result<(), SoaValidationError> {
        if self.positions.len() != self.count * 3 {
            return Err(SoaValidationError::InvalidPositionsLength {
                expected: self.count * 3,
                actual: self.positions.len(),
            });
        }
        if self.covariances_3d.len() != self.count * 6 {
            return Err(SoaValidationError::InvalidCovariancesLength {
                expected: self.count * 6,
                actual: self.covariances_3d.len(),
            });
        }
        if self.opacities.len() != self.count {
            return Err(SoaValidationError::InvalidOpacitiesLength {
                expected: self.count,
                actual: self.opacities.len(),
            });
        }
        if self.sh_coeffs.len() != self.count * 48 {
            return Err(SoaValidationError::InvalidShLength {
                expected: self.count * 48,
                actual: self.sh_coeffs.len(),
            });
        }
        Ok(())
    }

    /// Accessor for position of the $i$-th Gaussian.
    #[inline]
    pub fn position(&self, index: usize) -> [f32; 3] {
        let base = index * 3;
        [
            self.positions[base],
            self.positions[base + 1],
            self.positions[base + 2],
        ]
    }

    /// Accessor for 3D covariance of the $i$-th Gaussian `[xx, xy, xz, yy, yz, zz]`.
    #[inline]
    pub fn covariance_3d(&self, index: usize) -> [f32; 6] {
        let base = index * 6;
        [
            self.covariances_3d[base],
            self.covariances_3d[base + 1],
            self.covariances_3d[base + 2],
            self.covariances_3d[base + 3],
            self.covariances_3d[base + 4],
            self.covariances_3d[base + 5],
        ]
    }

    /// Accessor for opacity of the $i$-th Gaussian.
    #[inline]
    pub fn opacity(&self, index: usize) -> f32 {
        self.opacities[index]
    }

    /// Accessor for SH coefficients of the $i$-th Gaussian (slice of 48 floats).
    #[inline]
    pub fn sh_coeffs_at(&self, index: usize) -> &[f32] {
        let base = index * 48;
        &self.sh_coeffs[base..base + 48]
    }

    // --- GPU Buffer Upload Helpers (Zero-Copy Byte Slices) ---

    /// Casts positions slice to raw bytes for GPU buffer upload.
    #[inline]
    pub fn positions_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.positions)
    }

    /// Casts 3D covariances slice to raw bytes for GPU buffer upload.
    #[inline]
    pub fn covariances_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.covariances_3d)
    }

    /// Casts opacities slice to raw bytes for GPU buffer upload.
    #[inline]
    pub fn opacities_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.opacities)
    }

    /// Casts SH coefficients slice to raw bytes for GPU buffer upload.
    #[inline]
    pub fn sh_coeffs_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.sh_coeffs)
    }

    /// Total memory size in bytes of all SoA arrays combined.
    #[inline]
    pub fn total_bytes(&self) -> usize {
        self.positions_bytes().len()
            + self.covariances_bytes().len()
            + self.opacities_bytes().len()
            + self.sh_coeffs_bytes().len()
    }

    /// Quantizes covariances and SH coefficients into compact `f16` + octahedral rotation storage.
    #[inline]
    pub fn quantize(&self) -> QuantizedGaussianSceneSoa {
        QuantizedGaussianSceneSoa::from_soa(self)
    }

    /// Quantizes the scene keeping SH coefficients up to `sh_degree` (`0..=3`).
    #[inline]
    pub fn quantize_with_degree(&self, sh_degree: u32) -> QuantizedGaussianSceneSoa {
        QuantizedGaussianSceneSoa::from_soa_with_degree(self, sh_degree)
    }

    /// Reconstructs uncompressed `RawPlyVertex` records (pre-activation log-scale and logit-opacity)
    /// suitable for serializing deformed scenes back to standard binary `.ply` files.
    pub fn to_raw_vertices(&self) -> Vec<RawPlyVertex> {
        let mut vertices = Vec::with_capacity(self.count);
        for i in 0..self.count {
            let mut v = RawPlyVertex::ZERO;
            v.position = self.position(i);
            v.opacity = crate::math::covariance::inverse_activate_opacity(self.opacity(i));

            let cov = self.covariance_3d(i);
            let (scale, rot) = crate::math::covariance::decompose_covariance_3d(&cov);
            let log_scale = crate::math::covariance::inverse_activate_scale(scale);
            v.scale = [log_scale.x, log_scale.y, log_scale.z];
            // Canonical PLY quaternion order is [w, x, y, z].
            v.rot = [rot.w, rot.x, rot.y, rot.z];

            let sh = self.sh_coeffs_at(i);
            v.f_dc = [sh[0], sh[1], sh[2]];
            v.f_rest.copy_from_slice(&sh[3..48]);
            vertices.push(v);
        }
        vertices
    }
}

/// Number of SH scalar bases per color channel for degree `0..=3`.
#[inline]
pub const fn sh_bases_for_degree(sh_degree: u32) -> usize {
    match sh_degree {
        0 => 1,
        1 => 4,
        2 => 9,
        _ => 16,
    }
}

/// Number of packed `u32` words (two `f16` scalars each) per splat for `sh_degree`.
#[inline]
pub const fn sh_packed_words_for_degree(sh_degree: u32) -> usize {
    (sh_bases_for_degree(sh_degree) * 3).div_ceil(2)
}

/// Packs two `f32` values into a single `u32` (`low = a as f16`, `high = b as f16`).
#[inline]
pub fn pack_f16_pair(a: f32, b: f32) -> u32 {
    let ha = half::f16::from_f32(a).to_bits() as u32;
    let hb = half::f16::from_f32(b).to_bits() as u32;
    ha | (hb << 16)
}

/// Unpacks two `f16` values from a packed `u32` (`[low, high]`).
#[inline]
pub fn unpack_f16_pair(packed: u32) -> [f32; 2] {
    let a = half::f16::from_bits((packed & 0xFFFF) as u16).to_f32();
    let b = half::f16::from_bits((packed >> 16) as u16).to_f32();
    [a, b]
}

#[inline]
fn sign_not_zero(v: f32) -> f32 {
    if v >= 0.0 { 1.0 } else { -1.0 }
}

/// Encodes a unit rotation quaternion into 32 bits via octahedral axis encoding
/// (11 bits X, 11 bits Y) plus a 10-bit rotation angle in `[0, pi]`.
pub fn encode_octahedral_rotation(rot: glam::Quat) -> u32 {
    let mut q = if rot.length_squared() > 1e-12 {
        rot.normalize()
    } else {
        glam::Quat::IDENTITY
    };
    if q.w < 0.0 {
        q = -q;
    }

    let (axis, angle) = q.to_axis_angle();
    let dir = if angle <= 1e-6 || axis.length_squared() <= 1e-12 {
        glam::Vec3::Z
    } else {
        axis.normalize()
    };

    let l1 = dir.x.abs() + dir.y.abs() + dir.z.abs();
    let mut ox = dir.x / l1.max(1e-8);
    let mut oy = dir.y / l1.max(1e-8);
    if dir.z < 0.0 {
        let folded_x = (1.0 - oy.abs()) * sign_not_zero(ox);
        let folded_y = (1.0 - ox.abs()) * sign_not_zero(oy);
        ox = folded_x;
        oy = folded_y;
    }

    let ix = (((ox * 0.5 + 0.5).clamp(0.0, 1.0)) * 2047.0).round() as u32;
    let iy = (((oy * 0.5 + 0.5).clamp(0.0, 1.0)) * 2047.0).round() as u32;
    let ia = (((angle / std::f32::consts::PI).clamp(0.0, 1.0)) * 1023.0).round() as u32;

    (ix & 0x7FF) | ((iy & 0x7FF) << 11) | ((ia & 0x3FF) << 22)
}

/// Decodes an octahedral-encoded 32-bit rotation back to a normalized [`glam::Quat`].
pub fn decode_octahedral_rotation(packed: u32) -> glam::Quat {
    let ox = ((packed & 0x7FF) as f32 / 2047.0) * 2.0 - 1.0;
    let oy = (((packed >> 11) & 0x7FF) as f32 / 2047.0) * 2.0 - 1.0;
    let oz = 1.0 - ox.abs() - oy.abs();

    let (vx, vy) = if oz < 0.0 {
        (
            (1.0 - oy.abs()) * sign_not_zero(ox),
            (1.0 - ox.abs()) * sign_not_zero(oy),
        )
    } else {
        (ox, oy)
    };

    let axis = glam::Vec3::new(vx, vy, oz)
        .try_normalize()
        .unwrap_or(glam::Vec3::Z);
    let angle = (((packed >> 22) & 0x3FF) as f32 / 1023.0) * std::f32::consts::PI;
    glam::Quat::from_axis_angle(axis, angle).normalize()
}

/// Memory-efficient GPU/WebGPU representation packing symmetric 3D covariances
/// (`3 x u32` `f16` pairs per splat), octahedral unit rotations (`u32`), and
/// degree-trimmed `f16` spherical harmonic coefficients (`2..=24` `u32` words/splat).
#[derive(Clone, Debug, PartialEq)]
pub struct QuantizedGaussianSceneSoa {
    /// World positions `[x, y, z, ...]` (full `f32` precision for spatial stability).
    pub positions: Vec<f32>,
    /// Symmetric 3D covariances `[xx, xy, xz, yy, yz, zz]` packed as 3 `u32` (`f16` pairs) per splat.
    pub covariances_f16_packed: Vec<u32>,
    /// Principal rotation decomposed from covariance, packed with octahedral encoding (`u32` per splat).
    pub octahedral_rotations: Vec<u32>,
    /// Sigmoid-activated opacities `[alpha_0, ...]`.
    pub opacities: Vec<f32>,
    /// `f16`-packed SH coefficients `[R_0..B_{B-1}]` (`sh_packed_words_for_degree(sh_degree)` `u32`s/splat).
    pub sh_coeffs_f16_packed: Vec<u32>,
    /// Retained spherical harmonic degree (`0..=3`).
    pub sh_degree: u32,
    /// Number of Gaussians stored.
    pub count: usize,
}

impl QuantizedGaussianSceneSoa {
    /// Quantizes all SH degrees (`sh_degree = 3`) into `f16` + octahedral rotation storage.
    #[inline]
    pub fn from_soa(scene: &GaussianSceneSoa) -> Self {
        Self::from_soa_with_degree(scene, 3)
    }

    /// Quantizes `scene`, retaining only SH coefficients up to `sh_degree` (`0..=3`).
    pub fn from_soa_with_degree(scene: &GaussianSceneSoa, sh_degree: u32) -> Self {
        let deg = sh_degree.min(3);
        let bases = sh_bases_for_degree(deg);
        let sh_words = sh_packed_words_for_degree(deg);
        let count = scene.count;

        let mut covariances_f16_packed = Vec::with_capacity(count * 3);
        let mut octahedral_rotations = Vec::with_capacity(count);
        let mut sh_coeffs_f16_packed = Vec::with_capacity(count * sh_words);

        let mut temp_sh = vec![0.0f32; sh_words * 2];

        for i in 0..count {
            let cov = scene.covariance_3d(i);
            covariances_f16_packed.push(pack_f16_pair(cov[0], cov[1]));
            covariances_f16_packed.push(pack_f16_pair(cov[2], cov[3]));
            covariances_f16_packed.push(pack_f16_pair(cov[4], cov[5]));

            let (_, rot) = crate::math::covariance::decompose_covariance_3d(&cov);
            octahedral_rotations.push(encode_octahedral_rotation(rot));

            let sh = scene.sh_coeffs_at(i);
            let rest_per_ch = bases - 1;
            temp_sh.fill(0.0);
            temp_sh[0..3].copy_from_slice(&sh[0..3]);
            if rest_per_ch > 0 {
                temp_sh[3..3 + rest_per_ch].copy_from_slice(&sh[3..3 + rest_per_ch]);
                temp_sh[3 + rest_per_ch..3 + rest_per_ch * 2]
                    .copy_from_slice(&sh[18..18 + rest_per_ch]);
                temp_sh[3 + rest_per_ch * 2..3 + rest_per_ch * 3]
                    .copy_from_slice(&sh[33..33 + rest_per_ch]);
            }
            for pair in temp_sh.chunks_exact(2) {
                sh_coeffs_f16_packed.push(pack_f16_pair(pair[0], pair[1]));
            }
        }

        Self {
            positions: scene.positions.clone(),
            covariances_f16_packed,
            octahedral_rotations,
            opacities: scene.opacities.clone(),
            sh_coeffs_f16_packed,
            sh_degree: deg,
            count,
        }
    }

    /// Number of packed `u32` words per splat for `self.sh_degree`.
    #[inline]
    pub fn sh_words_per_splat(&self) -> usize {
        sh_packed_words_for_degree(self.sh_degree)
    }

    /// Unpacks the symmetric 3D covariance `[xx, xy, xz, yy, yz, zz]` for splat `index`.
    #[inline]
    pub fn unpack_covariance_3d(&self, index: usize) -> [f32; 6] {
        let base = index * 3;
        let p0 = unpack_f16_pair(self.covariances_f16_packed[base]);
        let p1 = unpack_f16_pair(self.covariances_f16_packed[base + 1]);
        let p2 = unpack_f16_pair(self.covariances_f16_packed[base + 2]);
        [p0[0], p0[1], p1[0], p1[1], p2[0], p2[1]]
    }

    /// Unpacks the full 48-float channel-major SH buffer for splat `index` (zero-padding higher degrees).
    pub fn unpack_sh_48(&self, index: usize) -> [f32; 48] {
        let bases = sh_bases_for_degree(self.sh_degree);
        let words = self.sh_words_per_splat();
        let word_base = index * words;

        let mut linear = [0.0f32; 48];
        for (w_idx, &word) in self.sh_coeffs_f16_packed[word_base..word_base + words]
            .iter()
            .enumerate()
        {
            let pair = unpack_f16_pair(word);
            let idx0 = w_idx * 2;
            if idx0 < bases * 3 {
                linear[idx0] = pair[0];
            }
            if idx0 + 1 < bases * 3 {
                linear[idx0 + 1] = pair[1];
            }
        }

        let rest_per_ch = bases - 1;
        let mut out = [0.0f32; 48];
        out[0..3].copy_from_slice(&linear[0..3]);
        if rest_per_ch > 0 {
            out[3..3 + rest_per_ch].copy_from_slice(&linear[3..3 + rest_per_ch]);
            out[18..18 + rest_per_ch]
                .copy_from_slice(&linear[3 + rest_per_ch..3 + rest_per_ch * 2]);
            out[33..33 + rest_per_ch]
                .copy_from_slice(&linear[3 + rest_per_ch * 2..3 + rest_per_ch * 3]);
        }
        out
    }

    /// Dequantizes back into an uncompressed [`GaussianSceneSoa`].
    pub fn to_soa(&self) -> GaussianSceneSoa {
        let mut covariances_3d = Vec::with_capacity(self.count * 6);
        let mut sh_coeffs = Vec::with_capacity(self.count * 48);
        for i in 0..self.count {
            covariances_3d.extend_from_slice(&self.unpack_covariance_3d(i));
            sh_coeffs.extend_from_slice(&self.unpack_sh_48(i));
        }

        GaussianSceneSoa {
            positions: self.positions.clone(),
            covariances_3d,
            opacities: self.opacities.clone(),
            sh_coeffs,
            count: self.count,
        }
    }

    /// Packed 3D covariance bytes for GPU upload.
    #[inline]
    pub fn covariances_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.covariances_f16_packed)
    }

    /// Packed SH bytes for GPU upload.
    #[inline]
    pub fn sh_coeffs_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.sh_coeffs_f16_packed)
    }

    /// Total quantized footprint in bytes.
    #[inline]
    pub fn total_bytes(&self) -> usize {
        self.positions.len() * 4
            + self.covariances_f16_packed.len() * 4
            + self.octahedral_rotations.len() * 4
            + self.opacities.len() * 4
            + self.sh_coeffs_f16_packed.len() * 4
    }

    /// Equivalent uncompressed [`GaussianSceneSoa`] footprint in bytes.
    #[inline]
    pub fn unquantized_bytes(&self) -> usize {
        self.count * (3 + 6 + 1 + 48) * 4
    }

    /// Memory savings ratio (`unquantized_bytes / total_bytes`).
    #[inline]
    pub fn compression_ratio(&self) -> f32 {
        let q = self.total_bytes();
        if q == 0 {
            1.0
        } else {
            self.unquantized_bytes() as f32 / q as f32
        }
    }
}

#[cfg(test)]
mod quantized_tests {
    use super::*;

    #[test]
    fn test_octahedral_rotation_and_quantized_soa_roundtrip() {
        let mut scene = GaussianSceneSoa::new();
        let mut raw = RawPlyVertex::ZERO;
        raw.position = [0.5, -0.25, 1.5];
        raw.opacity = 1.2;
        raw.scale = [-1.2, -1.8, -0.9];
        raw.rot = [0.8, 0.3, -0.2, 0.4];
        raw.f_dc = [0.4, -0.2, 0.6];
        raw.f_rest[0] = 0.15;
        raw.f_rest[15] = -0.12;
        raw.f_rest[30] = 0.08;
        scene.push_raw(&raw);

        let q3 = scene.quantize();
        let rec3 = q3.to_soa();
        for k in 0..6 {
            assert!((scene.covariances_3d[k] - rec3.covariances_3d[k]).abs() < 2e-3);
        }
        for k in 0..48 {
            assert!((scene.sh_coeffs[k] - rec3.sh_coeffs[k]).abs() < 2e-3);
        }
        assert!(q3.compression_ratio() > 1.8);

        let q2 = scene.quantize_with_degree(2);
        assert!(q2.compression_ratio() > 2.5);

        let r = glam::Quat::from_euler(glam::EulerRot::XYZ, 0.4, -0.7, 1.1).normalize();
        let decoded = decode_octahedral_rotation(encode_octahedral_rotation(r));
        assert!(r.angle_between(decoded) < 0.01);
    }
}

impl Default for GaussianSceneSoa {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soa_allocation_and_push() {
        let mut soa = GaussianSceneSoa::with_capacity(2);
        assert_eq!(soa.len(), 0);
        assert!(soa.is_empty());

        let mut v = RawPlyVertex::ZERO;
        v.position = [1.0, 2.0, 3.0];
        v.opacity = 0.0; // sigmoid(0) = 0.5
        v.scale = [0.0, 0.0, 0.0]; // exp(0) = 1.0
        v.rot = [1.0, 0.0, 0.0, 0.0]; // identity rotation
        v.f_dc = [0.1, 0.2, 0.3];

        soa.push_raw(&v);
        assert_eq!(soa.len(), 1);
        assert!(!soa.is_empty());

        assert_eq!(soa.position(0), [1.0, 2.0, 3.0]);
        assert_eq!(soa.opacity(0), 0.5);
        assert_eq!(soa.covariance_3d(0), [1.0, 0.0, 0.0, 1.0, 0.0, 1.0]);
        assert_eq!(&soa.sh_coeffs_at(0)[..3], &[0.1, 0.2, 0.3]);
        assert_eq!(soa.sh_coeffs_at(0).len(), 48);

        soa.validate().expect("SOA must be valid");
    }

    #[test]
    fn test_soa_parallel_ingestion_matches_sequential() {
        let n = 5000;
        let mut raw_data = Vec::with_capacity(n);
        for i in 0..n {
            let fi = i as f32;
            let mut v = RawPlyVertex::ZERO;
            v.position = [fi, fi * 2.0, fi * 3.0];
            v.opacity = (fi % 10.0) - 5.0;
            v.scale = [-1.0 + (fi % 3.0), -2.0, 0.5];
            v.rot = [1.0, (fi % 4.0) * 0.1, 0.0, 0.0];
            v.f_dc = [fi * 0.01, fi * 0.02, fi * 0.03];
            for (j, slot) in v.f_rest.iter_mut().enumerate() {
                *slot = (fi + j as f32) * 0.001;
            }
            raw_data.push(v);
        }

        let seq_soa = GaussianSceneSoa::from_raw_vertices(&raw_data);
        let par_soa = GaussianSceneSoa::from_raw_vertices_par(&raw_data);

        seq_soa.validate().expect("Sequential SOA valid");
        par_soa.validate().expect("Parallel SOA valid");

        assert_eq!(seq_soa.count, n);
        assert_eq!(par_soa.count, n);
        assert_eq!(seq_soa.positions, par_soa.positions);
        assert_eq!(seq_soa.covariances_3d, par_soa.covariances_3d);
        assert_eq!(seq_soa.opacities, par_soa.opacities);
        assert_eq!(seq_soa.sh_coeffs, par_soa.sh_coeffs);
    }

    #[test]
    fn test_soa_gpu_bytes() {
        let mut soa = GaussianSceneSoa::new();
        let mut v = RawPlyVertex::ZERO;
        v.position = [1.0, 2.0, 3.0];
        soa.push_raw(&v);

        assert_eq!(soa.positions_bytes().len(), 3 * 4);
        assert_eq!(soa.covariances_bytes().len(), 6 * 4);
        assert_eq!(soa.opacities_bytes().len(), 4);
        assert_eq!(soa.sh_coeffs_bytes().len(), 48 * 4);
        assert_eq!(soa.total_bytes(), (3 + 6 + 1 + 48) * 4);
    }
}
