use crate::math::{
    activate_opacity, activate_rotation, activate_scale, compute_covariance_3d,
};
use crate::scene::raw_vertex::RawPlyVertex;
use rayon::prelude::*;
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

        positions
            .par_chunks_mut(chunk_size * 3)
            .zip(covariances_3d.par_chunks_mut(chunk_size * 6))
            .zip(opacities.par_chunks_mut(chunk_size))
            .zip(sh_coeffs.par_chunks_mut(chunk_size * 48))
            .zip(raw.par_chunks(chunk_size))
            .for_each(|((((pos_chunk, cov_chunk), op_chunk), sh_chunk), raw_chunk)| {
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
            });

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
            for j in 0..45 {
                v.f_rest[j] = (fi + j as f32) * 0.001;
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
