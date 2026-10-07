use bytemuck::{Pod, Zeroable};

/// Raw binary PLY vertex corresponding to the standard 3D Gaussian Splatting format.
///
/// Standard 3DGS binary PLY files (little-endian) store 62 properties (`f32`, 4 bytes each = 248 bytes per vertex)
/// in this exact sequential order:
/// - `[0..3]`: `x, y, z` (Positions, 3 floats = 12 bytes, offset 0)
/// - `[3..6]`: `nx, ny, nz` (Normals - discarded during activation, 3 floats = 12 bytes, offset 12)
/// - `[6..9]`: `f_dc_0, f_dc_1, f_dc_2` (Spherical Harmonics Degree 0 / DC, 3 floats = 12 bytes, offset 24)
/// - `[9..54]`: `f_rest_0` to `f_rest_44` (Spherical Harmonics Degrees 1-3, 45 floats = 180 bytes, offset 36)
/// - `[54..55]`: `opacity` (Raw logit, 1 float = 4 bytes, offset 216)
/// - `[55..58]`: `scale_0, scale_1, scale_2` (Raw log-scale, 3 floats = 12 bytes, offset 220)
/// - `[58..62]`: `rot_0, rot_1, rot_2, rot_3` (Raw quaternion coefficients: w, x, y, z, 4 floats = 16 bytes, offset 232)
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct RawPlyVertex {
    /// World-space position [x, y, z].
    pub position: [f32; 3],

    /// Surface normals [nx, ny, nz] (typically 0 in 3DGS, discarded during activation).
    pub normal: [f32; 3],

    /// Degree 0 / DC Spherical Harmonics coefficients [f_dc_0, f_dc_1, f_dc_2].
    pub f_dc: [f32; 3],

    /// Degree 1-3 Spherical Harmonics coefficients (45 floats).
    pub f_rest: [f32; 45],

    /// Raw logit opacity (pre-sigmoid).
    pub opacity: f32,

    /// Raw log-scale [scale_0, scale_1, scale_2] (pre-exponential).
    pub scale: [f32; 3],

    /// Raw quaternion coefficients [rot_0, rot_1, rot_2, rot_3] (WXYZ convention: rot_0=w, rot_1=x, rot_2=y, rot_3=z).
    pub rot: [f32; 4],
}

impl RawPlyVertex {
    /// Total byte size of one `RawPlyVertex` (248 bytes).
    pub const BYTE_SIZE: usize = std::mem::size_of::<Self>();

    /// Total number of 32-bit floating point properties (62 floats).
    pub const FLOAT_COUNT: usize = 62;

    /// Number of Spherical Harmonics Degree 1-3 coefficients.
    pub const SH_REST_COUNT: usize = 45;

    /// Number of total Spherical Harmonics coefficients (3 DC + 45 Rest = 48).
    pub const TOTAL_SH_COUNT: usize = 48;

    /// Zero-initialized raw vertex.
    pub const ZERO: Self = Self {
        position: [0.0; 3],
        normal: [0.0; 3],
        f_dc: [0.0; 3],
        f_rest: [0.0; 45],
        opacity: 0.0,
        scale: [0.0; 3],
        rot: [0.0; 4],
    };

    /// Returns a byte slice view of this vertex.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }

    /// Returns a mutable byte slice view of this vertex.
    #[inline]
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        bytemuck::bytes_of_mut(self)
    }

    /// Casts an aligned slice of bytes to a slice of `RawPlyVertex`.
    #[inline]
    pub fn cast_slice(bytes: &[u8]) -> Result<&[Self], bytemuck::PodCastError> {
        bytemuck::try_cast_slice(bytes)
    }

    /// Casts an aligned mutable slice of bytes to a mutable slice of `RawPlyVertex`.
    #[inline]
    pub fn cast_slice_mut(bytes: &mut [u8]) -> Result<&mut [Self], bytemuck::PodCastError> {
        bytemuck::try_cast_slice_mut(bytes)
    }
}

impl Default for RawPlyVertex {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

// Static compile-time verification of layout and alignments
const _: () = {
    assert!(std::mem::size_of::<RawPlyVertex>() == 248);
    assert!(std::mem::align_of::<RawPlyVertex>() == 4);
    assert!(std::mem::offset_of!(RawPlyVertex, position) == 0);
    assert!(std::mem::offset_of!(RawPlyVertex, normal) == 12);
    assert!(std::mem::offset_of!(RawPlyVertex, f_dc) == 24);
    assert!(std::mem::offset_of!(RawPlyVertex, f_rest) == 36);
    assert!(std::mem::offset_of!(RawPlyVertex, opacity) == 216);
    assert!(std::mem::offset_of!(RawPlyVertex, scale) == 220);
    assert!(std::mem::offset_of!(RawPlyVertex, rot) == 232);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raw_ply_vertex_size_and_offsets() {
        assert_eq!(std::mem::size_of::<RawPlyVertex>(), 248);
        assert_eq!(std::mem::align_of::<RawPlyVertex>(), 4);
        assert_eq!(std::mem::offset_of!(RawPlyVertex, position), 0);
        assert_eq!(std::mem::offset_of!(RawPlyVertex, normal), 12);
        assert_eq!(std::mem::offset_of!(RawPlyVertex, f_dc), 24);
        assert_eq!(std::mem::offset_of!(RawPlyVertex, f_rest), 36);
        assert_eq!(std::mem::offset_of!(RawPlyVertex, opacity), 216);
        assert_eq!(std::mem::offset_of!(RawPlyVertex, scale), 220);
        assert_eq!(std::mem::offset_of!(RawPlyVertex, rot), 232);
    }

    #[test]
    fn test_raw_ply_vertex_pod_roundtrip() {
        let mut vertex = RawPlyVertex::ZERO;
        vertex.position = [1.0, 2.0, 3.0];
        vertex.opacity = 0.85;
        vertex.scale = [-1.0, 0.5, 2.0];
        vertex.rot = [1.0, 0.0, 0.0, 0.0];

        let bytes = vertex.as_bytes();
        assert_eq!(bytes.len(), 248);

        let recovered: &[RawPlyVertex] = RawPlyVertex::cast_slice(bytes).expect("cast failed");
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0], vertex);
    }
}
