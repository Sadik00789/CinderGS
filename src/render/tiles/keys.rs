use bytemuck::{Pod, Zeroable};
use std::cmp::Ordering;

/// Composite 64-bit sort key representing tile location and positive depth.
///
/// Fields:
/// - `tile_id`: Screen tile index $t_y \times T_x + t_x$ (upper 32 bits of 64-bit sort key).
/// - `depth_bits`: Bitwise-monotonic unsigned 32-bit reinterpretation of positive float depth $t_z$.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq, Eq, PartialOrd, Ord)]
pub struct SortKey {
    pub tile_id: u32,
    pub depth_bits: u32,
}

impl SortKey {
    /// Constructs a `SortKey` from tile ID and floating-point depth $t_z$.
    #[inline]
    pub fn new(tile_id: u32, depth: f32) -> Self {
        Self {
            tile_id,
            depth_bits: depth.to_bits(),
        }
    }

    /// Constructs a `SortKey` from raw bits.
    #[inline]
    pub fn from_bits(tile_id: u32, depth_bits: u32) -> Self {
        Self {
            tile_id,
            depth_bits,
        }
    }

    /// Reconstructs the floating-point depth $t_z$.
    #[inline]
    pub fn depth(&self) -> f32 {
        f32::from_bits(self.depth_bits)
    }

    /// Packs into a single 64-bit unsigned integer: `(tile_id << 32) | depth_bits`.
    #[inline]
    pub fn as_u64(&self) -> u64 {
        ((self.tile_id as u64) << 32) | (self.depth_bits as u64)
    }

    /// Unpacks from a single 64-bit unsigned integer.
    #[inline]
    pub fn from_u64(packed: u64) -> Self {
        Self {
            tile_id: (packed >> 32) as u32,
            depth_bits: packed as u32,
        }
    }
}

/// Sort entry pairing a 64-bit composite sort key with the original Gaussian index.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq, Eq)]
pub struct SortEntry {
    /// Composite sort key: `(tile_id, depth_bits)`.
    pub key: SortKey,

    /// Original index of the Gaussian in the scene SoA layout.
    pub gaussian_id: u32,
}

impl SortEntry {
    /// Constructs a `SortEntry`.
    #[inline]
    pub fn new(key: SortKey, gaussian_id: u32) -> Self {
        Self { key, gaussian_id }
    }
}

impl PartialOrd for SortEntry {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SortEntry {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.key
            .cmp(&other.key)
            .then_with(|| self.gaussian_id.cmp(&other.gaussian_id))
    }
}

/// Boundary index range of sorted Gaussians overlapping a specific tile.
///
/// Range is half-open: `[start, end)`. If `start == end`, the tile is empty.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq, Eq)]
pub struct TileRange {
    /// Inclusive starting index in the sorted `SortEntry` buffer.
    pub start: u32,

    /// Exclusive ending index in the sorted `SortEntry` buffer.
    pub end: u32,
}

impl TileRange {
    /// Constant representing an empty tile range `[0, 0)`.
    pub const EMPTY: Self = Self { start: 0, end: 0 };

    /// Creates a tile range with explicit `start` and `end`.
    #[inline]
    pub fn new(start: u32, end: u32) -> Self {
        Self { start, end }
    }

    /// Returns `true` if the tile contains no overlapping Gaussians (`start >= end`).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }

    /// Number of Gaussians overlapping this tile.
    #[inline]
    pub fn count(&self) -> usize {
        if self.end > self.start {
            (self.end - self.start) as usize
        } else {
            0
        }
    }
}

// Compile-time layout assertions
const _: () = {
    assert!(std::mem::size_of::<SortKey>() == 8);
    assert!(std::mem::align_of::<SortKey>() == 4);
    assert!(std::mem::size_of::<SortEntry>() == 12);
    assert!(std::mem::align_of::<SortEntry>() == 4);
    assert!(std::mem::size_of::<TileRange>() == 8);
    assert!(std::mem::align_of::<TileRange>() == 4);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort_key_depth_roundtrip_and_ordering() {
        let key_a = SortKey::new(0, 1.5);
        let key_b = SortKey::new(0, 2.5);
        let key_c = SortKey::new(1, 0.5);

        assert_eq!(key_a.depth(), 1.5);
        assert_eq!(key_b.depth(), 2.5);
        assert!(key_a < key_b, "Smaller depth must sort before larger depth within same tile");
        assert!(key_b < key_c, "Smaller tile_id must sort before larger tile_id");

        let packed = key_a.as_u64();
        let unpacked = SortKey::from_u64(packed);
        assert_eq!(key_a, unpacked);
    }

    #[test]
    fn test_sort_entry_lexicographical_ordering() {
        let entry1 = SortEntry::new(SortKey::new(5, 1.0), 10);
        let entry2 = SortEntry::new(SortKey::new(5, 1.0), 11);
        let entry3 = SortEntry::new(SortKey::new(5, 2.0), 2);
        let entry4 = SortEntry::new(SortKey::new(6, 0.5), 1);

        assert!(entry1 < entry2);
        assert!(entry2 < entry3);
        assert!(entry3 < entry4);
    }

    #[test]
    fn test_tile_range_counts() {
        let r_empty = TileRange::new(5, 5);
        assert!(r_empty.is_empty());
        assert_eq!(r_empty.count(), 0);

        let r_valid = TileRange::new(10, 25);
        assert!(!r_valid.is_empty());
        assert_eq!(r_valid.count(), 15);
    }
}
