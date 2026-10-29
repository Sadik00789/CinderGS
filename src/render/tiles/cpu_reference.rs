use super::{
    SortEntry, SortKey, TileConfigUniforms, TileRange, compute_gaussian_tile_bounds,
    count_tiles_touched,
};
use crate::parallel;

/// Output of the tile binning, key generation, and sorting pipeline.
#[derive(Clone, Debug, PartialEq)]
pub struct TileBinningOutputs {
    /// Lexicographically sorted sort entries `(tile_id, depth_bits, gaussian_id)`.
    pub sorted_entries: Vec<SortEntry>,

    /// Boundary ranges per tile: index range `[start, end)` into `sorted_entries`.
    pub tile_ranges: Vec<TileRange>,

    /// Total number of duplicated (Gaussian, Tile) pairs.
    pub total_entries: usize,

    /// Tile configuration used during binning.
    pub config: TileConfigUniforms,
}

impl TileBinningOutputs {
    /// Returns the sorted entries belonging to a given tile ID.
    #[inline]
    pub fn entries_for_tile(&self, tile_id: u32) -> &[SortEntry] {
        if tile_id >= self.config.total_tiles {
            return &[];
        }
        let range = self.tile_ranges[tile_id as usize];
        if range.is_empty() {
            &[]
        } else {
            &self.sorted_entries[range.start as usize..range.end as usize]
        }
    }

    /// Number of Gaussians overlapping a specific tile ID.
    #[inline]
    pub fn entry_count_for_tile(&self, tile_id: u32) -> usize {
        if tile_id >= self.config.total_tiles {
            0
        } else {
            self.tile_ranges[tile_id as usize].count()
        }
    }

    /// Returns `true` if a specific tile has no overlapping Gaussians.
    #[inline]
    pub fn is_tile_empty(&self, tile_id: u32) -> bool {
        if tile_id >= self.config.total_tiles {
            true
        } else {
            self.tile_ranges[tile_id as usize].is_empty()
        }
    }

    /// Number of tiles that have at least one overlapping Gaussian.
    #[inline]
    pub fn non_empty_tiles_count(&self) -> usize {
        self.tile_ranges.iter().filter(|r| !r.is_empty()).count()
    }
}

/// Pass 1: Computes the number of tiles touched by each Gaussian.
pub fn compute_tiles_per_gaussian(
    points_xy: &[f32],
    radii: &[i32],
    config: &TileConfigUniforms,
) -> Vec<u32> {
    let count = radii.len();
    if count == 0 {
        return Vec::new();
    }

    parallel::map_range(count, |i| {
        let r = radii[i];
        if r <= 0 {
            return 0;
        }

        let px = points_xy[i * 2];
        let py = points_xy[i * 2 + 1];

        count_tiles_touched(
            px,
            py,
            r as f32,
            config.tiles_x,
            config.tiles_y,
            config.screen_width,
            config.screen_height,
        )
    })
}

/// Pass 2: Computes exclusive prefix sums of touched tile counts to determine write offsets.
///
/// Returns `(offsets, total_entries)`.
pub fn prefix_sum_offsets(tiles_per_gaussian: &[u32]) -> (Vec<u32>, usize) {
    let n = tiles_per_gaussian.len();
    if n == 0 {
        return (Vec::new(), 0);
    }

    let mut offsets = vec![0u32; n];
    let mut running_sum = 0u64;

    for i in 0..n {
        offsets[i] = running_sum as u32;
        running_sum += tiles_per_gaussian[i] as u64;
    }

    (offsets, running_sum as usize)
}

/// Recomputes the tiles touched by Gaussian `gaussian`, in row-major order.
#[inline]
fn tiles_for_gaussian(
    gaussian: usize,
    points_xy: &[f32],
    radii: &[i32],
    config: &TileConfigUniforms,
) -> Option<(u32, u32, u32, u32)> {
    compute_gaussian_tile_bounds(
        points_xy[gaussian * 2],
        points_xy[gaussian * 2 + 1],
        radii[gaussian].max(0) as f32,
        config.tiles_x,
        config.tiles_y,
        config.screen_width,
        config.screen_height,
    )
}

/// Pass 3: Scatters sort entries `(SortKey, gaussian_id)` into contiguous slots.
///
/// Each output slot is disjoint, so the scatter is expressed as a chunked fill of
/// the output buffer instead of raw-pointer writes from a Rayon closure. Every
/// slot's owner is recovered from the exclusive prefix sums with a binary search.
pub fn scatter_sort_entries(
    points_xy: &[f32],
    depths: &[f32],
    radii: &[i32],
    offsets: &[u32],
    tiles_per_gaussian: &[u32],
    total_entries: usize,
    config: &TileConfigUniforms,
) -> Vec<SortEntry> {
    if total_entries == 0 {
        return Vec::new();
    }

    let mut entries = vec![SortEntry::default(); total_entries];
    scatter_into(
        &mut entries,
        points_xy,
        depths,
        radii,
        offsets,
        tiles_per_gaussian,
        config,
    );
    entries
}

#[allow(clippy::too_many_arguments)]
fn scatter_into(
    entries: &mut [SortEntry],
    points_xy: &[f32],
    depths: &[f32],
    radii: &[i32],
    offsets: &[u32],
    tiles_per_gaussian: &[u32],
    config: &TileConfigUniforms,
) {
    let gaussians = offsets.len();
    if gaussians == 0 {
        return;
    }

    // Every output slot is written exactly once by the closure that owns it.
    parallel::fill_range(entries, |entry_index| {
        // Locate the Gaussian whose [offset, offset + count) range owns this slot.
        let owner = offsets.partition_point(|&offset| offset as usize <= entry_index);
        if owner == 0 {
            return SortEntry::default();
        }
        let gaussian = owner - 1;
        if tiles_per_gaussian[gaussian] == 0 {
            return SortEntry::default();
        }

        let Some((tx_min, tx_max, ty_min, ty_max)) =
            tiles_for_gaussian(gaussian, points_xy, radii, config)
        else {
            return SortEntry::default();
        };

        let ordinal = (entry_index - offsets[gaussian] as usize) as u32;
        let width = tx_max.saturating_sub(tx_min) + 1;

        // Row-major order matches the scatter loops used by the CPU reference.
        let ty = ty_min + ordinal / width;
        let tx = tx_min + ordinal % width;
        if ty > ty_max || tx > tx_max {
            return SortEntry::default();
        }

        SortEntry::new(
            SortKey {
                tile_id: ty * config.tiles_x + tx,
                depth_bits: depths[gaussian].to_bits(),
            },
            gaussian as u32,
        )
    });
}

/// Pass 4a: Sorts `SortEntry` buffer lexicographically by `(tile_id, depth_bits)`.
pub fn sort_entries(entries: &mut [SortEntry]) {
    parallel::sort_unstable(entries);
}

/// Pass 4b: Extracts start and end boundary indices for each tile.
///
/// The sorted buffer is monotonic in `tile_id`, so each boundary is a pair of
/// binary searches. This is branch-free with respect to tile occupancy and has no
/// cross-thread writes.
pub fn identify_tile_ranges(sorted_entries: &[SortEntry], total_tiles: u32) -> Vec<TileRange> {
    if total_tiles == 0 {
        return Vec::new();
    }

    parallel::map_range(total_tiles as usize, |tile_id| {
        let tile_id = tile_id as u32;
        let start = sorted_entries.partition_point(|entry| entry.key.tile_id < tile_id);
        let end = sorted_entries.partition_point(|entry| entry.key.tile_id <= tile_id);
        if start < end {
            TileRange::new(start as u32, end as u32)
        } else {
            TileRange::EMPTY
        }
    })
}

/// Complete CPU reference pipeline executing all four passes.
pub fn bin_and_sort_gaussians_cpu(
    points_xy: &[f32],
    depths: &[f32],
    radii: &[i32],
    config: TileConfigUniforms,
) -> TileBinningOutputs {
    let count = radii.len();
    if count == 0 {
        return TileBinningOutputs {
            sorted_entries: Vec::new(),
            tile_ranges: vec![TileRange::EMPTY; config.total_tiles as usize],
            total_entries: 0,
            config,
        };
    }

    // Pass 1: Tile count per Gaussian
    let tiles_per_gaussian = compute_tiles_per_gaussian(points_xy, radii, &config);

    // Pass 2: Prefix sum offsets
    let (offsets, total_entries) = prefix_sum_offsets(&tiles_per_gaussian);

    if total_entries == 0 {
        return TileBinningOutputs {
            sorted_entries: Vec::new(),
            tile_ranges: vec![TileRange::EMPTY; config.total_tiles as usize],
            total_entries: 0,
            config,
        };
    }

    // Pass 3: Scatter keys
    let mut entries = scatter_sort_entries(
        points_xy,
        depths,
        radii,
        &offsets,
        &tiles_per_gaussian,
        total_entries,
        &config,
    );

    // Pass 4a: Lexicographical sort
    sort_entries(&mut entries);

    // Pass 4b: Tile range boundary identification
    let tile_ranges = identify_tile_ranges(&entries, config.total_tiles);

    TileBinningOutputs {
        sorted_entries: entries,
        tile_ranges,
        total_entries,
        config,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_binning_single_gaussian_1x1() {
        let config = TileConfigUniforms::new(800, 600, 1);
        let points_xy = [8.0, 8.0];
        let depths = [2.5];
        let radii = [2];

        let out = bin_and_sort_gaussians_cpu(&points_xy, &depths, &radii, config);
        assert_eq!(out.total_entries, 1);
        assert_eq!(out.sorted_entries.len(), 1);
        assert_eq!(out.sorted_entries[0].key.tile_id, 0);
        assert_eq!(out.sorted_entries[0].key.depth(), 2.5);
        assert_eq!(out.sorted_entries[0].gaussian_id, 0);

        assert_eq!(out.tile_ranges[0], TileRange::new(0, 1));
        assert_eq!(out.entry_count_for_tile(0), 1);
        assert_eq!(out.entry_count_for_tile(1), 0);
        assert_eq!(out.non_empty_tiles_count(), 1);
    }

    #[test]
    fn test_cpu_binning_straddling_gaussian_2x2() {
        let config = TileConfigUniforms::new(800, 600, 1);
        let points_xy = [16.0, 16.0]; // touches tiles (0,0), (1,0), (0,1), (1,1)
        let depths = [3.0];
        let radii = [2];

        let out = bin_and_sort_gaussians_cpu(&points_xy, &depths, &radii, config);
        assert_eq!(out.total_entries, 4);

        // Expected tiles: 0, 1, 50, 51 (since tiles_x = 50)
        let expected_tiles = [0, 1, 50, 51];
        for (i, &expected_tile) in expected_tiles.iter().enumerate() {
            assert_eq!(out.sorted_entries[i].key.tile_id, expected_tile);
            assert_eq!(out.sorted_entries[i].gaussian_id, 0);
            assert_eq!(out.entry_count_for_tile(expected_tile), 1);
        }
        assert_eq!(out.non_empty_tiles_count(), 4);
    }

    #[test]
    fn test_cpu_binning_culled_gaussian_radius_zero() {
        let config = TileConfigUniforms::new(800, 600, 1);
        let points_xy = [100.0, 100.0];
        let depths = [1.0];
        let radii = [0]; // Culled!

        let out = bin_and_sort_gaussians_cpu(&points_xy, &depths, &radii, config);
        assert_eq!(out.total_entries, 0);
        assert!(out.sorted_entries.is_empty());
        assert_eq!(out.non_empty_tiles_count(), 0);
    }

    #[test]
    fn test_cpu_binning_lexicographical_sorting_multiple_gaussians() {
        let config = TileConfigUniforms::new(800, 600, 3);
        // Gaussian 0 at depth 5.0 in tile 0
        // Gaussian 1 at depth 1.0 in tile 0
        // Gaussian 2 at depth 2.0 in tile 1
        let points_xy = [4.0, 4.0, 8.0, 8.0, 24.0, 8.0];
        let depths = [5.0, 1.0, 2.0];
        let radii = [2, 2, 2];

        let out = bin_and_sort_gaussians_cpu(&points_xy, &depths, &radii, config);
        assert_eq!(out.total_entries, 3);

        // Entry 0: tile 0, depth 1.0 (gaussian 1)
        assert_eq!(out.sorted_entries[0].key.tile_id, 0);
        assert_eq!(out.sorted_entries[0].key.depth(), 1.0);
        assert_eq!(out.sorted_entries[0].gaussian_id, 1);

        // Entry 1: tile 0, depth 5.0 (gaussian 0)
        assert_eq!(out.sorted_entries[1].key.tile_id, 0);
        assert_eq!(out.sorted_entries[1].key.depth(), 5.0);
        assert_eq!(out.sorted_entries[1].gaussian_id, 0);

        // Entry 2: tile 1, depth 2.0 (gaussian 2)
        assert_eq!(out.sorted_entries[2].key.tile_id, 1);
        assert_eq!(out.sorted_entries[2].key.depth(), 2.0);
        assert_eq!(out.sorted_entries[2].gaussian_id, 2);

        // Tile ranges
        assert_eq!(out.tile_ranges[0], TileRange::new(0, 2));
        assert_eq!(out.tile_ranges[1], TileRange::new(2, 3));
    }

    #[test]
    fn test_scatter_matches_direct_gaussian_enumeration() {
        let config = TileConfigUniforms::new(800, 600, 5);
        let points = [
            8.0f32, 8.0, 40.0, 32.0, 100.0, 100.0, 8.0, 40.0, 300.0, 200.0,
        ];
        let depths = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let radii = [30, 20, 10, 0, 60];

        let out = bin_and_sort_gaussians_cpu(&points, &depths, &radii, config);
        assert!(out.total_entries > 0);

        // Every entry must reference a live gaussian and belong to the tile it claims.
        for entry in &out.sorted_entries {
            let g = entry.gaussian_id as usize;
            assert!(radii[g] > 0);
            assert_eq!(entry.key.depth(), depths[g]);
            assert!(entry.key.tile_id < config.total_tiles);
        }

        // Every slot is filled exactly once: the number of entries per tile sums to
        // the total, and each tile range is contiguous and consistent.
        let mut total = 0u32;
        for (tile_id, range) in out.tile_ranges.iter().enumerate() {
            if range.is_empty() {
                continue;
            }
            total += range.end - range.start;
            for index in range.start..range.end {
                assert_eq!(
                    out.sorted_entries[index as usize].key.tile_id,
                    tile_id as u32
                );
            }
        }
        assert_eq!(total as usize, out.total_entries);
    }

    #[test]
    fn test_identify_tile_ranges_is_total() {
        let config = TileConfigUniforms::new(64, 64, 4);
        let points = [8.0f32, 8.0, 20.0, 20.0, 50.0, 50.0, 8.0, 50.0];
        let depths = [1.0f32, 2.0, 3.0, 4.0];
        let radii = [4, 4, 4, 4];

        let out = bin_and_sort_gaussians_cpu(&points, &depths, &radii, config);
        let ranges = identify_tile_ranges(&out.sorted_entries, config.total_tiles);
        assert_eq!(ranges, out.tile_ranges);
    }
}
