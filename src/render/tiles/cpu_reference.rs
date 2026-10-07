use super::{
    compute_gaussian_tile_bounds, count_tiles_touched, SortEntry, SortKey, TileConfigUniforms,
    TileRange,
};
use rayon::prelude::*;

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

    let mut tiles_per_gaussian = vec![0u32; count];

    const CHUNK_SIZE: usize = 2048;
    tiles_per_gaussian
        .par_chunks_mut(CHUNK_SIZE)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let base_i = chunk_idx * CHUNK_SIZE;
            for (local_i, count_out) in chunk.iter_mut().enumerate() {
                let i = base_i + local_i;
                let r = radii[i];
                if r <= 0 {
                    *count_out = 0;
                    continue;
                }

                let px = points_xy[i * 2];
                let py = points_xy[i * 2 + 1];

                *count_out = count_tiles_touched(
                    px,
                    py,
                    r as f32,
                    config.tiles_x,
                    config.tiles_y,
                    config.screen_width,
                    config.screen_height,
                );
            }
        });

    tiles_per_gaussian
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

/// Pass 3: Scatters sort entries `(SortKey, gaussian_id)` into contiguous buffer using offsets.
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

    let entries = vec![SortEntry::default(); total_entries];

    // Parallel scatter over chunks of Gaussians
    const CHUNK_SIZE: usize = 1024;
    let n = radii.len();
    let num_chunks = n.div_ceil(CHUNK_SIZE);

    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
        let start_i = chunk_idx * CHUNK_SIZE;
        let end_i = (start_i + CHUNK_SIZE).min(n);

        for i in start_i..end_i {
            let count = tiles_per_gaussian[i];
            if count == 0 {
                continue;
            }

            let r = radii[i];
            if r <= 0 {
                continue;
            }

            let px = points_xy[i * 2];
            let py = points_xy[i * 2 + 1];

            if let Some((tx_min, tx_max, ty_min, ty_max)) = compute_gaussian_tile_bounds(
                px,
                py,
                r as f32,
                config.tiles_x,
                config.tiles_y,
                config.screen_width,
                config.screen_height,
            ) {
                let depth = depths[i];
                let depth_bits = depth.to_bits();
                let mut curr_offset = offsets[i] as usize;

                for ty in ty_min..=ty_max {
                    for tx in tx_min..=tx_max {
                        let tile_id = ty * config.tiles_x + tx;
                        // Safe: write locations are disjoint per Gaussian
                        unsafe {
                            let ptr = entries.as_ptr() as *mut SortEntry;
                            *ptr.add(curr_offset) = SortEntry {
                                key: SortKey {
                                    tile_id,
                                    depth_bits,
                                },
                                gaussian_id: i as u32,
                            };
                        }
                        curr_offset += 1;
                    }
                }
            }
        }
    });

    entries
}

/// Pass 4a: Sorts `SortEntry` buffer lexicographically by `(tile_id, depth_bits)`.
pub fn sort_entries(entries: &mut [SortEntry]) {
    entries.par_sort_unstable();
}

/// Pass 4b: Scans sorted entries to extract start and end boundary indices for each tile.
pub fn identify_tile_ranges(sorted_entries: &[SortEntry], total_tiles: u32) -> Vec<TileRange> {
    let mut ranges = vec![TileRange::EMPTY; total_tiles as usize];
    let total_entries = sorted_entries.len();

    if total_entries == 0 {
        return ranges;
    }

    // Set first entry start
    let first_tile = sorted_entries[0].key.tile_id as usize;
    if first_tile < total_tiles as usize {
        ranges[first_tile].start = 0;
    }

    // Set last entry end
    let last_tile = sorted_entries[total_entries - 1].key.tile_id as usize;
    if last_tile < total_tiles as usize {
        ranges[last_tile].end = total_entries as u32;
    }

    // Parallel boundary transition detection
    const CHUNK_SIZE: usize = 4096;
    let num_chunks = (total_entries - 1).div_ceil(CHUNK_SIZE);

    (0..num_chunks).into_par_iter().for_each(|chunk_idx| {
        let start_k = chunk_idx * CHUNK_SIZE;
        let end_k = (start_k + CHUNK_SIZE).min(total_entries - 1);

        for k in start_k..end_k {
            let curr_tile = sorted_entries[k].key.tile_id as usize;
            let next_tile = sorted_entries[k + 1].key.tile_id as usize;

            if curr_tile != next_tile {
                let boundary_index = (k + 1) as u32;
                unsafe {
                    let ptr = ranges.as_ptr() as *mut TileRange;
                    if curr_tile < total_tiles as usize {
                        (*ptr.add(curr_tile)).end = boundary_index;
                    }
                    if next_tile < total_tiles as usize {
                        (*ptr.add(next_tile)).start = boundary_index;
                    }
                }
            }
        }
    });

    ranges
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
}
