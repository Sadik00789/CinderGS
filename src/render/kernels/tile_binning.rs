use cubecl::prelude::*;

/// CubeCL kernel for Pass 1: computing touched tile counts for all Gaussians in parallel.
#[cube(launch)]
pub fn count_tiles_kernel(
    points_xy: &[f32],
    radii: &[i32],
    tiles_per_gaussian: &mut [u32],
    config: &[u32],
    num_gaussians: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= num_gaussians as usize {
        terminate!();
    }

    let r = radii[idx];
    if r <= 0 {
        tiles_per_gaussian[idx] = 0;
        terminate!();
    }

    let screen_width = config[0];
    let screen_height = config[1];
    let tiles_x = config[2];
    let tiles_y = config[3];

    let px = points_xy[idx * 2];
    let py = points_xy[idx * 2 + 1];
    let r_f = r as f32;

    let min_x = px - r_f;
    let max_x = px + r_f;
    let min_y = py - r_f;
    let max_y = py + r_f;

    // Viewport culling check
    if max_x < 0.0
        || min_x >= screen_width as f32
        || max_y < 0.0
        || min_y >= screen_height as f32
    {
        tiles_per_gaussian[idx] = 0;
        terminate!();
    }

    let max_tx = (tiles_x - 1) as f32;
    let max_ty = (tiles_y - 1) as f32;

    // 0.0625 = 1.0 / 16.0
    let tx_min = f32::clamp(f32::floor(min_x * 0.0625), 0.0, max_tx) as u32;
    let tx_max = f32::clamp(f32::floor(max_x * 0.0625), 0.0, max_tx) as u32;
    let ty_min = f32::clamp(f32::floor(min_y * 0.0625), 0.0, max_ty) as u32;
    let ty_max = f32::clamp(f32::floor(max_y * 0.0625), 0.0, max_ty) as u32;

    if tx_max < tx_min || ty_max < ty_min {
        tiles_per_gaussian[idx] = 0;
        terminate!();
    }

    let count = (tx_max - tx_min + 1) * (ty_max - ty_min + 1);
    tiles_per_gaussian[idx] = count;
}

/// CubeCL kernel for Pass 3: scattering (SortKey, gaussian_id) into contiguous memory.
///
/// Output buffer layout: Stride of 3 u32s per `SortEntry`:
/// - `[0]`: `tile_id`
/// - `[1]`: `depth_bits`
/// - `[2]`: `gaussian_id`
#[cube(launch)]
pub fn scatter_keys_kernel(
    points_xy: &[f32],
    depths: &[f32],
    radii: &[i32],
    offsets: &[u32],
    tiles_per_gaussian: &[u32],
    sort_entries: &mut [u32],
    config: &[u32],
    num_gaussians: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= num_gaussians as usize {
        terminate!();
    }

    let count = tiles_per_gaussian[idx];
    if count == 0 {
        terminate!();
    }

    let r = radii[idx];
    if r <= 0 {
        terminate!();
    }

    let screen_width = config[0];
    let screen_height = config[1];
    let tiles_x = config[2];
    let tiles_y = config[3];

    let px = points_xy[idx * 2];
    let py = points_xy[idx * 2 + 1];
    let r_f = r as f32;

    let min_x = px - r_f;
    let max_x = px + r_f;
    let min_y = py - r_f;
    let max_y = py + r_f;

    // Viewport bounds
    if max_x < 0.0
        || min_x >= screen_width as f32
        || max_y < 0.0
        || min_y >= screen_height as f32
    {
        terminate!();
    }

    let max_tx = (tiles_x - 1) as f32;
    let max_ty = (tiles_y - 1) as f32;

    let tx_min = f32::clamp(f32::floor(min_x * 0.0625), 0.0, max_tx) as u32;
    let tx_max = f32::clamp(f32::floor(max_x * 0.0625), 0.0, max_tx) as u32;
    let ty_min = f32::clamp(f32::floor(min_y * 0.0625), 0.0, max_ty) as u32;
    let ty_max = f32::clamp(f32::floor(max_y * 0.0625), 0.0, max_ty) as u32;

    let depth = depths[idx];
    let depth_bits = depth.to_bits();
    let mut curr = offsets[idx] as usize;
    let g_id = idx as u32;

    for ty in ty_min..ty_max + 1 {
        for tx in tx_min..tx_max + 1 {
            let tile_id = ty * tiles_x + tx;
            let out_idx = curr * 3;
            sort_entries[out_idx] = tile_id;
            sort_entries[out_idx + 1] = depth_bits;
            sort_entries[out_idx + 2] = g_id;
            curr += 1;
        }
    }
}

/// CubeCL kernel initializing the tile ranges buffer to empty `[0, 0]`.
#[cube(launch)]
pub fn initialize_tile_ranges_kernel(
    tile_ranges: &mut [u32],
    total_tiles: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= total_tiles as usize {
        terminate!();
    }

    let out_idx = idx * 2;
    tile_ranges[out_idx] = 0;
    tile_ranges[out_idx + 1] = 0;
}

/// CubeCL kernel for Pass 4b: detecting tile boundary transitions in the sorted entries buffer.
///
/// Output buffer layout: Stride of 2 u32s per `TileRange`:
/// - `[tile_id * 2]`: `start`
/// - `[tile_id * 2 + 1]`: `end`
#[cube(launch)]
pub fn identify_tile_ranges_kernel(
    sorted_entries: &[u32],
    tile_ranges: &mut [u32],
    total_entries: u32,
    total_tiles: u32,
) {
    let k = ABSOLUTE_POS;
    if k >= total_entries as usize {
        terminate!();
    }

    let k_u32 = k as u32;
    let curr_tile = sorted_entries[k * 3] as usize;
    let total_tiles_usize = total_tiles as usize;

    // First entry marks the start of the first non-empty tile
    if k_u32 == 0 && curr_tile < total_tiles_usize {
        tile_ranges[curr_tile * 2] = 0;
    }

    // Last entry marks the end of the last non-empty tile
    if k_u32 == total_entries - 1 && curr_tile < total_tiles_usize {
        tile_ranges[curr_tile * 2 + 1] = total_entries;
    }

    // Interior transitions
    if k_u32 + 1 < total_entries {
        let next_tile = sorted_entries[(k + 1) * 3] as usize;
        if curr_tile != next_tile {
            let boundary = k_u32 + 1;
            if curr_tile < total_tiles_usize {
                tile_ranges[curr_tile * 2 + 1] = boundary;
            }
            if next_tile < total_tiles_usize {
                tile_ranges[next_tile * 2] = boundary;
            }
        }
    }
}
