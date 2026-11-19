use cubecl::prelude::*;

/// Workgroup size for Stage 3 binning & range identification kernels.
pub const BINNING_WORKGROUP_THREADS: u32 = 256;

/// Computes `(tile_bits, depth_bits)` for packed 32-bit `(tile_id << depth_bits) | depth_q`
/// Stage 3 sort keys (`total_tiles <= 65535`).
///
/// Allocates only the minimum bits required by `tile_id` so the remaining bits
/// (up to 28 relative IEEE-754 exponent+mantissa bits) maximize depth precision.
#[inline]
pub fn compute_packed_key_bits(total_tiles: usize) -> (u32, u32) {
    let tile_bits = if total_tiles <= 1 {
        1u32
    } else {
        32 - ((total_tiles as u32) - 1).leading_zeros()
    };
    let depth_bits = 32u32.saturating_sub(tile_bits).min(28);
    (tile_bits, depth_bits)
}

/// Packs positive view-space depth `tz` into `depth_bits` monotonic bits relative
/// to `near_plane`'s IEEE-754 exponent base, matching [`emit_tile_keys_kernel`].
#[inline]
pub fn pack_relative_depth_bits(tz: f32, near_plane: f32, depth_bits: u32) -> u32 {
    let db = depth_bits.min(28);
    let base_bits = near_plane.max(1e-6).to_bits() & 0x7F80_0000;
    let raw_bits = tz.max(1e-6).to_bits();
    let rel_bits = raw_bits.saturating_sub(base_bits).min(0x0FFF_FFFF);
    rel_bits >> (28 - db)
}

/// Exact convex quadratic ellipse-vs-tile intersection test mirroring
/// [`crate::render::tiles::ellipse_intersects_tile`].
#[allow(clippy::too_many_arguments, clippy::unnecessary_cast)]
#[cube]
pub fn ellipse_intersects_tile_gpu(
    mu_x: f32,
    mu_y: f32,
    conic_a: f32,
    conic_b: f32,
    conic_c: f32,
    alpha_2d: f32,
    tx: u32,
    ty: u32,
    screen_width: u32,
    screen_height: u32,
) -> bool {
    if alpha_2d < (1.0f32 / 255.0f32) || screen_width == 0u32 || screen_height == 0u32 {
        false
    } else {
        let x_start = tx * 16u32;
        let y_start = ty * 16u32;
        if x_start >= screen_width || y_start >= screen_height {
            false
        } else {
            let x_end = u32::min(x_start + 15u32, screen_width - 1u32);
            let y_end = u32::min(y_start + 15u32, screen_height - 1u32);

            let tile_min_x = x_start as f32 + 0.5f32;
            let tile_max_x = x_end as f32 + 0.5f32;
            let tile_min_y = y_start as f32 + 0.5f32;
            let tile_max_y = y_end as f32 + 0.5f32;

            let dx0 = tile_min_x - mu_x;
            let dx1 = tile_max_x - mu_x;
            let dy0 = tile_min_y - mu_y;
            let dy1 = tile_max_y - mu_y;

            let x_near = f32::clamp(0.0f32, dx0, dx1);
            let y1 = f32::clamp(-conic_b * x_near / f32::max(conic_c, 1e-12f32), dy0, dy1);
            let q1 = conic_a * x_near * x_near + 2.0f32 * conic_b * x_near * y1 + conic_c * y1 * y1;

            let y_near = f32::clamp(0.0f32, dy0, dy1);
            let x2 = f32::clamp(-conic_b * y_near / f32::max(conic_a, 1e-12f32), dx0, dx1);
            let q2 = conic_a * x2 * x2 + 2.0f32 * conic_b * x2 * y_near + conic_c * y_near * y_near;

            let min_quad = f32::min(q1, q2);
            let q_cutoff = f32::clamp(2.0f32 * f32::ln(255.0f32 * alpha_2d), 0.0f32, 8.0f32);
            min_quad <= q_cutoff + 1e-4f32
        }
    }
}

/// Counts intersected 16x16 tiles per Gaussian using tight ellipse AABBs and exact
/// quadratic ellipse-vs-tile culling.
///
/// Indices in `[active_gaussians..max_gaussians)` write `0` so prefix scan across
/// `max_gaussians` works without host synchronization between Stage 1 and Stage 3.
#[allow(clippy::too_many_arguments, clippy::unnecessary_cast)]
#[cube(launch)]
pub fn count_tiles_kernel(
    splat_attrs: &[f32],
    extents_packed: &[u32],
    split_counter: &[u32],
    tiles_touched: &mut [u32],
    screen_width: u32,
    screen_height: u32,
    tiles_x: u32,
    tiles_y: u32,
    num_rest_gaussians: u32,
    max_gaussians: u32,
) {
    let block_id = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) as usize;
    let idx = block_id * 256usize + (UNIT_POS as usize);
    if idx >= max_gaussians as usize {
        terminate!();
    }

    let active_gaussians = num_rest_gaussians + split_counter[0usize];
    if idx >= active_gaussians as usize || tiles_x == 0u32 || tiles_y == 0u32 {
        tiles_touched[idx] = 0u32;
        terminate!();
    }

    let packed = extents_packed[idx];
    if packed == 0u32 {
        tiles_touched[idx] = 0u32;
        terminate!();
    }

    let ext_x = (packed & 0xFFFFu32) as f32;
    let ext_y = ((packed >> 16u32) & 0xFFFFu32) as f32;
    let attr_base = idx * 10usize;
    let mu_x = splat_attrs[attr_base];
    let mu_y = splat_attrs[attr_base + 1usize];
    let conic_a = splat_attrs[attr_base + 2usize];
    let conic_b = splat_attrs[attr_base + 3usize];
    let conic_c = splat_attrs[attr_base + 4usize];
    let alpha_2d = splat_attrs[attr_base + 5usize];

    let max_tx = tiles_x as i32 - 1i32;
    let max_ty = tiles_y as i32 - 1i32;
    let tx_min = i32::clamp(f32::floor((mu_x - ext_x) / 16.0f32) as i32, 0i32, max_tx) as u32;
    let tx_max = i32::clamp(f32::floor((mu_x + ext_x) / 16.0f32) as i32, 0i32, max_tx) as u32;
    let ty_min = i32::clamp(f32::floor((mu_y - ext_y) / 16.0f32) as i32, 0i32, max_ty) as u32;
    let ty_max = i32::clamp(f32::floor((mu_y + ext_y) / 16.0f32) as i32, 0i32, max_ty) as u32;

    let mut count = 0u32;
    let mut ty = ty_min;
    while ty <= ty_max {
        let mut tx = tx_min;
        while tx <= tx_max {
            if ellipse_intersects_tile_gpu(
                mu_x,
                mu_y,
                conic_a,
                conic_b,
                conic_c,
                alpha_2d,
                tx,
                ty,
                screen_width,
                screen_height,
            ) {
                count += 1u32;
            }
            tx += 1u32;
        }
        ty += 1u32;
    }

    tiles_touched[idx] = count;
}

/// Emits packed 32-bit `(tile_id << depth_bits) | depth_q` sort keys and `splat_id` values
/// starting at `tile_offsets[idx]` (`total_tiles <= 65535`).
#[allow(clippy::too_many_arguments, clippy::unnecessary_cast)]
#[cube(launch)]
pub fn emit_tile_keys_kernel(
    splat_attrs: &[f32],
    extents_packed: &[u32],
    tile_offsets: &[u32],
    split_counter: &[u32],
    out_keys: &mut [u32],
    out_splat_ids: &mut [u32],
    screen_width: u32,
    screen_height: u32,
    tiles_x: u32,
    tiles_y: u32,
    num_rest_gaussians: u32,
    base_depth_bits: u32,
    depth_bits: u32,
) {
    let block_id = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) as usize;
    let idx = block_id * 256usize + (UNIT_POS as usize);
    let active_gaussians = num_rest_gaussians + split_counter[0usize];
    if idx >= active_gaussians as usize || tiles_x == 0u32 || tiles_y == 0u32 {
        terminate!();
    }

    let packed = extents_packed[idx];
    if packed == 0u32 {
        terminate!();
    }

    let ext_x = (packed & 0xFFFFu32) as f32;
    let ext_y = ((packed >> 16u32) & 0xFFFFu32) as f32;
    let attr_base = idx * 10usize;
    let mu_x = splat_attrs[attr_base];
    let mu_y = splat_attrs[attr_base + 1usize];
    let conic_a = splat_attrs[attr_base + 2usize];
    let conic_b = splat_attrs[attr_base + 3usize];
    let conic_c = splat_attrs[attr_base + 4usize];
    let alpha_2d = splat_attrs[attr_base + 5usize];
    let tz = splat_attrs[attr_base + 9usize];

    let raw_bits = u32::reinterpret(f32::max(tz, 1e-6f32));
    let rel_bits = if raw_bits > base_depth_bits {
        u32::min(raw_bits - base_depth_bits, 0x0FFF_FFFFu32)
    } else {
        0u32
    };
    let depth_shift = 28u32 - depth_bits;
    let depth_q = rel_bits >> depth_shift;

    let max_tx = tiles_x as i32 - 1i32;
    let max_ty = tiles_y as i32 - 1i32;
    let tx_min = i32::clamp(f32::floor((mu_x - ext_x) / 16.0f32) as i32, 0i32, max_tx) as u32;
    let tx_max = i32::clamp(f32::floor((mu_x + ext_x) / 16.0f32) as i32, 0i32, max_tx) as u32;
    let ty_min = i32::clamp(f32::floor((mu_y - ext_y) / 16.0f32) as i32, 0i32, max_ty) as u32;
    let ty_max = i32::clamp(f32::floor((mu_y + ext_y) / 16.0f32) as i32, 0i32, max_ty) as u32;

    let mut cursor = tile_offsets[idx] as usize;
    let mut ty = ty_min;
    while ty <= ty_max {
        let mut tx = tx_min;
        while tx <= tx_max {
            if ellipse_intersects_tile_gpu(
                mu_x,
                mu_y,
                conic_a,
                conic_b,
                conic_c,
                alpha_2d,
                tx,
                ty,
                screen_width,
                screen_height,
            ) {
                let tile_id = ty * tiles_x + tx;
                out_keys[cursor] = (tile_id << depth_bits) | depth_q;
                out_splat_ids[cursor] = idx as u32;
                cursor += 1usize;
            }
            tx += 1u32;
        }
        ty += 1u32;
    }
}

/// Emits split 32-bit `tile_id`, `depth_u32`, and `splat_id` arrays for large
/// framebuffers (`total_tiles > 65535`) sorted via two-pass [`crate::render::kernels::gpu_radix_sort_u64_pairs`].
#[allow(clippy::too_many_arguments, clippy::unnecessary_cast)]
#[cube(launch)]
pub fn emit_tile_keys_u64_kernel(
    splat_attrs: &[f32],
    extents_packed: &[u32],
    tile_offsets: &[u32],
    split_counter: &[u32],
    out_tile_ids: &mut [u32],
    out_depths_u32: &mut [u32],
    out_splat_ids: &mut [u32],
    screen_width: u32,
    screen_height: u32,
    tiles_x: u32,
    tiles_y: u32,
    num_rest_gaussians: u32,
) {
    let block_id = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) as usize;
    let idx = block_id * 256usize + (UNIT_POS as usize);
    let active_gaussians = num_rest_gaussians + split_counter[0usize];
    if idx >= active_gaussians as usize || tiles_x == 0u32 || tiles_y == 0u32 {
        terminate!();
    }

    let packed = extents_packed[idx];
    if packed == 0u32 {
        terminate!();
    }

    let ext_x = (packed & 0xFFFFu32) as f32;
    let ext_y = ((packed >> 16u32) & 0xFFFFu32) as f32;
    let attr_base = idx * 10usize;
    let mu_x = splat_attrs[attr_base];
    let mu_y = splat_attrs[attr_base + 1usize];
    let conic_a = splat_attrs[attr_base + 2usize];
    let conic_b = splat_attrs[attr_base + 3usize];
    let conic_c = splat_attrs[attr_base + 4usize];
    let alpha_2d = splat_attrs[attr_base + 5usize];
    let tz = splat_attrs[attr_base + 9usize];
    let depth_bits = u32::reinterpret(f32::max(tz, 0.0f32));

    let max_tx = tiles_x as i32 - 1i32;
    let max_ty = tiles_y as i32 - 1i32;
    let tx_min = i32::clamp(f32::floor((mu_x - ext_x) / 16.0f32) as i32, 0i32, max_tx) as u32;
    let tx_max = i32::clamp(f32::floor((mu_x + ext_x) / 16.0f32) as i32, 0i32, max_tx) as u32;
    let ty_min = i32::clamp(f32::floor((mu_y - ext_y) / 16.0f32) as i32, 0i32, max_ty) as u32;
    let ty_max = i32::clamp(f32::floor((mu_y + ext_y) / 16.0f32) as i32, 0i32, max_ty) as u32;

    let mut cursor = tile_offsets[idx] as usize;
    let mut ty = ty_min;
    while ty <= ty_max {
        let mut tx = tx_min;
        while tx <= tx_max {
            if ellipse_intersects_tile_gpu(
                mu_x,
                mu_y,
                conic_a,
                conic_b,
                conic_c,
                alpha_2d,
                tx,
                ty,
                screen_width,
                screen_height,
            ) {
                let tile_id = ty * tiles_x + tx;
                out_tile_ids[cursor] = tile_id;
                out_depths_u32[cursor] = depth_bits;
                out_splat_ids[cursor] = idx as u32;
                cursor += 1usize;
            }
            tx += 1u32;
        }
        ty += 1u32;
    }
}

/// Resets `tile_ranges[0..total_tiles * 2]` to `(0, 0)` on the GPU.
#[allow(clippy::unnecessary_cast)]
#[cube(launch)]
pub fn clear_tile_ranges_kernel(tile_ranges: &mut [u32], total_tiles: u32) {
    let block_id = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) as usize;
    let tile_id = block_id * 256usize + (UNIT_POS as usize);
    if tile_id < total_tiles as usize {
        tile_ranges[tile_id * 2usize] = 0u32;
        tile_ranges[tile_id * 2usize + 1usize] = 0u32;
    }
}

/// Identifies `(start, end)` entry spans per tile from sorted 32-bit packed keys
/// `(tile_id << depth_bits) | depth_q`.
#[allow(clippy::unnecessary_cast)]
#[cube(launch)]
pub fn identify_tile_ranges_kernel(
    sorted_keys: &[u32],
    tile_ranges: &mut [u32],
    num_entries: u32,
    total_tiles: u32,
    depth_bits: u32,
) {
    let block_id = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) as usize;
    let idx = block_id * 256usize + (UNIT_POS as usize);
    if idx >= num_entries as usize {
        terminate!();
    }

    let tile_id = (sorted_keys[idx] >> depth_bits) as usize;
    if tile_id >= total_tiles as usize {
        terminate!();
    }

    if idx == 0usize {
        tile_ranges[tile_id * 2usize] = 0u32;
    } else {
        let prev_tile = (sorted_keys[idx - 1usize] >> depth_bits) as usize;
        if tile_id != prev_tile {
            if prev_tile < total_tiles as usize {
                tile_ranges[prev_tile * 2usize + 1usize] = idx as u32;
            }
            tile_ranges[tile_id * 2usize] = idx as u32;
        }
    }

    if idx + 1usize == num_entries as usize {
        tile_ranges[tile_id * 2usize + 1usize] = num_entries;
    }
}

/// Identifies `(start, end)` entry spans per tile from sorted `u32` tile IDs
/// (`total_tiles > 65535` 64-bit sort fallback).
#[allow(clippy::unnecessary_cast)]
#[cube(launch)]
pub fn identify_tile_ranges_u32_tile_kernel(
    sorted_tile_ids: &[u32],
    tile_ranges: &mut [u32],
    num_entries: u32,
    total_tiles: u32,
) {
    let block_id = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) as usize;
    let idx = block_id * 256usize + (UNIT_POS as usize);
    if idx >= num_entries as usize {
        terminate!();
    }

    let tile_id = sorted_tile_ids[idx] as usize;
    if tile_id >= total_tiles as usize {
        terminate!();
    }

    if idx == 0usize {
        tile_ranges[tile_id * 2usize] = 0u32;
    } else {
        let prev_tile = sorted_tile_ids[idx - 1usize] as usize;
        if tile_id != prev_tile {
            if prev_tile < total_tiles as usize {
                tile_ranges[prev_tile * 2usize + 1usize] = idx as u32;
            }
            tile_ranges[tile_id * 2usize] = idx as u32;
        }
    }

    if idx + 1usize == num_entries as usize {
        tile_ranges[tile_id * 2usize + 1usize] = num_entries;
    }
}
