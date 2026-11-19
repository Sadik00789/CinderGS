use crate::render::gpu::GpuError;
use crate::render::gpu_pool::calculate_exponential_growth;
use crate::render::kernels::scan::{ScanScratch, dispatch_cube_count, gpu_exclusive_scan};
use cubecl::prelude::*;
use cubecl::server::Handle;

/// Number of threads per workgroup for 4-bit LSD radix histogram & scatter passes.
pub const RADIX_WORKGROUP_THREADS: u32 = 256;

/// Radix size (`2^4 = 16` buckets per nibble pass).
pub const RADIX_BUCKETS: usize = 16;

/// Counts 4-bit digit occurrences per 256-element workgroup block into digit-major
/// `histograms[digit * num_blocks + block_id]`.
#[allow(clippy::unnecessary_cast)]
#[cube(launch)]
pub fn radix_histogram_kernel(
    keys: &[u32],
    histograms: &mut [u32],
    num_elements: u32,
    num_blocks: u32,
    shift: u32,
) {
    let lid = UNIT_POS as usize;
    let block_id = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) as usize;

    let local_hist = Shared::<[Atomic<u32>]>::new_slice(16usize);
    if lid < 16usize {
        local_hist[lid].store(0u32);
    }
    sync_cube();

    if block_id < num_blocks as usize {
        let idx = block_id * 256usize + lid;
        if idx < num_elements as usize {
            let digit = ((keys[idx] >> shift) & 0xFu32) as usize;
            local_hist[digit].fetch_add(1u32);
        }
    }
    sync_cube();

    if lid < 16usize && block_id < num_blocks as usize {
        histograms[lid * (num_blocks as usize) + block_id] = local_hist[lid].load();
    }
}

/// Stable per-block scatter for one 4-bit radix pass.
///
/// When `use_subgroups` is true, computes in-block digit ranks using hardware
/// `plane_exclusive_sum` + small inter-plane prefix scan; otherwise uses a
/// shared-memory 256-element Blelloch exclusive scan. Never calls `terminate!()`
/// before barriers so uniform control flow holds across all lanes.
#[allow(clippy::too_many_arguments, clippy::unnecessary_cast)]
#[cube(launch)]
pub fn radix_scatter_kernel(
    keys_in: &[u32],
    vals_in: &[u32],
    scanned_histograms: &[u32],
    keys_out: &mut [u32],
    vals_out: &mut [u32],
    num_elements: u32,
    num_blocks: u32,
    shift: u32,
    #[comptime] use_subgroups: bool,
) {
    let lid = UNIT_POS as usize;
    let block_id = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) as usize;
    let in_bounds_block = block_id < num_blocks as usize;
    let idx = block_id * 256usize + lid;
    let valid = in_bounds_block && idx < num_elements as usize;

    let key = if valid { keys_in[idx] } else { 0u32 };
    let val = if valid { vals_in[idx] } else { 0u32 };
    let my_digit = (key >> shift) & 0xFu32;

    let mut shared_tmp = Shared::<[u32]>::new_slice(256usize);

    for d in 0u32..16u32 {
        let has_d = valid && my_digit == d;
        if comptime!(use_subgroups) {
            let pred = if has_d { 1u32 } else { 0u32 };
            let in_plane_prefix = plane_exclusive_sum(pred);
            let num_planes = (CUBE_DIM / PLANE_DIM) as usize;
            if UNIT_POS_PLANE == PLANE_DIM - 1u32 {
                shared_tmp[PLANE_POS as usize] = in_plane_prefix + pred;
            }
            sync_cube();

            if lid == 0usize {
                let mut acc = 0u32;
                for p in 0usize..num_planes {
                    let count = shared_tmp[p];
                    shared_tmp[p] = acc;
                    acc += count;
                }
            }
            sync_cube();

            if has_d {
                let local_rank = shared_tmp[PLANE_POS as usize] + in_plane_prefix;
                let global_base =
                    scanned_histograms[(d as usize) * (num_blocks as usize) + block_id];
                let out_pos = (global_base + local_rank) as usize;
                keys_out[out_pos] = key;
                vals_out[out_pos] = val;
            }
            sync_cube();
        } else {
            shared_tmp[lid] = if has_d { 1u32 } else { 0u32 };
            sync_cube();

            let mut offset = 1usize;
            while offset < 256usize {
                let step = offset * 2usize;
                let active_pairs = 256usize / step;
                if lid < active_pairs {
                    let ai = step * lid + offset - 1usize;
                    let bi = step * lid + step - 1usize;
                    shared_tmp[bi] += shared_tmp[ai];
                }
                sync_cube();
                offset = step;
            }

            if lid == 0usize {
                shared_tmp[255usize] = 0u32;
            }
            sync_cube();

            let mut down_offset = 128usize;
            while down_offset >= 1usize {
                let step = down_offset * 2usize;
                let active_pairs = 256usize / step;
                if lid < active_pairs {
                    let ai = step * lid + down_offset - 1usize;
                    let bi = step * lid + step - 1usize;
                    let tmp = shared_tmp[ai];
                    shared_tmp[ai] = shared_tmp[bi];
                    shared_tmp[bi] += tmp;
                }
                sync_cube();
                down_offset /= 2usize;
            }

            if has_d {
                let local_rank = shared_tmp[lid];
                let global_base =
                    scanned_histograms[(d as usize) * (num_blocks as usize) + block_id];
                let out_pos = (global_base + local_rank) as usize;
                keys_out[out_pos] = key;
                vals_out[out_pos] = val;
            }
            sync_cube();
        }
    }
}

/// Writes sequential identity indices `[0, 1, ..., num_elements - 1]`.
#[allow(clippy::unnecessary_cast)]
#[cube(launch)]
pub fn iota_u32_kernel(out: &mut [u32], num_elements: u32) {
    let block_id = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) as usize;
    let idx = block_id * 256usize + (UNIT_POS as usize);
    if idx < num_elements as usize {
        out[idx] = idx as u32;
    }
}

/// Gathers two parallel `u32` arrays through permutation `indices`.
#[allow(clippy::unnecessary_cast)]
#[cube(launch)]
pub fn gather_u32_pair_kernel(
    src_a: &[u32],
    src_b: &[u32],
    indices: &[u32],
    out_a: &mut [u32],
    out_b: &mut [u32],
    num_elements: u32,
) {
    let block_id = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) as usize;
    let idx = block_id * 256usize + (UNIT_POS as usize);
    if idx < num_elements as usize {
        let p = indices[idx] as usize;
        out_a[idx] = src_a[p];
        out_b[idx] = src_b[p];
    }
}

/// Persistent GPU scratch buffers for 4-bit LSD radix sort across `u32` and 64-bit pair paths.
pub struct RadixSortScratch {
    histograms: Option<(Handle, usize)>,
    scanned_histograms: Option<(Handle, usize)>,
    alt_keys: Option<(Handle, usize)>,
    alt_vals: Option<(Handle, usize)>,
    perm_a: Option<(Handle, usize)>,
    gathered_keys: Option<(Handle, usize)>,
    gathered_vals: Option<(Handle, usize)>,
    pub scan_scratch: ScanScratch,
}

impl Default for RadixSortScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl RadixSortScratch {
    /// Creates empty radix sort scratch buffers (allocates lazily and grows 1.5x).
    pub fn new() -> Self {
        Self {
            histograms: None,
            scanned_histograms: None,
            alt_keys: None,
            alt_vals: None,
            perm_a: None,
            gathered_keys: None,
            gathered_vals: None,
            scan_scratch: ScanScratch::new(),
        }
    }

    fn ensure_u32(
        slot: &mut Option<(Handle, usize)>,
        client: &Client,
        needed_elems: usize,
    ) -> Handle {
        let needed = needed_elems.max(1);
        match slot {
            Some((handle, cap)) if *cap >= needed => handle.clone(),
            Some((_, cap)) => {
                let new_cap = calculate_exponential_growth(*cap, needed);
                let handle = client.empty(new_cap * 4);
                *slot = Some((handle.clone(), new_cap));
                handle
            }
            None => {
                let new_cap = needed.max(256);
                let handle = client.empty(new_cap * 4);
                *slot = Some((handle.clone(), new_cap));
                handle
            }
        }
    }

    /// Returns `(sorted_keys, sorted_vals)` after [`gpu_radix_sort_u32_pairs`].
    #[inline]
    pub fn u32_sorted_outputs<'a>(
        &'a self,
        keys: &'a Handle,
        vals: &'a Handle,
        in_alt: bool,
    ) -> (&'a Handle, &'a Handle) {
        if in_alt {
            (
                &self.alt_keys.as_ref().expect("alt_keys allocated").0,
                &self.alt_vals.as_ref().expect("alt_vals allocated").0,
            )
        } else {
            (keys, vals)
        }
    }

    /// Returns `(sorted_tile_ids, sorted_splat_ids)` after [`gpu_radix_sort_u64_pairs`].
    #[inline]
    pub fn u64_sorted_outputs(&self, in_alt: bool) -> (&Handle, &Handle) {
        if in_alt {
            (
                &self.alt_keys.as_ref().expect("alt_keys allocated").0,
                &self.alt_vals.as_ref().expect("alt_vals allocated").0,
            )
        } else {
            (
                &self
                    .gathered_keys
                    .as_ref()
                    .expect("gathered_keys allocated")
                    .0,
                &self
                    .gathered_vals
                    .as_ref()
                    .expect("gathered_vals allocated")
                    .0,
            )
        }
    }
}

/// Stable GPU LSD 4-bit radix sort over `(keys, vals)` covering the low `sort_bits` bits.
///
/// Returns `Ok(true)` if the final sorted result resides in `(scratch.alt_keys, scratch.alt_vals)`,
/// or `Ok(false)` if it resides in `(keys, vals)`.
pub fn gpu_radix_sort_u32_pairs(
    client: &Client,
    keys: &Handle,
    vals: &Handle,
    num_elements: usize,
    sort_bits: u32,
    use_subgroups: bool,
    scratch: &mut RadixSortScratch,
) -> Result<bool, GpuError> {
    if num_elements <= 1 || sort_bits == 0 {
        return Ok(false);
    }

    let num_passes = sort_bits.div_ceil(4).max(1);
    let num_blocks = num_elements.div_ceil(RADIX_WORKGROUP_THREADS as usize);
    let hist_len = num_blocks * RADIX_BUCKETS;

    let histograms = RadixSortScratch::ensure_u32(&mut scratch.histograms, client, hist_len);
    let scanned_histograms =
        RadixSortScratch::ensure_u32(&mut scratch.scanned_histograms, client, hist_len);
    let alt_keys = RadixSortScratch::ensure_u32(&mut scratch.alt_keys, client, num_elements);
    let alt_vals = RadixSortScratch::ensure_u32(&mut scratch.alt_vals, client, num_elements);

    let wg_count = dispatch_cube_count(num_blocks as u32);
    let mut src_in_alt = false;

    for pass in 0..num_passes {
        let shift = pass * 4;
        let (k_in, v_in, k_out, v_out) = if !src_in_alt {
            (keys, vals, &alt_keys, &alt_vals)
        } else {
            (&alt_keys, &alt_vals, keys, vals)
        };

        unsafe {
            radix_histogram_kernel::launch(
                client,
                wg_count.clone(),
                CubeDim::new_1d(RADIX_WORKGROUP_THREADS),
                BufferArg::from_raw_parts(k_in.clone(), num_elements),
                BufferArg::from_raw_parts(histograms.clone(), hist_len),
                num_elements as u32,
                num_blocks as u32,
                shift,
            );
        }

        let _ = gpu_exclusive_scan(
            client,
            &histograms,
            &scanned_histograms,
            hist_len,
            &mut scratch.scan_scratch,
        )?;

        unsafe {
            radix_scatter_kernel::launch(
                client,
                wg_count.clone(),
                CubeDim::new_1d(RADIX_WORKGROUP_THREADS),
                BufferArg::from_raw_parts(k_in.clone(), num_elements),
                BufferArg::from_raw_parts(v_in.clone(), num_elements),
                BufferArg::from_raw_parts(scanned_histograms.clone(), hist_len),
                BufferArg::from_raw_parts(k_out.clone(), num_elements),
                BufferArg::from_raw_parts(v_out.clone(), num_elements),
                num_elements as u32,
                num_blocks as u32,
                shift,
                use_subgroups,
            );
        }

        src_in_alt = !src_in_alt;
    }

    Ok(src_in_alt)
}

/// Stable GPU 64-bit `(tile_id, depth_u32)` radix sort fallback when `total_tiles > 65535`.
///
/// Sorts `depths_u32` ascending using 8 4-bit passes with permutation indices, gathers
/// `(tile_ids, splat_ids)` in depth order, then stably sorts by `tile_ids` over `tile_bits`.
/// Call [`RadixSortScratch::u64_sorted_outputs`] with the returned `bool` to obtain the
/// sorted `(tile_ids, splat_ids)` handles.
#[allow(clippy::too_many_arguments)]
pub fn gpu_radix_sort_u64_pairs(
    client: &Client,
    tile_ids: &Handle,
    depths_u32: &Handle,
    splat_ids: &Handle,
    num_elements: usize,
    tile_bits: u32,
    use_subgroups: bool,
    scratch: &mut RadixSortScratch,
) -> Result<bool, GpuError> {
    let perm_a = RadixSortScratch::ensure_u32(&mut scratch.perm_a, client, num_elements);
    let gathered_keys =
        RadixSortScratch::ensure_u32(&mut scratch.gathered_keys, client, num_elements);
    let gathered_vals =
        RadixSortScratch::ensure_u32(&mut scratch.gathered_vals, client, num_elements);

    if num_elements <= 1 {
        if num_elements == 1 {
            let wg = dispatch_cube_count(1);
            unsafe {
                iota_u32_kernel::launch(
                    client,
                    wg.clone(),
                    CubeDim::new_1d(RADIX_WORKGROUP_THREADS),
                    BufferArg::from_raw_parts(perm_a.clone(), 1),
                    1u32,
                );
                gather_u32_pair_kernel::launch(
                    client,
                    wg,
                    CubeDim::new_1d(RADIX_WORKGROUP_THREADS),
                    BufferArg::from_raw_parts(tile_ids.clone(), 1),
                    BufferArg::from_raw_parts(splat_ids.clone(), 1),
                    BufferArg::from_raw_parts(perm_a, 1),
                    BufferArg::from_raw_parts(gathered_keys, 1),
                    BufferArg::from_raw_parts(gathered_vals, 1),
                    1u32,
                );
            }
        }
        return Ok(false);
    }

    let num_blocks = num_elements.div_ceil(RADIX_WORKGROUP_THREADS as usize);
    let wg = dispatch_cube_count(num_blocks as u32);

    unsafe {
        iota_u32_kernel::launch(
            client,
            wg.clone(),
            CubeDim::new_1d(RADIX_WORKGROUP_THREADS),
            BufferArg::from_raw_parts(perm_a.clone(), num_elements),
            num_elements as u32,
        );
    }

    let depth_in_alt = gpu_radix_sort_u32_pairs(
        client,
        depths_u32,
        &perm_a,
        num_elements,
        32,
        use_subgroups,
        scratch,
    )?;

    let perm_handle = if depth_in_alt {
        scratch
            .alt_vals
            .as_ref()
            .expect("alt_vals allocated")
            .0
            .clone()
    } else {
        perm_a
    };

    unsafe {
        gather_u32_pair_kernel::launch(
            client,
            wg,
            CubeDim::new_1d(RADIX_WORKGROUP_THREADS),
            BufferArg::from_raw_parts(tile_ids.clone(), num_elements),
            BufferArg::from_raw_parts(splat_ids.clone(), num_elements),
            BufferArg::from_raw_parts(perm_handle, num_elements),
            BufferArg::from_raw_parts(gathered_keys.clone(), num_elements),
            BufferArg::from_raw_parts(gathered_vals.clone(), num_elements),
            num_elements as u32,
        );
    }

    gpu_radix_sort_u32_pairs(
        client,
        &gathered_keys,
        &gathered_vals,
        num_elements,
        tile_bits.max(1),
        use_subgroups,
        scratch,
    )
}
