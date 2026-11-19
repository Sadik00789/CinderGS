use crate::render::gpu::GpuError;
use crate::render::gpu_pool::calculate_exponential_growth;
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl_common::bytes::Bytes;

/// Number of threads per workgroup for the 512-element Blelloch prefix scan.
pub const SCAN_WORKGROUP_THREADS: u32 = 256;

/// Number of `u32` elements exclusively scanned by one workgroup (`2 * 256`).
pub const SCAN_BLOCK_SIZE: usize = 512;

#[inline]
pub fn dispatch_cube_count(num_workgroups: u32) -> CubeCount {
    if num_workgroups == 0 {
        CubeCount::Static(0, 1, 1)
    } else if num_workgroups <= 65_535 {
        CubeCount::Static(num_workgroups, 1, 1)
    } else {
        let wx = 65_535u32;
        CubeCount::Static(wx, num_workgroups.div_ceil(wx), 1)
    }
}

/// 512-element per-workgroup Blelloch exclusive prefix sum (`input[..num_elements]`).
///
/// Writes per-element exclusive sums into `output` and per-block totals into `block_sums`.
#[allow(clippy::unnecessary_cast)]
#[cube(launch)]
pub fn scan_block_512_kernel(
    input: &[u32],
    output: &mut [u32],
    block_sums: &mut [u32],
    num_elements: u32,
    num_blocks: u32,
) {
    let lid = UNIT_POS as usize;
    let block_id = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) as usize;

    let mut shared = Shared::<[u32]>::new_slice(512usize);
    let block_base = block_id * 512usize;
    let i0 = block_base + lid;
    let i1 = block_base + lid + 256usize;
    let valid_block = block_id < num_blocks as usize;

    shared[lid] = if valid_block && i0 < num_elements as usize {
        input[i0]
    } else {
        0u32
    };
    shared[lid + 256usize] = if valid_block && i1 < num_elements as usize {
        input[i1]
    } else {
        0u32
    };
    sync_cube();

    let mut offset = 1usize;
    while offset < 512usize {
        let step = offset * 2usize;
        let active = 512usize / step;
        if lid < active {
            let ai = step * lid + offset - 1usize;
            let bi = step * lid + step - 1usize;
            shared[bi] += shared[ai];
        }
        sync_cube();
        offset = step;
    }

    if lid == 0usize {
        if valid_block {
            block_sums[block_id] = shared[511usize];
        }
        shared[511usize] = 0u32;
    }
    sync_cube();

    let mut down_offset = 256usize;
    while down_offset >= 1usize {
        let step = down_offset * 2usize;
        let active = 512usize / step;
        if lid < active {
            let ai = step * lid + down_offset - 1usize;
            let bi = step * lid + step - 1usize;
            let tmp = shared[ai];
            shared[ai] = shared[bi];
            shared[bi] += tmp;
        }
        sync_cube();
        down_offset /= 2usize;
    }

    if valid_block {
        if i0 < num_elements as usize {
            output[i0] = shared[lid];
        }
        if i1 < num_elements as usize {
            output[i1] = shared[lid + 256usize];
        }
    }
}

/// Adds scanned per-block offsets back to each 512-element block in `data`.
#[allow(clippy::unnecessary_cast)]
#[cube(launch)]
pub fn scan_add_block_offsets_kernel(
    data: &mut [u32],
    block_offsets: &[u32],
    num_elements: u32,
    num_blocks: u32,
) {
    let lid = UNIT_POS as usize;
    let block_id = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) as usize;
    if block_id >= num_blocks as usize {
        terminate!();
    }

    let add_val = block_offsets[block_id];
    let block_base = block_id * 512usize;
    let i0 = block_base + lid;
    let i1 = block_base + lid + 256usize;

    if i0 < num_elements as usize {
        data[i0] += add_val;
    }
    if i1 < num_elements as usize {
        data[i1] += add_val;
    }
}

/// Persistent GPU scratch buffers for hierarchical multi-level exclusive prefix sum
/// covering up to $512^3 = 134{,}217{,}728$ `u32` elements with zero per-frame allocation.
pub struct ScanScratch {
    level1_sums: Option<(Handle, usize)>,
    level1_scanned: Option<(Handle, usize)>,
    level2_sums: Option<(Handle, usize)>,
    level2_scanned: Option<(Handle, usize)>,
    total_sum: Option<Handle>,
}

impl Default for ScanScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl ScanScratch {
    /// Creates an empty scratch container (allocates lazily on first scan).
    pub fn new() -> Self {
        Self {
            level1_sums: None,
            level1_scanned: None,
            level2_sums: None,
            level2_scanned: None,
            total_sum: None,
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
                let new_cap = needed.max(64);
                let handle = client.empty(new_cap * 4);
                *slot = Some((handle.clone(), new_cap));
                handle
            }
        }
    }

    fn ensure_total_sum(&mut self, client: &Client) -> Handle {
        if let Some(handle) = &self.total_sum {
            handle.clone()
        } else {
            let handle = client.create_from_slice(bytemuck::bytes_of(&0u32));
            self.total_sum = Some(handle.clone());
            handle
        }
    }
}

/// Runs a multi-level hierarchical Blelloch exclusive prefix sum on the GPU.
///
/// Supports up to $512^3 = 134{,}217{,}728$ elements and returns a handle containing the
/// single `u32` total sum across `[0..num_elements)`.
pub fn gpu_exclusive_scan(
    client: &Client,
    input: &Handle,
    output: &Handle,
    num_elements: usize,
    scratch: &mut ScanScratch,
) -> Result<Handle, GpuError> {
    let total_handle = scratch.ensure_total_sum(client);
    if num_elements == 0 {
        client.write(&total_handle, Bytes::from_bytes_vec(vec![0u8; 4]));
        return Ok(total_handle);
    }

    let num_blocks_l1 = num_elements.div_ceil(SCAN_BLOCK_SIZE);
    let l1_sums = ScanScratch::ensure_u32(&mut scratch.level1_sums, client, num_blocks_l1);

    unsafe {
        scan_block_512_kernel::launch(
            client,
            dispatch_cube_count(num_blocks_l1 as u32),
            CubeDim::new_1d(SCAN_WORKGROUP_THREADS),
            BufferArg::from_raw_parts(input.clone(), num_elements),
            BufferArg::from_raw_parts(output.clone(), num_elements),
            BufferArg::from_raw_parts(
                if num_blocks_l1 == 1 {
                    total_handle.clone()
                } else {
                    l1_sums.clone()
                },
                num_blocks_l1,
            ),
            num_elements as u32,
            num_blocks_l1 as u32,
        );
    }

    if num_blocks_l1 == 1 {
        return Ok(total_handle);
    }

    let l1_scanned = ScanScratch::ensure_u32(&mut scratch.level1_scanned, client, num_blocks_l1);
    let num_blocks_l2 = num_blocks_l1.div_ceil(SCAN_BLOCK_SIZE);
    let l2_sums = ScanScratch::ensure_u32(&mut scratch.level2_sums, client, num_blocks_l2);

    unsafe {
        scan_block_512_kernel::launch(
            client,
            dispatch_cube_count(num_blocks_l2 as u32),
            CubeDim::new_1d(SCAN_WORKGROUP_THREADS),
            BufferArg::from_raw_parts(l1_sums, num_blocks_l1),
            BufferArg::from_raw_parts(l1_scanned.clone(), num_blocks_l1),
            BufferArg::from_raw_parts(
                if num_blocks_l2 == 1 {
                    total_handle.clone()
                } else {
                    l2_sums.clone()
                },
                num_blocks_l2,
            ),
            num_blocks_l1 as u32,
            num_blocks_l2 as u32,
        );
    }

    if num_blocks_l2 > 1 {
        let num_blocks_l3 = num_blocks_l2.div_ceil(SCAN_BLOCK_SIZE);
        if num_blocks_l3 > 1 {
            return Err(GpuError::Execution(format!(
                "gpu_exclusive_scan exceeded 3-level limit ({} elements)",
                num_elements
            )));
        }
        let l2_scanned =
            ScanScratch::ensure_u32(&mut scratch.level2_scanned, client, num_blocks_l2);
        unsafe {
            scan_block_512_kernel::launch(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(SCAN_WORKGROUP_THREADS),
                BufferArg::from_raw_parts(l2_sums, num_blocks_l2),
                BufferArg::from_raw_parts(l2_scanned.clone(), num_blocks_l2),
                BufferArg::from_raw_parts(total_handle.clone(), 1),
                num_blocks_l2 as u32,
                1u32,
            );

            scan_add_block_offsets_kernel::launch(
                client,
                dispatch_cube_count(num_blocks_l2 as u32),
                CubeDim::new_1d(SCAN_WORKGROUP_THREADS),
                BufferArg::from_raw_parts(l1_scanned.clone(), num_blocks_l1),
                BufferArg::from_raw_parts(l2_scanned, num_blocks_l2),
                num_blocks_l1 as u32,
                num_blocks_l2 as u32,
            );
        }
    }

    unsafe {
        scan_add_block_offsets_kernel::launch(
            client,
            dispatch_cube_count(num_blocks_l1 as u32),
            CubeDim::new_1d(SCAN_WORKGROUP_THREADS),
            BufferArg::from_raw_parts(output.clone(), num_elements),
            BufferArg::from_raw_parts(l1_scanned, num_blocks_l1),
            num_elements as u32,
            num_blocks_l1 as u32,
        );
    }

    Ok(total_handle)
}

/// Runs [`gpu_exclusive_scan`] and reads back the scalar `u32` total sum.
pub fn gpu_exclusive_scan_total(
    client: &Client,
    input: &Handle,
    output: &Handle,
    num_elements: usize,
    scratch: &mut ScanScratch,
) -> Result<u32, GpuError> {
    let total_handle = gpu_exclusive_scan(client, input, output, num_elements, scratch)?;
    if num_elements == 0 {
        return Ok(0);
    }
    let bytes = client
        .read_one(total_handle)
        .map_err(|e| GpuError::Execution(format!("scan total readback failed: {e:?}")))?;
    if bytes.len() < 4 {
        return Err(GpuError::Execution(format!(
            "scan total readback too short: {} bytes",
            bytes.len()
        )));
    }
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}
