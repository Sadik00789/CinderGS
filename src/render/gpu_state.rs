use crate::deformation::CageScene;
use crate::render::camera::CameraUniforms;
use crate::render::gpu::{GpuCompute, GpuError};
use crate::render::gpu_pool::calculate_exponential_growth;
use crate::render::kernels::{
    RadixSortScratch, ScanScratch, clear_tile_ranges_kernel, compute_packed_key_bits,
    compute_tet_transforms_kernel, count_tiles_kernel, deform_and_split_compact_kernel,
    dispatch_cube_count, emit_tile_keys_kernel, emit_tile_keys_u64_kernel,
    gpu_exclusive_scan_total, gpu_radix_sort_u32_pairs, gpu_radix_sort_u64_pairs,
    identify_tile_ranges_kernel, identify_tile_ranges_u32_tile_kernel, project_gaussians_v2_kernel,
    tile_compositor_v2_kernel,
};
use crate::render::rasterizer::{CompositorUniforms, RasterizedImage};
use crate::scene::soa::GaussianSceneSoa;
use cubecl::prelude::*;
use cubecl::server::{Handle, ProfileError};
use cubecl::wgpu::{AutoCompiler, WgpuServer};
use cubecl_common::bytes::Bytes;
use std::time::Instant;

const COMPUTE_WG: u32 = 256;

/// Returns row stride in `u32` pixels aligned to `wgpu::COPY_BYTES_PER_ROW_ALIGNMENT` (256 bytes = 64 RGBA8 pixels).
#[inline]
pub fn padded_rgba8_row_u32(width: u32) -> u32 {
    width.div_ceil(64) * 64
}

fn fnv1a_step(mut hash: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x1000_0000_01B3);
    }
    hash
}

fn hash_f32_sample(mut hash: u64, data: &[f32]) -> u64 {
    hash = fnv1a_step(hash, &(data.len() as u64).to_le_bytes());
    let step = (data.len() / 64).max(1);
    let mut i = 0;
    while i < data.len() {
        hash = fnv1a_step(hash, &data[i].to_bits().to_le_bytes());
        i += step;
    }
    if let Some(&last) = data.last() {
        hash = fnv1a_step(hash, &last.to_bits().to_le_bytes());
    }
    hash
}

fn hash_u32_sample(mut hash: u64, data: &[u32]) -> u64 {
    hash = fnv1a_step(hash, &(data.len() as u64).to_le_bytes());
    let step = (data.len() / 64).max(1);
    let mut i = 0;
    while i < data.len() {
        hash = fnv1a_step(hash, &data[i].to_le_bytes());
        i += step;
    }
    if let Some(&last) = data.last() {
        hash = fnv1a_step(hash, &last.to_le_bytes());
    }
    hash
}

fn scene_fingerprint(scene: &GaussianSceneSoa) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    h = fnv1a_step(h, &(scene.count as u64).to_le_bytes());
    h = fnv1a_step(h, &(scene.positions.as_ptr() as usize as u64).to_le_bytes());
    h = fnv1a_step(
        h,
        &(scene.covariances_3d.as_ptr() as usize as u64).to_le_bytes(),
    );
    h = fnv1a_step(h, &(scene.opacities.as_ptr() as usize as u64).to_le_bytes());
    h = fnv1a_step(h, &(scene.sh_coeffs.as_ptr() as usize as u64).to_le_bytes());
    h = hash_f32_sample(h, &scene.positions);
    h = hash_f32_sample(h, &scene.covariances_3d);
    h = hash_f32_sample(h, &scene.opacities);
    hash_f32_sample(h, &scene.sh_coeffs)
}

fn read_single_u32(client: &Client, handle: Handle, label: &str) -> Result<u32, GpuError> {
    let bytes = client
        .read_one(handle)
        .map_err(|e| GpuError::Execution(format!("failed to read {label}: {e:?}")))?;
    if bytes.len() < 4 {
        return Err(GpuError::Execution(format!(
            "short readback for {label}: got {} bytes",
            bytes.len()
        )));
    }
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

/// Persistent GPU VRAM residency for rest-pose Gaussian attributes and flattened multi-cage topology.
pub struct GpuSceneState {
    scene_hash: Option<u64>,
    pub num_rest_gaussians: usize,
    rest_positions: Option<Handle>,
    rest_covariances: Option<Handle>,
    rest_opacities: Option<Handle>,
    rest_sh_coeffs: Option<Handle>,

    cage_topo_hash: Option<u64>,
    pub has_cages: bool,
    pub total_tets: usize,
    pub total_cage_vertices: usize,
    tet_elements: Option<Handle>,
    tet_inv_dm: Option<Handle>,
    bindings_tet: Option<Handle>,
    bindings_weights: Option<Handle>,
    cage_vertices: Option<(Handle, usize)>,

    dummy_u32: Option<Handle>,
    dummy_f32: Option<Handle>,

    last_cage_vertices: Vec<f32>,
    scratch_flat_vertices: Vec<f32>,
    last_deform_params: (u32, u32),
    deformed_valid: bool,
    pub deform_generation: u64,
}

impl Default for GpuSceneState {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuSceneState {
    /// Creates an empty persistent scene residency tracker.
    pub fn new() -> Self {
        Self {
            scene_hash: None,
            num_rest_gaussians: 0,
            rest_positions: None,
            rest_covariances: None,
            rest_opacities: None,
            rest_sh_coeffs: None,
            cage_topo_hash: None,
            has_cages: false,
            total_tets: 0,
            total_cage_vertices: 0,
            tet_elements: None,
            tet_inv_dm: None,
            bindings_tet: None,
            bindings_weights: None,
            cage_vertices: None,
            dummy_u32: None,
            dummy_f32: None,
            last_cage_vertices: Vec::new(),
            scratch_flat_vertices: Vec::new(),
            last_deform_params: (0, 0),
            deformed_valid: false,
            deform_generation: 0,
        }
    }

    /// Invalidates cached rest buffers, cage topology, and deformation outputs.
    pub fn invalidate(&mut self) {
        self.scene_hash = None;
        self.cage_topo_hash = None;
        self.last_cage_vertices.clear();
        self.deformed_valid = false;
        self.deform_generation = self.deform_generation.wrapping_add(1);
    }

    fn ensure_dummies(&mut self, client: &Client) -> (Handle, Handle) {
        let du = self
            .dummy_u32
            .get_or_insert_with(|| client.create_from_slice(bytemuck::cast_slice(&[0u32, 0u32])))
            .clone();
        let df = self
            .dummy_f32
            .get_or_insert_with(|| client.create_from_slice(bytemuck::cast_slice(&[0.0f32; 9])))
            .clone();
        (du, df)
    }

    /// Synchronizes rest scene and cage topology/vertices into VRAM, uploading only changed buffers.
    ///
    /// Returns `true` if cage vertices or deform parameters differ from the cached deformed frame.
    pub fn sync_scene_and_cages(
        &mut self,
        client: &Client,
        scene: &GaussianSceneSoa,
        cages: Option<&CageScene>,
        strain_split_threshold: f32,
        mip_filter_3d_var: f32,
    ) -> bool {
        self.ensure_dummies(client);

        let s_hash = scene_fingerprint(scene);
        if self.scene_hash != Some(s_hash) || self.num_rest_gaussians != scene.count {
            self.num_rest_gaussians = scene.count;
            if scene.count > 0 {
                self.rest_positions = Some(client.create_from_slice(scene.positions_bytes()));
                self.rest_covariances = Some(client.create_from_slice(scene.covariances_bytes()));
                self.rest_opacities = Some(client.create_from_slice(scene.opacities_bytes()));
                self.rest_sh_coeffs = Some(client.create_from_slice(scene.sh_coeffs_bytes()));
            } else {
                self.rest_positions = None;
                self.rest_covariances = None;
                self.rest_opacities = None;
                self.rest_sh_coeffs = None;
            }
            self.scene_hash = Some(s_hash);
            self.deformed_valid = false;
            self.cage_topo_hash = None;
            self.deform_generation = self.deform_generation.wrapping_add(1);
        }

        let active_cages =
            cages.filter(|c| !c.is_empty() && c.cages.iter().any(|cg| cg.mesh.num_tets() > 0));
        let Some(cage_scene) = active_cages else {
            let was_active = self.has_cages;
            self.has_cages = false;
            self.total_tets = 0;
            self.total_cage_vertices = 0;
            self.cage_topo_hash = Some(0);
            self.last_cage_vertices.clear();
            self.deformed_valid = !was_active;
            if was_active {
                self.deform_generation = self.deform_generation.wrapping_add(1);
            }
            return false;
        };

        let mut topo_hash = 0xcbf2_9ce4_8422_2325u64;
        topo_hash = fnv1a_step(topo_hash, &(cage_scene.cages.len() as u64).to_le_bytes());
        topo_hash = fnv1a_step(
            topo_hash,
            &(cage_scene.bindings.as_ptr() as usize as u64).to_le_bytes(),
        );
        topo_hash = fnv1a_step(topo_hash, &(cage_scene.bindings.len() as u64).to_le_bytes());
        for cage in &cage_scene.cages {
            topo_hash = fnv1a_step(topo_hash, &(cage.mesh.num_vertices() as u64).to_le_bytes());
            topo_hash = fnv1a_step(topo_hash, &(cage.mesh.num_tets() as u64).to_le_bytes());
            topo_hash = fnv1a_step(
                topo_hash,
                &(cage.mesh.elements.as_ptr() as usize as u64).to_le_bytes(),
            );
            topo_hash = fnv1a_step(
                topo_hash,
                &(cage.mesh.precomputed.as_ptr() as usize as u64).to_le_bytes(),
            );
        }

        if self.cage_topo_hash != Some(topo_hash) || !self.has_cages {
            let mut flat_elems: Vec<u32> = Vec::new();
            let mut flat_inv_dm: Vec<f32> = Vec::new();
            let mut vert_offset = 0u32;
            let mut tet_offsets: Vec<u32> = Vec::with_capacity(cage_scene.cages.len());

            for cage in &cage_scene.cages {
                let tet_off = (flat_inv_dm.len() / 9) as u32;
                tet_offsets.push(tet_off);
                for elem in &cage.mesh.elements {
                    for &v in &elem.indices {
                        flat_elems.push(v + vert_offset);
                    }
                }
                for pre in &cage.mesh.precomputed {
                    flat_inv_dm.extend_from_slice(&pre.inv_dm);
                }
                vert_offset += cage.mesh.num_vertices() as u32;
            }

            let n = scene.count;
            let mut bind_tets = vec![u32::MAX; n.max(1)];
            let mut bind_weights = vec![0.0f32; (n * 4).max(4)];
            for (i, slot) in bind_tets.iter_mut().enumerate().take(n) {
                if let Some(b) = cage_scene.bindings.get(i)
                    && b.is_bound()
                    && let Some(cg) = cage_scene.cages.get(b.cage as usize)
                    && (b.tet as usize) < cg.mesh.num_tets()
                {
                    *slot = tet_offsets[b.cage as usize] + b.tet;
                    let w_base = i * 4;
                    bind_weights[w_base..w_base + 4].copy_from_slice(&b.weights);
                }
            }

            let mut topo_check = hash_u32_sample(topo_hash, &flat_elems);
            topo_check = hash_f32_sample(topo_check, &flat_inv_dm);
            let _ = topo_check;

            self.total_tets = flat_inv_dm.len() / 9;
            self.total_cage_vertices = vert_offset as usize;
            self.tet_elements = Some(client.create_from_slice(bytemuck::cast_slice(&flat_elems)));
            self.tet_inv_dm = Some(client.create_from_slice(bytemuck::cast_slice(&flat_inv_dm)));
            self.bindings_tet = Some(client.create_from_slice(bytemuck::cast_slice(&bind_tets)));
            self.bindings_weights =
                Some(client.create_from_slice(bytemuck::cast_slice(&bind_weights)));
            self.has_cages = true;
            self.cage_topo_hash = Some(topo_hash);
            self.last_cage_vertices.clear();
            self.deformed_valid = false;
        }

        // Gather current cage vertices into scratch buffer without reallocating when steady
        self.scratch_flat_vertices.clear();
        self.scratch_flat_vertices
            .reserve(self.total_cage_vertices * 3);
        for cage in &cage_scene.cages {
            for v in cage.vertices() {
                self.scratch_flat_vertices.push(v[0]);
                self.scratch_flat_vertices.push(v[1]);
                self.scratch_flat_vertices.push(v[2]);
            }
        }

        let deform_params = (
            strain_split_threshold.to_bits(),
            mip_filter_3d_var.to_bits(),
        );
        let verts_changed = self.scratch_flat_vertices != self.last_cage_vertices;
        let params_changed = deform_params != self.last_deform_params;

        if verts_changed {
            let needed = self.scratch_flat_vertices.len().max(3);
            let bytes = bytemuck::cast_slice(&self.scratch_flat_vertices).to_vec();
            match &mut self.cage_vertices {
                Some((handle, cap)) if *cap >= needed => {
                    client.write(handle, Bytes::from_bytes_vec(bytes));
                }
                _ => {
                    let handle = client.create_from_slice(&bytes);
                    self.cage_vertices = Some((handle, needed));
                }
            }
            self.last_cage_vertices.clear();
            self.last_cage_vertices
                .extend_from_slice(&self.scratch_flat_vertices);
            self.deformed_valid = false;
        }

        if params_changed {
            self.last_deform_params = deform_params;
            self.deformed_valid = false;
        }

        !self.deformed_valid
    }
}

/// Pre-allocated GPU intermediate buffers grown exponentially (`1.5x`) only when scene
/// splat count, tet count, or viewport dimensions increase.
pub struct GpuFrameArena {
    tet_jacobians: Option<(Handle, usize)>,
    tet_rotations: Option<(Handle, usize)>,
    tet_max_stretch: Option<(Handle, usize)>,
    deformed_positions: Option<(Handle, usize)>,
    deformed_covariances: Option<(Handle, usize)>,
    deformed_opacities: Option<(Handle, usize)>,
    split_meta: Option<(Handle, usize)>,
    split_counter: Option<Handle>,

    camera_uniforms: Option<Handle>,
    splat_attrs: Option<(Handle, usize)>,
    extents_packed: Option<(Handle, usize)>,
    surviving_ids: Option<(Handle, usize)>,
    surviving_counter: Option<Handle>,

    tiles_touched: Option<(Handle, usize)>,
    tile_offsets: Option<(Handle, usize)>,
    tile_ranges: Option<(Handle, usize)>,
    keys: Option<(Handle, usize)>,
    depths_u32: Option<(Handle, usize)>,
    splat_ids: Option<(Handle, usize)>,

    output_rgba8: Option<(Handle, usize)>,

    scan_scratch: ScanScratch,
    radix_scratch: RadixSortScratch,
}

impl Default for GpuFrameArena {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuFrameArena {
    /// Creates an empty GPU intermediate arena.
    pub fn new() -> Self {
        Self {
            tet_jacobians: None,
            tet_rotations: None,
            tet_max_stretch: None,
            deformed_positions: None,
            deformed_covariances: None,
            deformed_opacities: None,
            split_meta: None,
            split_counter: None,
            camera_uniforms: None,
            splat_attrs: None,
            extents_packed: None,
            surviving_ids: None,
            surviving_counter: None,
            tiles_touched: None,
            tile_offsets: None,
            tile_ranges: None,
            keys: None,
            depths_u32: None,
            splat_ids: None,
            output_rgba8: None,
            scan_scratch: ScanScratch::new(),
            radix_scratch: RadixSortScratch::new(),
        }
    }

    fn ensure_words(slot: &mut Option<(Handle, usize)>, client: &Client, needed: usize) -> Handle {
        let req = needed.max(1);
        match slot {
            Some((handle, cap)) if *cap >= req => handle.clone(),
            Some((_, cap)) => {
                let new_cap = calculate_exponential_growth(*cap, req);
                let handle = client.empty(new_cap * 4);
                *slot = Some((handle.clone(), new_cap));
                handle
            }
            None => {
                let new_cap = req.max(256);
                let handle = client.empty(new_cap * 4);
                *slot = Some((handle.clone(), new_cap));
                handle
            }
        }
    }

    fn ensure_scalar_zero(slot: &mut Option<Handle>, client: &Client) -> Handle {
        if let Some(h) = slot {
            client.write(h, Bytes::from_bytes_vec(vec![0u8; 4]));
            h.clone()
        } else {
            let h = client.create_from_slice(&[0u8; 4]);
            *slot = Some(h.clone());
            h
        }
    }

    fn ensure_scalar_existing(slot: &mut Option<Handle>, client: &Client) -> Handle {
        if let Some(h) = slot {
            h.clone()
        } else {
            let h = client.create_from_slice(&[0u8; 4]);
            *slot = Some(h.clone());
            h
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Stage3CacheKey {
    deform_generation: u64,
    view_bits: [u32; 16],
    focal_x_bits: u32,
    focal_y_bits: u32,
    principal_x_bits: u32,
    principal_y_bits: u32,
    width: u32,
    height: u32,
    near_bits: u32,
    mip_bits: u32,
    min_opacity_bits: u32,
    min_radius_bits: u32,
}

impl Stage3CacheKey {
    #[allow(clippy::too_many_arguments)]
    fn from_params(
        deform_generation: u64,
        camera: &CameraUniforms,
        width: u32,
        height: u32,
        mip_var: f32,
        min_opacity: f32,
        min_radius_px: f32,
    ) -> Self {
        let mut view_bits = [0u32; 16];
        for (i, &v) in camera.view_matrix.iter().enumerate() {
            view_bits[i] = v.to_bits();
        }
        Self {
            deform_generation,
            view_bits,
            focal_x_bits: camera.focal_x.to_bits(),
            focal_y_bits: camera.focal_y.to_bits(),
            principal_x_bits: camera.principal_x.to_bits(),
            principal_y_bits: camera.principal_y.to_bits(),
            width,
            height,
            near_bits: camera.near_plane.to_bits(),
            mip_bits: mip_var.to_bits(),
            min_opacity_bits: min_opacity.to_bits(),
            min_radius_bits: min_radius_px.to_bits(),
        }
    }
}

struct CachedStage3Handles {
    key: Stage3CacheKey,
    sorted_splat_ids: Handle,
    sorted_len: usize,
    tile_ranges: Handle,
    active_gaussians: usize,
    surviving_gaussians: usize,
    total_tile_entries: usize,
}

/// Per-stage GPU timings and counts produced by [`GpuPipelineState::execute_frame_gpu`].
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuStageTimings {
    /// Stage 1 volumetric cage deformation & splitting time (ms); `0.0` when stationary.
    pub stage1_deform_ms: f64,
    /// Stage 2 EWA projection, inverse-ray SH evaluation & compaction time (ms).
    pub stage2_project_ms: f64,
    /// Stage 3 prefix-scan, radix sort & tile range build time (ms); `0.0` when cached.
    pub stage3_bin_sort_ms: f64,
    /// Stage 4 cooperative tile compositor time (ms).
    pub stage4_composite_ms: f64,
    /// True when Stage 1 was skipped due to stationary cages.
    pub stage1_skipped: bool,
    /// True when Stage 3 sorted lists & tile ranges were reused from the prior frame.
    pub stage3_cached: bool,
    /// True when hardware `TIMESTAMP_QUERY` counters supplied stage timings.
    pub used_hardware_timestamps: bool,
    /// Total active Gaussians after splitting (`rest + split_children`).
    pub active_gaussians: usize,
    /// Gaussians surviving frustum/opacity/radius culling in Stage 2.
    pub surviving_gaussians: usize,
    /// Total tile-splat entries emitted and sorted in Stage 3.
    pub total_tile_entries: usize,
}

/// Persistent GPU-resident 4-stage pipeline orchestrating Stages 1–4 without per-frame
/// re-uploads or host framebuffer readbacks.
pub struct GpuPipelineState {
    pub scene: GpuSceneState,
    pub arena: GpuFrameArena,
    stage3_cache: Option<CachedStage3Handles>,
    last_width: u32,
    last_height: u32,
}

impl Default for GpuPipelineState {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuPipelineState {
    /// Creates an empty persistent GPU pipeline state.
    pub fn new() -> Self {
        Self {
            scene: GpuSceneState::new(),
            arena: GpuFrameArena::new(),
            stage3_cache: None,
            last_width: 0,
            last_height: 0,
        }
    }

    /// Invalidates all persistent scene/cage buffers and cached Stage 3 outputs.
    pub fn invalidate(&mut self) {
        self.scene.invalidate();
        self.stage3_cache = None;
    }

    /// Clears only the Stage 3 sort cache (e.g., on viewport resize).
    pub fn invalidate_stage3(&mut self) {
        self.stage3_cache = None;
    }

    fn measure_stage<F>(
        client: &Client,
        profile_stages: bool,
        allow_hw_timestamps: bool,
        used_hw: &mut bool,
        f: F,
    ) -> Result<f64, GpuError>
    where
        F: FnOnce() -> Result<(), GpuError>,
    {
        if !profile_stages {
            f()?;
            return Ok(0.0);
        }

        if allow_hw_timestamps && let Ok(win) = client.profile_start() {
            f()?;
            match client.profile_end(win) {
                Ok(dur) => {
                    if let Some(ticks) = pollster::block_on(dur.resolve()) {
                        *used_hw = true;
                        return Ok(ticks.duration().as_secs_f64() * 1000.0);
                    }
                    return Ok(0.0);
                }
                Err(ProfileError::NotMeasured { .. }) | Err(_) => return Ok(0.0),
            }
        }

        let start = Instant::now();
        f()?;
        let _ = client.flush();
        Ok(start.elapsed().as_secs_f64() * 1000.0)
    }

    /// Executes all active GPU stages on persistent VRAM buffers, writing packed RGBA8
    /// (`padded_row_u32` stride) directly into `arena.output_rgba8`.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_frame_gpu(
        &mut self,
        compute: &GpuCompute,
        scene: &GaussianSceneSoa,
        cages: Option<&CageScene>,
        camera: &CameraUniforms,
        compositor: &CompositorUniforms,
        strain_split_threshold: f32,
        mip_filter_3d_var: f32,
        min_opacity: f32,
        min_radius_px: f32,
        profile_stages: bool,
        force_stage3: bool,
    ) -> Result<GpuStageTimings, GpuError> {
        let client = compute.client();
        let width = compositor.screen_width.max(1);
        let height = compositor.screen_height.max(1);
        let tiles_x = width.div_ceil(16);
        let tiles_y = height.div_ceil(16);
        let total_tiles = (tiles_x * tiles_y) as usize;
        let padded_row_u32 = padded_rgba8_row_u32(width);

        if width != self.last_width || height != self.last_height {
            self.stage3_cache = None;
            self.last_width = width;
            self.last_height = height;
        }

        let out_words = (padded_row_u32 as usize) * (height as usize);
        let out_handle =
            GpuFrameArena::ensure_words(&mut self.arena.output_rgba8, client, out_words);

        let need_deform = self.scene.sync_scene_and_cages(
            client,
            scene,
            cages,
            strain_split_threshold,
            mip_filter_3d_var,
        );
        let (dummy_u32, dummy_f32) = self.scene.ensure_dummies(client);

        let allow_hw = compute.supports_timestamps();
        let use_subgroups = compute.supports_subgroups();
        let mut timings = GpuStageTimings::default();

        if scene.count == 0 {
            let tile_ranges =
                GpuFrameArena::ensure_words(&mut self.arena.tile_ranges, client, total_tiles * 2);
            let dummy_attrs = GpuFrameArena::ensure_words(&mut self.arena.splat_attrs, client, 10);
            let dummy_ids = GpuFrameArena::ensure_words(&mut self.arena.splat_ids, client, 1);

            unsafe {
                clear_tile_ranges_kernel::launch(
                    client,
                    dispatch_cube_count((total_tiles as u32).div_ceil(COMPUTE_WG)),
                    CubeDim::new_1d(COMPUTE_WG),
                    BufferArg::from_raw_parts(tile_ranges.clone(), total_tiles * 2),
                    total_tiles as u32,
                );
            }
            timings.stage4_composite_ms = Self::measure_stage(
                client,
                profile_stages,
                allow_hw,
                &mut timings.used_hardware_timestamps,
                || {
                    unsafe {
                        tile_compositor_v2_kernel::launch(
                            client,
                            CubeCount::Static(tiles_x, tiles_y, 1),
                            CubeDim::new_1d(COMPUTE_WG),
                            BufferArg::from_raw_parts(dummy_attrs, 10),
                            BufferArg::from_raw_parts(dummy_ids, 1),
                            BufferArg::from_raw_parts(tile_ranges, total_tiles * 2),
                            BufferArg::from_raw_parts(out_handle, out_words),
                            width,
                            height,
                            tiles_x,
                            padded_row_u32,
                            compositor.background_color[0],
                            compositor.background_color[1],
                            compositor.background_color[2],
                            compositor.background_color[3],
                            use_subgroups,
                        );
                    }
                    Ok(())
                },
            )?;
            timings.stage1_skipped = true;
            return Ok(timings);
        }

        let num_rest = self.scene.num_rest_gaussians;
        let has_cages = self.scene.has_cages;
        let split_possible = has_cages && strain_split_threshold > 0.0;
        let max_gaussians = if split_possible {
            num_rest * 2
        } else {
            num_rest
        };

        // Stage 1: Volumetric cage deformation & splitting (dirty-flag skipped when static)
        let (pos_handle, cov_handle, opa_handle, tet_rot_handle, split_meta_handle, split_cnt_h) =
            if has_cages {
                let num_tets = self.scene.total_tets;
                let jac_h = GpuFrameArena::ensure_words(
                    &mut self.arena.tet_jacobians,
                    client,
                    num_tets * 9,
                );
                let rot_h = GpuFrameArena::ensure_words(
                    &mut self.arena.tet_rotations,
                    client,
                    num_tets * 9,
                );
                let stretch_h =
                    GpuFrameArena::ensure_words(&mut self.arena.tet_max_stretch, client, num_tets);
                let def_pos_h = GpuFrameArena::ensure_words(
                    &mut self.arena.deformed_positions,
                    client,
                    max_gaussians * 3,
                );
                let def_cov_h = GpuFrameArena::ensure_words(
                    &mut self.arena.deformed_covariances,
                    client,
                    max_gaussians * 6,
                );
                let def_opa_h = GpuFrameArena::ensure_words(
                    &mut self.arena.deformed_opacities,
                    client,
                    max_gaussians,
                );
                let s_meta_h = if split_possible {
                    GpuFrameArena::ensure_words(&mut self.arena.split_meta, client, num_rest * 2)
                } else {
                    dummy_u32.clone()
                };

                if need_deform {
                    let s_cnt_h =
                        GpuFrameArena::ensure_scalar_zero(&mut self.arena.split_counter, client);
                    let cage_verts_h = self
                        .scene
                        .cage_vertices
                        .as_ref()
                        .expect("cage vertices uploaded")
                        .0
                        .clone();
                    let tet_elems_h = self
                        .scene
                        .tet_elements
                        .as_ref()
                        .expect("tet elements uploaded")
                        .clone();
                    let tet_inv_dm_h = self
                        .scene
                        .tet_inv_dm
                        .as_ref()
                        .expect("tet inv_dm uploaded")
                        .clone();
                    let bind_tet_h = self
                        .scene
                        .bindings_tet
                        .as_ref()
                        .expect("bindings tet uploaded")
                        .clone();
                    let bind_w_h = self
                        .scene
                        .bindings_weights
                        .as_ref()
                        .expect("bindings weights uploaded")
                        .clone();
                    let rest_p_h = self.scene.rest_positions.as_ref().unwrap().clone();
                    let rest_c_h = self.scene.rest_covariances.as_ref().unwrap().clone();
                    let rest_o_h = self.scene.rest_opacities.as_ref().unwrap().clone();

                    timings.stage1_deform_ms = Self::measure_stage(
                        client,
                        profile_stages,
                        allow_hw,
                        &mut timings.used_hardware_timestamps,
                        || {
                            unsafe {
                                compute_tet_transforms_kernel::launch(
                                    client,
                                    dispatch_cube_count((num_tets as u32).div_ceil(64).max(1)),
                                    CubeDim::new_1d(64.min(num_tets.max(1) as u32)),
                                    BufferArg::from_raw_parts(
                                        cage_verts_h.clone(),
                                        self.scene.total_cage_vertices * 3,
                                    ),
                                    BufferArg::from_raw_parts(tet_elems_h.clone(), num_tets * 4),
                                    BufferArg::from_raw_parts(tet_inv_dm_h, num_tets * 9),
                                    BufferArg::from_raw_parts(jac_h.clone(), num_tets * 9),
                                    BufferArg::from_raw_parts(rot_h.clone(), num_tets * 9),
                                    BufferArg::from_raw_parts(stretch_h.clone(), num_tets),
                                    num_tets as u32,
                                );

                                deform_and_split_compact_kernel::launch(
                                    client,
                                    dispatch_cube_count(
                                        (num_rest as u32).div_ceil(COMPUTE_WG).max(1),
                                    ),
                                    CubeDim::new_1d(COMPUTE_WG),
                                    BufferArg::from_raw_parts(
                                        cage_verts_h,
                                        self.scene.total_cage_vertices * 3,
                                    ),
                                    BufferArg::from_raw_parts(tet_elems_h, num_tets * 4),
                                    BufferArg::from_raw_parts(jac_h, num_tets * 9),
                                    BufferArg::from_raw_parts(stretch_h, num_tets),
                                    BufferArg::from_raw_parts(bind_tet_h, num_rest),
                                    BufferArg::from_raw_parts(bind_w_h, num_rest * 4),
                                    BufferArg::from_raw_parts(rest_p_h, num_rest * 3),
                                    BufferArg::from_raw_parts(rest_c_h, num_rest * 6),
                                    BufferArg::from_raw_parts(rest_o_h, num_rest),
                                    BufferArg::from_raw_parts(def_pos_h.clone(), max_gaussians * 3),
                                    BufferArg::from_raw_parts(def_cov_h.clone(), max_gaussians * 6),
                                    BufferArg::from_raw_parts(def_opa_h.clone(), max_gaussians),
                                    BufferArg::from_raw_parts(
                                        s_meta_h.clone(),
                                        if split_possible { num_rest * 2 } else { 2 },
                                    ),
                                    BufferArg::from_raw_parts(s_cnt_h.clone(), 1),
                                    strain_split_threshold,
                                    mip_filter_3d_var,
                                    num_rest as u32,
                                    num_tets as u32,
                                );
                            }
                            Ok(())
                        },
                    )?;
                    self.scene.deformed_valid = true;
                    self.scene.deform_generation = self.scene.deform_generation.wrapping_add(1);
                    timings.stage1_skipped = false;
                    (def_pos_h, def_cov_h, def_opa_h, rot_h, s_meta_h, s_cnt_h)
                } else {
                    timings.stage1_skipped = true;
                    let s_cnt_h = GpuFrameArena::ensure_scalar_existing(
                        &mut self.arena.split_counter,
                        client,
                    );
                    (def_pos_h, def_cov_h, def_opa_h, rot_h, s_meta_h, s_cnt_h)
                }
            } else {
                timings.stage1_skipped = true;
                let s_cnt_h =
                    GpuFrameArena::ensure_scalar_zero(&mut self.arena.split_counter, client);
                (
                    self.scene.rest_positions.as_ref().unwrap().clone(),
                    self.scene.rest_covariances.as_ref().unwrap().clone(),
                    self.scene.rest_opacities.as_ref().unwrap().clone(),
                    dummy_f32.clone(),
                    dummy_u32.clone(),
                    s_cnt_h,
                )
            };

        // Upload 96-byte camera uniforms in place
        let cam_bytes = camera.as_bytes().to_vec();
        let cam_h = match &self.arena.camera_uniforms {
            Some(h) => {
                client.write(h, Bytes::from_bytes_vec(cam_bytes));
                h.clone()
            }
            None => {
                let h = client.create_from_slice(&cam_bytes);
                self.arena.camera_uniforms = Some(h.clone());
                h
            }
        };

        let rest_sh_h = self.scene.rest_sh_coeffs.as_ref().unwrap().clone();
        let bind_tet_stage2 = if has_cages {
            self.scene.bindings_tet.as_ref().unwrap().clone()
        } else {
            dummy_u32.clone()
        };
        let stage2_mip = if has_cages { 0.0 } else { mip_filter_3d_var };

        let splat_attrs =
            GpuFrameArena::ensure_words(&mut self.arena.splat_attrs, client, max_gaussians * 10);
        let extents_packed =
            GpuFrameArena::ensure_words(&mut self.arena.extents_packed, client, max_gaussians);
        let surviving_ids =
            GpuFrameArena::ensure_words(&mut self.arena.surviving_ids, client, max_gaussians);
        let surviving_counter =
            GpuFrameArena::ensure_scalar_zero(&mut self.arena.surviving_counter, client);

        let split_meta_len = if split_possible { num_rest * 2 } else { 2 };
        let rot_len = if has_cages {
            self.scene.total_tets * 9
        } else {
            9
        };
        let bind_len = if has_cages { num_rest } else { 1 };
        let has_deform_u32 = u32::from(has_cages);

        // Stage 2: Projection, tight extents, inverse-ray SH evaluation, and atomic compaction
        timings.stage2_project_ms = Self::measure_stage(
            client,
            profile_stages,
            allow_hw,
            &mut timings.used_hardware_timestamps,
            || {
                unsafe {
                    project_gaussians_v2_kernel::launch(
                        client,
                        dispatch_cube_count((max_gaussians as u32).div_ceil(COMPUTE_WG).max(1)),
                        CubeDim::new_1d(COMPUTE_WG),
                        BufferArg::from_raw_parts(pos_handle, max_gaussians * 3),
                        BufferArg::from_raw_parts(cov_handle, max_gaussians * 6),
                        BufferArg::from_raw_parts(opa_handle, max_gaussians),
                        BufferArg::from_raw_parts(rest_sh_h, num_rest * 48),
                        BufferArg::from_raw_parts(bind_tet_stage2, bind_len),
                        BufferArg::from_raw_parts(tet_rot_handle, rot_len),
                        BufferArg::from_raw_parts(split_meta_handle, split_meta_len),
                        BufferArg::from_raw_parts(split_cnt_h.clone(), 1),
                        BufferArg::from_raw_parts(cam_h, 24),
                        BufferArg::from_raw_parts(splat_attrs.clone(), max_gaussians * 10),
                        BufferArg::from_raw_parts(extents_packed.clone(), max_gaussians),
                        BufferArg::from_raw_parts(surviving_ids, max_gaussians),
                        BufferArg::from_raw_parts(surviving_counter.clone(), 1),
                        compositor.camera_position[0],
                        compositor.camera_position[1],
                        compositor.camera_position[2],
                        stage2_mip,
                        min_opacity,
                        min_radius_px,
                        compositor.sh_degree.min(3),
                        has_deform_u32,
                        self.scene.total_tets as u32,
                        num_rest as u32,
                    );
                }
                Ok(())
            },
        )?;

        // Stage 3: Cache lookup or GPU Prefix Scan + 4-bit LSD Radix Sort + Tile Ranges
        let cache_key = Stage3CacheKey::from_params(
            self.scene.deform_generation,
            camera,
            width,
            height,
            mip_filter_3d_var,
            min_opacity,
            min_radius_px,
        );

        let (sorted_splat_ids, sorted_len, tile_ranges) = if !force_stage3
            && self
                .stage3_cache
                .as_ref()
                .is_some_and(|c| c.key == cache_key)
        {
            let cached = self.stage3_cache.as_ref().unwrap();
            timings.stage3_cached = true;
            timings.active_gaussians = cached.active_gaussians;
            timings.surviving_gaussians = cached.surviving_gaussians;
            timings.total_tile_entries = cached.total_tile_entries;
            (
                cached.sorted_splat_ids.clone(),
                cached.sorted_len,
                cached.tile_ranges.clone(),
            )
        } else {
            let tiles_touched =
                GpuFrameArena::ensure_words(&mut self.arena.tiles_touched, client, max_gaussians);
            let tile_offsets =
                GpuFrameArena::ensure_words(&mut self.arena.tile_offsets, client, max_gaussians);
            let tile_ranges =
                GpuFrameArena::ensure_words(&mut self.arena.tile_ranges, client, total_tiles * 2);

            let mut result_ids = GpuFrameArena::ensure_words(&mut self.arena.splat_ids, client, 1);
            let mut result_len = 1usize;
            let mut total_entries = 0usize;
            let mut surviving = 0usize;
            let mut active = num_rest;

            timings.stage3_bin_sort_ms = Self::measure_stage(
                client,
                profile_stages,
                allow_hw,
                &mut timings.used_hardware_timestamps,
                || {
                    unsafe {
                        count_tiles_kernel::launch(
                            client,
                            dispatch_cube_count((max_gaussians as u32).div_ceil(COMPUTE_WG).max(1)),
                            CubeDim::new_1d(COMPUTE_WG),
                            BufferArg::from_raw_parts(splat_attrs.clone(), max_gaussians * 10),
                            BufferArg::from_raw_parts(extents_packed.clone(), max_gaussians),
                            BufferArg::from_raw_parts(split_cnt_h.clone(), 1),
                            BufferArg::from_raw_parts(tiles_touched.clone(), max_gaussians),
                            width,
                            height,
                            tiles_x,
                            tiles_y,
                            num_rest as u32,
                            max_gaussians as u32,
                        );
                    }

                    let entries_u32 = gpu_exclusive_scan_total(
                        client,
                        &tiles_touched,
                        &tile_offsets,
                        max_gaussians,
                        &mut self.arena.scan_scratch,
                    )?;
                    total_entries = entries_u32 as usize;

                    surviving =
                        read_single_u32(client, surviving_counter.clone(), "surviving_counter")?
                            as usize;
                    if split_possible {
                        let splits = read_single_u32(client, split_cnt_h.clone(), "split_counter")?;
                        active = num_rest + splits as usize;
                    }

                    unsafe {
                        clear_tile_ranges_kernel::launch(
                            client,
                            dispatch_cube_count((total_tiles as u32).div_ceil(COMPUTE_WG).max(1)),
                            CubeDim::new_1d(COMPUTE_WG),
                            BufferArg::from_raw_parts(tile_ranges.clone(), total_tiles * 2),
                            total_tiles as u32,
                        );
                    }

                    if total_entries > 0 {
                        let splat_wg =
                            dispatch_cube_count((active as u32).div_ceil(COMPUTE_WG).max(1));
                        let entry_wg =
                            dispatch_cube_count((total_entries as u32).div_ceil(COMPUTE_WG).max(1));

                        let keys_h = GpuFrameArena::ensure_words(
                            &mut self.arena.keys,
                            client,
                            total_entries,
                        );
                        let ids_h = GpuFrameArena::ensure_words(
                            &mut self.arena.splat_ids,
                            client,
                            total_entries,
                        );

                        if total_tiles <= 65_535 {
                            let (tile_bits, depth_bits) = compute_packed_key_bits(total_tiles);
                            let base_depth_bits =
                                camera.near_plane.max(1e-6).to_bits() & 0x7F80_0000;
                            let sort_bits = (tile_bits + depth_bits).min(32);

                            unsafe {
                                emit_tile_keys_kernel::launch(
                                    client,
                                    splat_wg,
                                    CubeDim::new_1d(COMPUTE_WG),
                                    BufferArg::from_raw_parts(
                                        splat_attrs.clone(),
                                        max_gaussians * 10,
                                    ),
                                    BufferArg::from_raw_parts(
                                        extents_packed.clone(),
                                        max_gaussians,
                                    ),
                                    BufferArg::from_raw_parts(tile_offsets, max_gaussians),
                                    BufferArg::from_raw_parts(split_cnt_h, 1),
                                    BufferArg::from_raw_parts(keys_h.clone(), total_entries),
                                    BufferArg::from_raw_parts(ids_h.clone(), total_entries),
                                    width,
                                    height,
                                    tiles_x,
                                    tiles_y,
                                    num_rest as u32,
                                    base_depth_bits,
                                    depth_bits,
                                );
                            }

                            let in_alt = gpu_radix_sort_u32_pairs(
                                client,
                                &keys_h,
                                &ids_h,
                                total_entries,
                                sort_bits,
                                use_subgroups,
                                &mut self.arena.radix_scratch,
                            )?;
                            let (sorted_keys_h, sorted_ids_h) = self
                                .arena
                                .radix_scratch
                                .u32_sorted_outputs(&keys_h, &ids_h, in_alt);

                            unsafe {
                                identify_tile_ranges_kernel::launch(
                                    client,
                                    entry_wg,
                                    CubeDim::new_1d(COMPUTE_WG),
                                    BufferArg::from_raw_parts(sorted_keys_h.clone(), total_entries),
                                    BufferArg::from_raw_parts(tile_ranges.clone(), total_tiles * 2),
                                    total_entries as u32,
                                    total_tiles as u32,
                                    depth_bits,
                                );
                            }
                            result_ids = sorted_ids_h.clone();
                            result_len = total_entries;
                        } else {
                            let depths_h = GpuFrameArena::ensure_words(
                                &mut self.arena.depths_u32,
                                client,
                                total_entries,
                            );
                            unsafe {
                                emit_tile_keys_u64_kernel::launch(
                                    client,
                                    splat_wg,
                                    CubeDim::new_1d(COMPUTE_WG),
                                    BufferArg::from_raw_parts(
                                        splat_attrs.clone(),
                                        max_gaussians * 10,
                                    ),
                                    BufferArg::from_raw_parts(
                                        extents_packed.clone(),
                                        max_gaussians,
                                    ),
                                    BufferArg::from_raw_parts(tile_offsets, max_gaussians),
                                    BufferArg::from_raw_parts(split_cnt_h, 1),
                                    BufferArg::from_raw_parts(keys_h.clone(), total_entries),
                                    BufferArg::from_raw_parts(depths_h.clone(), total_entries),
                                    BufferArg::from_raw_parts(ids_h.clone(), total_entries),
                                    width,
                                    height,
                                    tiles_x,
                                    tiles_y,
                                    num_rest as u32,
                                );
                            }

                            let tile_bits = 32 - ((total_tiles as u32) - 1).leading_zeros();
                            let in_alt = gpu_radix_sort_u64_pairs(
                                client,
                                &keys_h,
                                &depths_h,
                                &ids_h,
                                total_entries,
                                tile_bits,
                                use_subgroups,
                                &mut self.arena.radix_scratch,
                            )?;
                            let (sorted_tiles_h, sorted_ids_h) =
                                self.arena.radix_scratch.u64_sorted_outputs(in_alt);

                            unsafe {
                                identify_tile_ranges_u32_tile_kernel::launch(
                                    client,
                                    entry_wg,
                                    CubeDim::new_1d(COMPUTE_WG),
                                    BufferArg::from_raw_parts(
                                        sorted_tiles_h.clone(),
                                        total_entries,
                                    ),
                                    BufferArg::from_raw_parts(tile_ranges.clone(), total_tiles * 2),
                                    total_entries as u32,
                                    total_tiles as u32,
                                );
                            }
                            result_ids = sorted_ids_h.clone();
                            result_len = total_entries;
                        }
                    }

                    Ok(())
                },
            )?;

            timings.active_gaussians = active;
            timings.surviving_gaussians = surviving;
            timings.total_tile_entries = total_entries;

            self.stage3_cache = Some(CachedStage3Handles {
                key: cache_key,
                sorted_splat_ids: result_ids.clone(),
                sorted_len: result_len,
                tile_ranges: tile_ranges.clone(),
                active_gaussians: active,
                surviving_gaussians: surviving,
                total_tile_entries: total_entries,
            });

            (result_ids, result_len, tile_ranges)
        };

        // Stage 4: Cooperative shared-memory tile compositor -> packed RGBA8 u32 buffer
        timings.stage4_composite_ms = Self::measure_stage(
            client,
            profile_stages,
            allow_hw,
            &mut timings.used_hardware_timestamps,
            || {
                unsafe {
                    tile_compositor_v2_kernel::launch(
                        client,
                        CubeCount::Static(tiles_x, tiles_y, 1),
                        CubeDim::new_1d(COMPUTE_WG),
                        BufferArg::from_raw_parts(splat_attrs, max_gaussians * 10),
                        BufferArg::from_raw_parts(sorted_splat_ids, sorted_len),
                        BufferArg::from_raw_parts(tile_ranges, total_tiles * 2),
                        BufferArg::from_raw_parts(out_handle, out_words),
                        width,
                        height,
                        tiles_x,
                        padded_row_u32,
                        compositor.background_color[0],
                        compositor.background_color[1],
                        compositor.background_color[2],
                        compositor.background_color[3],
                        use_subgroups,
                    );
                }
                Ok(())
            },
        )?;

        Ok(timings)
    }

    /// Encodes a zero-copy GPU-to-GPU buffer copy from `arena.output_rgba8` into
    /// a `wgpu::Texture` (`Rgba8UnormSrgb` with `COPY_DST`) sharing the same WGPU queue.
    pub fn copy_framebuffer_to_texture(
        &self,
        compute: &GpuCompute,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
        width: u32,
        height: u32,
    ) -> Result<(), GpuError> {
        let Some((handle, _)) = &self.arena.output_rgba8 else {
            return Err(GpuError::Execution(
                "framebuffer handle missing before zero-copy blit".to_string(),
            ));
        };
        let client = compute.client();
        client
            .flush()
            .map_err(|e| GpuError::Execution(format!("client.flush failed: {e:?}")))?;
        let managed = client
            .get_resource::<WgpuServer<AutoCompiler>>(handle.clone())
            .map_err(|e| GpuError::Execution(format!("client.get_resource failed: {e:?}")))?;
        let res = managed.resource();
        let padded_bytes_per_row = padded_rgba8_row_u32(width) * 4;

        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &res.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: res.offset,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        Ok(())
    }

    /// Reads back the packed RGBA8 GPU framebuffer into a tight `[width * height * 4]`
    /// byte buffer without allocating a float shadow copy.
    pub fn readback_framebuffer_rgba8(
        &self,
        compute: &GpuCompute,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, GpuError> {
        let Some((handle, _)) = &self.arena.output_rgba8 else {
            return Err(GpuError::Execution(
                "framebuffer handle missing before readback".to_string(),
            ));
        };
        let padded_row_bytes = (padded_rgba8_row_u32(width) as usize) * 4;
        let tight_row_bytes = (width as usize) * 4;
        let expected_bytes = padded_row_bytes * (height as usize);
        let bytes = compute
            .client()
            .read_one(handle.clone())
            .map_err(|e| GpuError::Execution(format!("framebuffer readback failed: {e:?}")))?;
        if bytes.len() < expected_bytes {
            return Err(GpuError::Execution(format!(
                "short framebuffer readback: got {} bytes, expected {}",
                bytes.len(),
                expected_bytes
            )));
        }

        let mut rgba8 = Vec::with_capacity(tight_row_bytes * (height as usize));
        for row in 0..(height as usize) {
            let start = row * padded_row_bytes;
            rgba8.extend_from_slice(&bytes[start..start + tight_row_bytes]);
        }
        Ok(rgba8)
    }

    /// Reads back the packed RGBA8 GPU framebuffer for headless PNG export or parity tests.
    pub fn readback_framebuffer(
        &self,
        compute: &GpuCompute,
        width: u32,
        height: u32,
    ) -> Result<RasterizedImage, GpuError> {
        let rgba8 = self.readback_framebuffer_rgba8(compute, width, height)?;
        let rgba_f32: Vec<f32> = rgba8.iter().map(|&b| b as f32 / 255.0).collect();

        Ok(RasterizedImage {
            width,
            height,
            rgba_f32,
            rgba8,
        })
    }
}
