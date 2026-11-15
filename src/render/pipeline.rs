use crate::deformation::{CageScene, DeformationUniforms, GaussianBinding, deform_cage_scene_cpu};
use crate::render::blit::{BlitPipeline, OFFSCREEN_FORMAT};
use crate::render::camera::CameraUniforms;
use crate::render::gpu::{GpuCompute, GpuError};
use crate::render::gpu_bridge::{
    DeformedGeometry, composite_gpu, deform_geometry_gpu, deform_geometry_with_splitting_gpu,
    project_gaussians_cull_gpu, project_gaussians_gpu,
};
use crate::render::gpu_pool::GpuBufferPool;
use crate::render::rasterizer::sh_rotation::ShRotator;
use crate::render::rasterizer::{CompositorUniforms, RasterizedImage, rasterize_scene_cpu};
use crate::render::tiles::TileBinningOutputs;
use crate::render::wireframe::{CageGuide, WireframeRenderer};
use crate::render::{project_scene_cpu, project_scene_cull_cpu};
use crate::scene::GaussianSceneSoa;

/// Execution backend selector between the CPU reference and native CubeCL GPU kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ExecutionBackend {
    #[default]
    CpuReference,
    CubeClGpu,
}

/// Telemetry metrics produced by a rendered frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameStats {
    pub total_gaussians: usize,
    pub culled_gaussians: usize,
    pub rendered_gaussians: usize,
    pub active_tiles: usize,
    /// True when the previous frame's tile binning was reused (temporal hysteresis).
    pub sort_reused: bool,
}

/// A rendered frame: telemetry plus the composited image.
#[derive(Clone, Debug)]
pub struct FrameResult {
    pub stats: FrameStats,
    pub image: RasterizedImage,
}

/// Camera matrix difference below which tile binning is reused.
pub const CAMERA_MOTION_EPSILON: f32 = 1e-5;

/// Maximum cage vertex displacement below which tile binning is reused.
pub const CAGE_MOTION_EPSILON: f32 = 1e-5;

fn max_abs_difference(a: &[f32; 16], b: &[f32; 16]) -> f32 {
    let mut max = 0.0f32;
    for i in 0..16 {
        max = max.max((a[i] - b[i]).abs());
    }
    max
}

fn max_cage_difference(a: &[[f32; 3]], b: &[[f32; 3]]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    let mut max = 0.0f32;
    for (pa, pb) in a.iter().zip(b.iter()) {
        for axis in 0..3 {
            max = max.max((pa[axis] - pb[axis]).abs());
        }
    }
    max
}

/// Temporal hysteresis for Stage 3: reuse the previous sort when nothing moved.
#[derive(Default)]
pub struct SortCache {
    view_matrix: Option<[f32; 16]>,
    cage_state: Vec<[f32; 3]>,
    width: u32,
    height: u32,
    splat_count: usize,
    focal_x_bits: u32,
    focal_y_bits: u32,
    split_threshold_bits: u32,
    mip_var_bits: u32,
    binning: Option<TileBinningOutputs>,
    /// Number of frames served from this cache.
    pub reuse_count: u64,
}

impl SortCache {
    /// Returns true when the cached binning still matches the current frame.
    pub fn can_reuse(
        &self,
        view_matrix: &[f32; 16],
        cage_state: &[[f32; 3]],
        width: u32,
        height: u32,
    ) -> bool {
        let Some(cached_view) = self.view_matrix.as_ref() else {
            return false;
        };
        if self.binning.is_none() || self.width != width || self.height != height {
            return false;
        }
        max_abs_difference(cached_view, view_matrix) <= CAMERA_MOTION_EPSILON
            && max_cage_difference(&self.cage_state, cage_state) <= CAGE_MOTION_EPSILON
    }

    /// Removes and returns the cached binning when it is still valid.
    pub fn take_if_reusable(
        &mut self,
        view_matrix: &[f32; 16],
        cage_state: &[[f32; 3]],
        width: u32,
        height: u32,
    ) -> Option<TileBinningOutputs> {
        if self.can_reuse(view_matrix, cage_state, width, height) {
            self.reuse_count += 1;
            self.binning.take()
        } else {
            self.binning = None;
            None
        }
    }

    /// Extended cache check also tracking scene count, focal lengths, and adaptive parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn take_if_reusable_for_frame(
        &mut self,
        view_matrix: &[f32; 16],
        cage_state: &[[f32; 3]],
        width: u32,
        height: u32,
        splat_count: usize,
        focal_x: f32,
        focal_y: f32,
        split_threshold: f32,
        mip_var: f32,
    ) -> Option<TileBinningOutputs> {
        let params_match = self.splat_count == splat_count
            && self.focal_x_bits == focal_x.to_bits()
            && self.focal_y_bits == focal_y.to_bits()
            && self.split_threshold_bits == split_threshold.to_bits()
            && self.mip_var_bits == mip_var.to_bits();
        if params_match && self.can_reuse(view_matrix, cage_state, width, height) {
            self.reuse_count += 1;
            self.binning.take()
        } else {
            self.binning = None;
            None
        }
    }

    /// Stores the binning computed for this frame.
    pub fn commit(
        &mut self,
        view_matrix: &[f32; 16],
        cage_state: &[[f32; 3]],
        width: u32,
        height: u32,
        binning: TileBinningOutputs,
    ) {
        self.view_matrix = Some(*view_matrix);
        self.cage_state = cage_state.to_vec();
        self.width = width;
        self.height = height;
        self.binning = Some(binning);
    }

    /// Stores the binning along with extended frame key metadata.
    #[allow(clippy::too_many_arguments)]
    pub fn commit_for_frame(
        &mut self,
        view_matrix: &[f32; 16],
        cage_state: &[[f32; 3]],
        width: u32,
        height: u32,
        splat_count: usize,
        focal_x: f32,
        focal_y: f32,
        split_threshold: f32,
        mip_var: f32,
        binning: TileBinningOutputs,
    ) {
        self.splat_count = splat_count;
        self.focal_x_bits = focal_x.to_bits();
        self.focal_y_bits = focal_y.to_bits();
        self.split_threshold_bits = split_threshold.to_bits();
        self.mip_var_bits = mip_var.to_bits();
        self.commit(view_matrix, cage_state, width, height, binning);
    }

    /// Drops the cached binning (e.g. on resize or scene replacement).
    pub fn invalidate(&mut self) {
        self.binning = None;
        self.view_matrix = None;
        self.cage_state.clear();
        self.splat_count = 0;
    }
}

/// Orchestrator coordinating GPU buffer pools, deformation, projection,
/// binning, compositing, wireframe overlays, and fullscreen blitting.
pub struct RenderPipelineOrchestrator {
    pub gpu_pool: GpuBufferPool,
    pub blit_pipeline: BlitPipeline,
    pub wireframe: WireframeRenderer,
    pub blit_bind_group: Option<wgpu::BindGroup>,
    pub offscreen_texture: Option<wgpu::Texture>,
    pub offscreen_view: Option<wgpu::TextureView>,
    pub offscreen_width: u32,
    pub offscreen_height: u32,
    pub backend: ExecutionBackend,
    /// Persistent compute client. Created once, never per frame.
    pub gpu: Option<GpuCompute>,
    /// Set when GPU acquisition failed, so it is not retried every frame.
    pub gpu_error: Option<String>,
    /// Last per-frame GPU dispatch failure, if any (the frame fell back to CPU).
    pub last_backend_error: Option<String>,
    pub sort_cache: SortCache,
    pub frames_rendered: u64,
    /// Transient strain-driven Gaussian split threshold (`0.0` disables splitting).
    pub strain_split_threshold: f32,
    /// 3D Mip-Splatting isotropic filter variance (`0.0` disables 3D mip filter).
    pub mip_filter_3d_var: f32,
}

impl RenderPipelineOrchestrator {
    /// Creates a new pipeline orchestrator targeting the given swapchain surface format.
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let gpu_pool = GpuBufferPool::new(device);
        let blit_pipeline = BlitPipeline::new(device, target_format);
        let wireframe = WireframeRenderer::new(device, target_format);

        Self {
            gpu_pool,
            blit_pipeline,
            wireframe,
            blit_bind_group: None,
            offscreen_texture: None,
            offscreen_view: None,
            offscreen_width: 0,
            offscreen_height: 0,
            backend: ExecutionBackend::CpuReference,
            gpu: None,
            gpu_error: None,
            last_backend_error: None,
            sort_cache: SortCache::default(),
            frames_rendered: 0,
            strain_split_threshold: 0.0,
            mip_filter_3d_var: 0.0,
        }
    }

    /// Sets the transient stretch threshold for adaptive Gaussian splitting (`<= 0.0` disables).
    #[inline]
    pub fn set_strain_split_threshold(&mut self, threshold: f32) {
        self.strain_split_threshold = threshold.max(0.0);
        self.sort_cache.invalidate();
    }

    /// Sets the 3D Mip-Splatting isotropic regularization variance (`<= 0.0` disables).
    #[inline]
    pub fn set_mip_filter_3d_var(&mut self, var: f32) {
        self.mip_filter_3d_var = var.max(0.0);
        self.sort_cache.invalidate();
    }

    /// Registers the presentation `wgpu` device with CubeCL so one device serves both.
    ///
    /// On failure the orchestrator keeps working and falls back to a lazily acquired
    /// standalone compute device.
    #[allow(clippy::too_many_arguments)]
    pub fn attach_shared_gpu(
        &mut self,
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
        backend: wgpu::Backend,
    ) -> Result<(), GpuError> {
        let compute = GpuCompute::from_shared(instance, adapter, device, queue, backend)?;
        self.gpu = Some(compute);
        self.gpu_error = None;
        Ok(())
    }

    /// Returns the persistent compute client, acquiring it once on first use.
    pub fn compute(&mut self) -> Option<&GpuCompute> {
        if self.gpu.is_none() && self.gpu_error.is_none() {
            match GpuCompute::acquire() {
                Ok(compute) => self.gpu = Some(compute),
                Err(err) => self.gpu_error = Some(err.to_string()),
            }
        }
        self.gpu.as_ref()
    }

    /// Description of the active compute device, if any.
    pub fn compute_description(&self) -> Option<String> {
        self.gpu.as_ref().map(GpuCompute::describe)
    }

    /// Resizes the offscreen compositing texture to match new dimensions.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        if self.offscreen_width == width
            && self.offscreen_height == height
            && self.offscreen_texture.is_some()
        {
            return;
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("OffscreenCompositorTexture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: OFFSCREEN_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.blit_pipeline.create_bind_group(device, &view);

        self.offscreen_texture = Some(texture);
        self.offscreen_view = Some(view);
        self.blit_bind_group = Some(bind_group);
        self.offscreen_width = width;
        self.offscreen_height = height;
        // A resize changes the tiling geometry, so any cached sort is invalid.
        self.sort_cache.invalidate();
    }

    /// Builds the effective compositor uniforms from the offscreen target size.
    ///
    /// Rasterization, binning, and texture upload all use these dimensions, which
    /// removes the mismatch that could make `write_texture` panic after a resize.
    fn effective_compositor(&self, compositor: &CompositorUniforms) -> CompositorUniforms {
        if compositor.screen_width == self.offscreen_width
            && compositor.screen_height == self.offscreen_height
        {
            *compositor
        } else {
            CompositorUniforms::new(
                self.offscreen_width,
                self.offscreen_height,
                compositor.background_color,
                compositor.camera_position,
                compositor.sh_degree,
            )
        }
    }

    /// Flattens every cage's current vertices for sort-cache comparisons.
    fn cage_state(_cages: Option<&CageScene>, deformed: &[Vec<[f32; 3]>]) -> Vec<[f32; 3]> {
        deformed.iter().flatten().copied().collect()
    }

    /// Executes the CPU reference pipeline across all stages.
    ///
    /// Returns the framebuffer, telemetry, and `Some(binning)` only when the sort
    /// had to be recomputed (so the caller can cache it).
    #[allow(clippy::too_many_arguments)]
    pub fn execute_cpu_reference(
        scene: &GaussianSceneSoa,
        cages: Option<&CageScene>,
        camera: &CameraUniforms,
        compositor: &CompositorUniforms,
        width: u32,
        height: u32,
        previous_binning: Option<&TileBinningOutputs>,
    ) -> (
        RasterizedImage,
        usize,
        usize,
        usize,
        Option<TileBinningOutputs>,
    ) {
        Self::execute_cpu_reference_with_options(
            scene,
            cages,
            camera,
            compositor,
            width,
            height,
            previous_binning,
            0.0,
            0.0,
        )
    }

    /// CPU reference execution path supporting optional transient strain-driven
    /// splitting and 3D/2D Mip-Splatting anti-aliasing with active-only binning.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_cpu_reference_with_options(
        scene: &GaussianSceneSoa,
        cages: Option<&CageScene>,
        camera: &CameraUniforms,
        compositor: &CompositorUniforms,
        width: u32,
        height: u32,
        previous_binning: Option<&TileBinningOutputs>,
        strain_split_threshold: f32,
        mip_filter_3d_var: f32,
    ) -> (
        RasterizedImage,
        usize,
        usize,
        usize,
        Option<TileBinningOutputs>,
    ) {
        let has_cages = cages.is_some_and(|c| !c.is_empty());
        let deformed_scene: std::borrow::Cow<'_, GaussianSceneSoa> = match cages {
            Some(cage_scene) if !cage_scene.is_empty() => {
                let num_tets: usize = cage_scene.cages.iter().map(|c| c.mesh.num_tets()).sum();
                let uniforms = DeformationUniforms::new(
                    [0.0, 0.0, 0.0],
                    compositor.sh_degree,
                    scene.count as u32,
                    num_tets as u32,
                )
                .with_strain_splitting(strain_split_threshold)
                .with_mip_filter_3d(mip_filter_3d_var);
                std::borrow::Cow::Owned(deform_cage_scene_cpu(scene, cage_scene, &uniforms).scene)
            }
            _ => std::borrow::Cow::Borrowed(scene),
        };

        let use_cull_projection = strain_split_threshold > 0.0 || mip_filter_3d_var > 0.0;
        let (proj, effective_opacities): (
            crate::render::ProjectionOutputs,
            std::borrow::Cow<'_, [f32]>,
        ) = if use_cull_projection {
            let options = crate::math::ProjectionOptions {
                mip_filter_3d_var: if has_cages { 0.0 } else { mip_filter_3d_var },
                ..Default::default()
            };
            let culled = project_scene_cull_cpu(&deformed_scene, camera, &options);
            (culled.projection, std::borrow::Cow::Owned(culled.opacities))
        } else {
            (
                project_scene_cpu(&deformed_scene, camera),
                std::borrow::Cow::Borrowed(deformed_scene.opacities.as_slice()),
            )
        };

        let culled_count = proj.count.saturating_sub(proj.num_surviving());
        let rendered_count = proj.num_surviving();

        let computed = if previous_binning.is_none() {
            if use_cull_projection {
                Some(proj.bin_and_sort_active(width, height))
            } else {
                Some(proj.bin_and_sort(width, height))
            }
        } else {
            None
        };
        let Some(binning) = previous_binning.or(computed.as_ref()) else {
            return (
                RasterizedImage::new(width, height, compositor.background_color),
                culled_count,
                rendered_count,
                0,
                None,
            );
        };
        let active_tiles = binning.non_empty_tiles_count();

        let img = rasterize_scene_cpu(
            &proj.points_xy,
            &proj.conics,
            &effective_opacities,
            &deformed_scene.sh_coeffs,
            &deformed_scene.positions,
            &binning.sorted_entries,
            &binning.tile_ranges,
            compositor,
        );

        (img, culled_count, rendered_count, active_tiles, computed)
    }

    fn prepare_gpu_sh_coeffs<'a>(
        scene: &'a GaussianSceneSoa,
        cages: Option<&CageScene>,
        sh_degree: u32,
        split_sources: &[u32],
    ) -> std::borrow::Cow<'a, [f32]> {
        let Some(cage_scene) = cages.filter(|c| !c.is_empty()) else {
            return std::borrow::Cow::Borrowed(&scene.sh_coeffs);
        };
        let deg = sh_degree.min(3);
        if deg == 0 && split_sources.is_empty() {
            return std::borrow::Cow::Borrowed(&scene.sh_coeffs);
        }

        let mut rotated = scene.sh_coeffs.clone();
        if deg > 0 {
            let deformed = cage_scene.deformed_vertices();
            let rotators: Vec<Vec<ShRotator>> = cage_scene
                .cages
                .iter()
                .zip(deformed.iter())
                .map(|(cage, verts)| {
                    let (_, rots) =
                        crate::deformation::compute_tet_transforms_cpu(&cage.mesh, verts);
                    rots.into_iter().map(|r| ShRotator::new(r, deg)).collect()
                })
                .collect();

            for (i, binding) in cage_scene.bindings.iter().enumerate() {
                if !binding.is_bound() {
                    continue;
                }
                if let Some(rotator) = rotators
                    .get(binding.cage as usize)
                    .and_then(|tets| tets.get(binding.tet as usize))
                {
                    let base = i * 48;
                    if let Ok(coeffs) = rotated
                        .get_mut(base..base + 48)
                        .ok_or(())
                        .and_then(|s| <&mut [f32; 48]>::try_from(s).map_err(|_| ()))
                    {
                        rotator.apply(coeffs);
                    }
                }
            }
        }

        if !split_sources.is_empty() {
            rotated.reserve(split_sources.len() * 48);
            for &src in split_sources {
                let base = (src as usize) * 48;
                if base + 48 <= rotated.len() {
                    rotated.extend_from_within(base..base + 48);
                } else {
                    rotated.extend(std::iter::repeat_n(0.0, 48));
                }
            }
        }

        std::borrow::Cow::Owned(rotated)
    }

    /// Executes the native CubeCL GPU path.
    ///
    /// Deformation is applied cage by cage: each pass treats the previous pass's
    /// output as its "rest" input and only deforms the Gaussians it owns, so
    /// independent cages compose without a multi-cage kernel.
    #[allow(clippy::too_many_arguments)]
    fn execute_gpu(
        gpu: &GpuCompute,
        scene: &GaussianSceneSoa,
        cages: Option<&CageScene>,
        camera: &CameraUniforms,
        compositor: &CompositorUniforms,
        previous_binning: Option<&TileBinningOutputs>,
        strain_split_threshold: f32,
        mip_filter_3d_var: f32,
    ) -> Result<
        (
            RasterizedImage,
            usize,
            usize,
            usize,
            Option<TileBinningOutputs>,
        ),
        anyhow::Error,
    > {
        let client = gpu.client();
        let count = scene.count;
        if count == 0 {
            return Ok((
                RasterizedImage::new(
                    compositor.screen_width,
                    compositor.screen_height,
                    compositor.background_color,
                ),
                0,
                0,
                0,
                None,
            ));
        }

        let mut geometry: Option<DeformedGeometry> = None;
        let mut cumulative_split_sources: Vec<u32> = Vec::new();
        let use_adaptive_or_cull = strain_split_threshold > 0.0 || mip_filter_3d_var > 0.0;
        let mut has_cages = false;

        if let Some(cage_scene) = cages.filter(|c| !c.is_empty()) {
            let deformed = cage_scene.deformed_vertices();
            for (cage_index, cage) in cage_scene.cages.iter().enumerate() {
                if cage.mesh.num_tets() == 0 {
                    continue;
                }
                has_cages = true;

                let current_count = geometry.as_ref().map_or(count, |g| g.opacities.len());
                let mut bindings: Vec<GaussianBinding> = cage_scene
                    .bindings
                    .iter()
                    .map(|binding| {
                        if binding.is_bound() && binding.cage as usize == cage_index {
                            GaussianBinding::new(binding.tet, binding.weights)
                        } else {
                            GaussianBinding::unbound()
                        }
                    })
                    .collect();
                if bindings.len() < current_count {
                    bindings.resize(current_count, GaussianBinding::unbound());
                }

                let (positions, covariances, opacities) = match geometry.as_ref() {
                    Some(previous) => (
                        previous.positions.as_slice(),
                        previous.covariances.as_slice(),
                        previous.opacities.as_slice(),
                    ),
                    None => (
                        scene.positions.as_slice(),
                        scene.covariances_3d.as_slice(),
                        scene.opacities.as_slice(),
                    ),
                };

                let next_geom = if use_adaptive_or_cull {
                    deform_geometry_with_splitting_gpu(
                        client,
                        positions,
                        covariances,
                        opacities,
                        &cage.mesh,
                        &bindings,
                        &deformed[cage_index],
                        strain_split_threshold,
                        mip_filter_3d_var,
                    )?
                } else {
                    deform_geometry_gpu(
                        client,
                        positions,
                        covariances,
                        opacities,
                        &cage.mesh,
                        &bindings,
                        &deformed[cage_index],
                    )?
                };
                if !next_geom.split_source_indices.is_empty() {
                    for &src in &next_geom.split_source_indices {
                        let mapped = if (src as usize) < count {
                            src
                        } else {
                            let child_offset = (src as usize) - count;
                            cumulative_split_sources
                                .get(child_offset)
                                .copied()
                                .unwrap_or(src)
                        };
                        cumulative_split_sources.push(mapped);
                    }
                }
                geometry = Some(next_geom);
            }
        }

        let (positions, covariances, raw_opacities) = match geometry.as_ref() {
            Some(deformed) => (
                deformed.positions.as_slice(),
                deformed.covariances.as_slice(),
                deformed.opacities.as_slice(),
            ),
            None => (
                scene.positions.as_slice(),
                scene.covariances_3d.as_slice(),
                scene.opacities.as_slice(),
            ),
        };

        let (projection, opacities): (
            crate::render::ProjectionOutputs,
            std::borrow::Cow<'_, [f32]>,
        ) = if use_adaptive_or_cull {
            let culled = project_gaussians_cull_gpu(
                client,
                positions,
                covariances,
                raw_opacities,
                camera,
                if has_cages { 0.0 } else { mip_filter_3d_var },
                crate::math::MIN_CULL_OPACITY,
                crate::math::MIN_EFFECTIVE_RADIUS_PX,
            )?;
            (culled.projection, std::borrow::Cow::Owned(culled.opacities))
        } else {
            (
                project_gaussians_gpu(client, positions, covariances, camera)?,
                std::borrow::Cow::Borrowed(raw_opacities),
            )
        };

        let culled_count = projection.count.saturating_sub(projection.num_surviving());
        let rendered_count = projection.num_surviving();

        let computed = if previous_binning.is_none() {
            if use_adaptive_or_cull {
                Some(
                    projection
                        .bin_and_sort_active(compositor.screen_width, compositor.screen_height),
                )
            } else {
                Some(projection.bin_and_sort(compositor.screen_width, compositor.screen_height))
            }
        } else {
            None
        };
        let Some(binning) = previous_binning.or(computed.as_ref()) else {
            return Ok((
                RasterizedImage::new(
                    compositor.screen_width,
                    compositor.screen_height,
                    compositor.background_color,
                ),
                culled_count,
                rendered_count,
                0,
                None,
            ));
        };
        let active_tiles = binning.non_empty_tiles_count();
        let sh_coeffs = Self::prepare_gpu_sh_coeffs(
            scene,
            cages,
            compositor.sh_degree,
            &cumulative_split_sources,
        );

        let image = composite_gpu(
            client,
            &projection,
            &opacities,
            &sh_coeffs,
            positions,
            binning,
            compositor,
        )?;

        Ok((image, culled_count, rendered_count, active_tiles, computed))
    }

    /// Renders a frame and optionally blits it into a target view.
    #[allow(clippy::too_many_arguments)]
    pub fn render_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        scene: &GaussianSceneSoa,
        cages: Option<&CageScene>,
        camera: &CameraUniforms,
        compositor: &CompositorUniforms,
        surface_view: Option<&wgpu::TextureView>,
        show_wireframe: bool,
        selected: Option<(usize, usize)>,
    ) -> FrameResult {
        let width = self.offscreen_width;
        let height = self.offscreen_height;
        if width == 0 || height == 0 {
            return FrameResult {
                stats: FrameStats::default(),
                image: RasterizedImage::new(width, height, compositor.background_color),
            };
        }

        let compositor = self.effective_compositor(compositor);
        let camera = camera.with_viewport(width, height);

        self.gpu_pool.update_camera_uniforms(queue, &camera);
        self.gpu_pool.update_compositor_uniforms(queue, &compositor);

        let deformed_vertices = cages.map(CageScene::deformed_vertices).unwrap_or_default();
        let cage_state = Self::cage_state(cages, &deformed_vertices);
        let split_threshold = self.strain_split_threshold;
        let mip_var = self.mip_filter_3d_var;

        let reused_binning = self.sort_cache.take_if_reusable_for_frame(
            &camera.view_matrix,
            &cage_state,
            width,
            height,
            scene.count,
            camera.focal_x,
            camera.focal_y,
            split_threshold,
            mip_var,
        );
        let previous_ref = reused_binning.as_ref();
        let was_reused = previous_ref.is_some();

        let mut backend_error: Option<String> = None;

        let (image, culled, rendered, active_tiles, computed_binning) = match self.backend {
            ExecutionBackend::CubeClGpu => {
                if self.gpu.is_none() && self.gpu_error.is_none() {
                    let _ = self.compute();
                }
                let gpu_result = match self.gpu.as_ref() {
                    Some(gpu) => Self::execute_gpu(
                        gpu,
                        scene,
                        cages,
                        &camera,
                        &compositor,
                        previous_ref,
                        split_threshold,
                        mip_var,
                    ),
                    None => {
                        let message = self
                            .gpu_error
                            .clone()
                            .unwrap_or_else(|| "GPU compute unavailable".to_string());
                        Err(anyhow::anyhow!(message))
                    }
                };

                match gpu_result {
                    Ok(result) => result,
                    Err(err) => {
                        backend_error = Some(err.to_string());
                        Self::execute_cpu_reference_with_options(
                            scene,
                            cages,
                            &camera,
                            &compositor,
                            width,
                            height,
                            None,
                            split_threshold,
                            mip_var,
                        )
                    }
                }
            }
            ExecutionBackend::CpuReference => Self::execute_cpu_reference_with_options(
                scene,
                cages,
                &camera,
                &compositor,
                width,
                height,
                previous_ref,
                split_threshold,
                mip_var,
            ),
        };

        let sort_reused = was_reused && computed_binning.is_none();
        let final_binning = computed_binning.or(reused_binning);
        if let Some(binning) = final_binning {
            self.sort_cache.commit_for_frame(
                &camera.view_matrix,
                &cage_state,
                width,
                height,
                scene.count,
                camera.focal_x,
                camera.focal_y,
                split_threshold,
                mip_var,
                binning,
            );
        }

        self.last_backend_error = backend_error;
        self.frames_rendered += 1;

        // Upload the composited image into the offscreen sRGB texture.
        if let Some(texture) = &self.offscreen_texture {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &image.rgba8,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * width),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }

        if let (Some(view), Some(bind_group)) = (surface_view, &self.blit_bind_group) {
            let mut blit_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("FullscreenBlitPass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });

            self.blit_pipeline.render(&mut blit_pass, bind_group);
        }

        // Cage wireframe and pick-handle overlay pass.
        if show_wireframe {
            if let Some(cage_scene) =
                cages.filter(|c| !c.is_empty() && !deformed_vertices.is_empty())
            {
                let guides: Vec<CageGuide<'_>> = cage_scene
                    .cages
                    .iter()
                    .zip(deformed_vertices.iter())
                    .map(|(cage, vertices)| CageGuide {
                        vertices,
                        elements: &cage.mesh.elements,
                        edge_color: cage.edge_color,
                    })
                    .collect();

                if let Some(view) = surface_view {
                    let mut wireframe_pass =
                        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("CageWireframePass"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view,
                                depth_slice: None,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: None,
                            occlusion_query_set: None,
                            timestamp_writes: None,
                            multiview_mask: None,
                        });

                    self.wireframe.render_cages(
                        device,
                        queue,
                        &mut wireframe_pass,
                        &guides,
                        selected,
                        &camera,
                    );
                }
            }
        }

        FrameResult {
            stats: FrameStats {
                total_gaussians: scene.count,
                culled_gaussians: culled,
                rendered_gaussians: rendered,
                active_tiles,
                sort_reused,
            },
            image,
        }
    }

    /// Executes the full pipeline for a single frame and returns telemetry only.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        scene: &GaussianSceneSoa,
        cages: Option<&CageScene>,
        camera: &CameraUniforms,
        compositor: &CompositorUniforms,
        surface_view: &wgpu::TextureView,
        show_wireframe: bool,
        selected: Option<(usize, usize)>,
    ) -> FrameStats {
        self.render_frame(
            device,
            queue,
            encoder,
            scene,
            cages,
            camera,
            compositor,
            Some(surface_view),
            show_wireframe,
            selected,
        )
        .stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deformation::{Cage, TetMesh};
    use crate::render::tiles::TileConfigUniforms;

    fn tile_config(width: u32, height: u32) -> TileConfigUniforms {
        TileConfigUniforms::new(width, height, 1)
    }

    fn empty_binning(width: u32, height: u32) -> TileBinningOutputs {
        TileBinningOutputs {
            sorted_entries: Vec::new(),
            tile_ranges: vec![
                crate::render::tiles::TileRange::EMPTY;
                tile_config(width, height).total_tiles as usize
            ],
            total_entries: 0,
            config: tile_config(width, height),
        }
    }

    #[test]
    fn test_sort_cache_reuses_when_nothing_moves() {
        let view = [0.0f32; 16];
        let cage = vec![[0.0f32; 3], [1.0, 2.0, 3.0]];
        let mut cache = SortCache::default();

        assert!(!cache.can_reuse(&view, &cage, 800, 600));
        cache.commit(&view, &cage, 800, 600, empty_binning(800, 600));
        assert!(cache.can_reuse(&view, &cage, 800, 600));
        assert!(cache.take_if_reusable(&view, &cage, 800, 600).is_some());
        assert_eq!(cache.reuse_count, 1);
    }

    #[test]
    fn test_sort_cache_invalidates_on_camera_or_cage_motion() {
        let view = [0.0f32; 16];
        let cage = vec![[0.0f32; 3]];
        let mut cache = SortCache::default();
        cache.commit(&view, &cage, 800, 600, empty_binning(800, 600));

        let mut moved_view = view;
        moved_view[12] = 0.5;
        assert!(!cache.can_reuse(&moved_view, &cage, 800, 600));

        let moved_cage = vec![[0.5f32, 0.0, 0.0]];
        assert!(!cache.can_reuse(&view, &moved_cage, 800, 600));

        // Resize also invalidates.
        assert!(!cache.can_reuse(&view, &cage, 1024, 768));
    }

    #[test]
    fn test_sort_cache_tiny_motion_is_below_threshold() {
        let view = [0.0f32; 16];
        let cage = vec![[0.0f32; 3]];
        let mut cache = SortCache::default();
        cache.commit(&view, &cage, 800, 600, empty_binning(800, 600));

        let mut jittered = view;
        jittered[0] = CAGE_MOTION_EPSILON * 0.5;
        let cage_jitter = vec![[CAGE_MOTION_EPSILON * 0.5, 0.0, 0.0]];
        assert!(cache.can_reuse(&jittered, &cage_jitter, 800, 600));
    }

    #[test]
    fn test_effective_compositor_tracks_offscreen_size() {
        let compositor =
            CompositorUniforms::new(1280, 720, [0.1, 0.2, 0.3, 1.0], [1.0, 2.0, 3.0], 3);
        // Rebuild the same logic without a GPU device.
        let width = 800u32;
        let height = 600u32;
        let effective = if compositor.screen_width == width && compositor.screen_height == height {
            compositor
        } else {
            CompositorUniforms::new(
                width,
                height,
                compositor.background_color,
                compositor.camera_position,
                compositor.sh_degree,
            )
        };
        assert_eq!(effective.screen_width, 800);
        assert_eq!(effective.screen_height, 600);
        assert_eq!(effective.tiles_x, 50);
        assert_eq!(effective.background_color, compositor.background_color);
        assert_eq!(effective.sh_degree, 3);
    }

    #[test]
    fn test_camera_viewport_is_rebuilt_consistently() {
        let camera = CameraUniforms::from_look_at(
            glam::Vec3::new(0.0, 0.0, -3.0),
            glam::Vec3::ZERO,
            glam::Vec3::Y,
            45.0f32.to_radians(),
            1280.0,
            720.0,
            0.2,
        );
        let resized = camera.with_viewport(800, 600);
        assert_eq!(resized.viewport_width, 800.0);
        assert_eq!(resized.viewport_height, 600.0);
        assert_eq!(resized.principal_x, 400.0);
        assert_eq!(resized.principal_y, 300.0);
        // Vertical FOV is preserved.
        let fov_old = 2.0 * (0.5 * 720.0 / camera.focal_y).atan();
        let fov_new = 2.0 * (0.5 * 600.0 / resized.focal_y).atan();
        assert!((fov_old - fov_new).abs() < 1e-5);
    }

    #[test]
    fn test_multi_cage_state_flattens_deterministically() {
        let cage_a = Cage::new(
            "a",
            TetMesh::create_box_cage([-1.0, -1.0, -1.0], [0.0, 0.0, 0.0]),
            [1.0, 0.0, 0.0, 1.0],
        );
        let cage_b = Cage::new(
            "b",
            TetMesh::create_box_cage([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            [0.0, 1.0, 0.0, 1.0],
        );
        let cages = CageScene::new(vec![cage_a, cage_b], Vec::new());
        let deformed = cages.deformed_vertices();
        let flat = RenderPipelineOrchestrator::cage_state(Some(&cages), &deformed);
        assert_eq!(flat.len(), 16);
    }
}
