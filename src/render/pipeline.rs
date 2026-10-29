use crate::deformation::{CageScene, DeformationUniforms, GaussianBinding, deform_cage_scene_cpu};
use crate::render::blit::{BlitPipeline, OFFSCREEN_FORMAT};
use crate::render::camera::CameraUniforms;
use crate::render::gpu::{GpuCompute, GpuError};
use crate::render::gpu_bridge::{
    DeformedGeometry, composite_gpu, deform_geometry_gpu, project_gaussians_gpu,
};
use crate::render::gpu_pool::GpuBufferPool;
use crate::render::project_scene_cpu;
use crate::render::rasterizer::{CompositorUniforms, RasterizedImage, rasterize_scene_cpu};
use crate::render::tiles::TileBinningOutputs;
use crate::render::wireframe::{CageGuide, WireframeRenderer};
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

    /// Drops the cached binning (e.g. on resize or scene replacement).
    pub fn invalidate(&mut self) {
        self.binning = None;
        self.view_matrix = None;
        self.cage_state.clear();
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
        }
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
        let deformed_scene: std::borrow::Cow<'_, GaussianSceneSoa> = match cages {
            Some(cage_scene) if !cage_scene.is_empty() => {
                let num_tets: usize = cage_scene.cages.iter().map(|c| c.mesh.num_tets()).sum();
                let uniforms = DeformationUniforms::new(
                    [0.0, 0.0, 0.0],
                    compositor.sh_degree,
                    scene.count as u32,
                    num_tets as u32,
                );
                std::borrow::Cow::Owned(deform_cage_scene_cpu(scene, cage_scene, &uniforms).scene)
            }
            _ => std::borrow::Cow::Borrowed(scene),
        };

        let proj = project_scene_cpu(&deformed_scene, camera);
        let culled_count = proj.count.saturating_sub(proj.num_surviving());
        let rendered_count = proj.num_surviving();

        let computed = if previous_binning.is_none() {
            Some(proj.bin_and_sort(width, height))
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
            &deformed_scene.opacities,
            &deformed_scene.sh_coeffs,
            &deformed_scene.positions,
            &binning.sorted_entries,
            &binning.tile_ranges,
            compositor,
        );

        (img, culled_count, rendered_count, active_tiles, computed)
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

        if let Some(cage_scene) = cages.filter(|c| !c.is_empty()) {
            let deformed = cage_scene.deformed_vertices();
            for (cage_index, cage) in cage_scene.cages.iter().enumerate() {
                if cage.mesh.num_tets() == 0 {
                    continue;
                }

                let bindings: Vec<GaussianBinding> = cage_scene
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

                geometry = Some(deform_geometry_gpu(
                    client,
                    positions,
                    covariances,
                    opacities,
                    &cage.mesh,
                    &bindings,
                    &deformed[cage_index],
                )?);
            }
        }

        let (positions, covariances, opacities) = match geometry.as_ref() {
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

        let projection = project_gaussians_gpu(client, positions, covariances, camera)?;
        let culled_count = projection.count.saturating_sub(projection.num_surviving());
        let rendered_count = projection.num_surviving();

        let computed = if previous_binning.is_none() {
            Some(projection.bin_and_sort(compositor.screen_width, compositor.screen_height))
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

        let image = composite_gpu(
            client,
            &projection,
            opacities,
            &scene.sh_coeffs,
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

        let reused_binning =
            self.sort_cache
                .take_if_reusable(&camera.view_matrix, &cage_state, width, height);
        let previous_ref = reused_binning.as_ref();
        let was_reused = previous_ref.is_some();

        let mut backend_error: Option<String> = None;

        let (image, culled, rendered, active_tiles, computed_binning) = match self.backend {
            ExecutionBackend::CubeClGpu => {
                if self.gpu.is_none() && self.gpu_error.is_none() {
                    let _ = self.compute();
                }
                let gpu_result = match self.gpu.as_ref() {
                    Some(gpu) => {
                        Self::execute_gpu(gpu, scene, cages, &camera, &compositor, previous_ref)
                    }
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
                        Self::execute_cpu_reference(
                            scene,
                            cages,
                            &camera,
                            &compositor,
                            width,
                            height,
                            None,
                        )
                    }
                }
            }
            ExecutionBackend::CpuReference => Self::execute_cpu_reference(
                scene,
                cages,
                &camera,
                &compositor,
                width,
                height,
                previous_ref,
            ),
        };

        let sort_reused = was_reused && computed_binning.is_none();
        let final_binning = computed_binning.or(reused_binning);
        if let Some(binning) = final_binning {
            self.sort_cache
                .commit(&camera.view_matrix, &cage_state, width, height, binning);
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
