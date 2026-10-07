use crate::deformation::{deform_scene_cpu, DeformationUniforms, GaussianBinding, TetMesh};
use crate::render::blit::BlitPipeline;
use crate::render::camera::CameraUniforms;
use crate::render::gpu_pool::GpuBufferPool;
use crate::render::kernels::{
    compute_tet_jacobians_kernel, deform_gaussians_kernel,
    tile_compositor_kernel,
};
use crate::render::rasterizer::{rasterize_scene_cpu, CompositorUniforms, RasterizedImage};
use crate::render::wireframe::WireframeRenderer;
use crate::render::{project_scene_cpu, project_scene_cubecl, ProjectionOutputs};
use crate::scene::soa::GaussianSceneSoa;
use cubecl::prelude::*;
use cubecl::Device;

/// Execution backend selector between CPU reference and native CubeCL GPU compute kernels.
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
}

/// Orchestrator coordinating dynamic GPU buffer pools, deformation, projection,
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
        }
    }

    /// Resizes the offscreen `Rgba8Unorm` compositing texture to match new dimensions.
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
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.blit_pipeline.create_bind_group(device, &view);

        self.offscreen_texture = Some(texture);
        self.offscreen_view = Some(view);
        self.blit_bind_group = Some(bind_group);
        self.offscreen_width = width;
        self.offscreen_height = height;
    }

    /// Executes the CPU reference pipeline across all stages.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_cpu_reference(
        scene: &GaussianSceneSoa,
        tet_mesh: Option<&TetMesh>,
        bindings: Option<&[GaussianBinding]>,
        deformed_cage_vertices: Option<&[[f32; 3]]>,
        camera: &CameraUniforms,
        compositor: &CompositorUniforms,
        width: u32,
        height: u32,
    ) -> (RasterizedImage, usize, usize, usize) {
        let active_scene: std::borrow::Cow<'_, GaussianSceneSoa> =
            match (tet_mesh, bindings, deformed_cage_vertices) {
                (Some(mesh), Some(bind), Some(cage_verts)) => {
                    let def_uniforms = DeformationUniforms::new(
                        [
                            camera.view_matrix[12],
                            camera.view_matrix[13],
                            camera.view_matrix[14],
                        ],
                        compositor.sh_degree,
                        scene.count as u32,
                        mesh.num_tets() as u32,
                    );
                    let deformed_output =
                        deform_scene_cpu(scene, mesh, bind, cage_verts, &def_uniforms);
                    std::borrow::Cow::Owned(deformed_output.scene)
                }
                _ => std::borrow::Cow::Borrowed(scene),
            };

        let proj = project_scene_cpu(&active_scene, camera);
        let culled_count = proj.count.saturating_sub(proj.num_surviving());
        let rendered_count = proj.num_surviving();

        let binning = proj.bin_and_sort(width, height);
        let active_tiles = binning.non_empty_tiles_count();

        let img = rasterize_scene_cpu(
            &proj.points_xy,
            &proj.conics,
            &active_scene.opacities,
            &active_scene.sh_coeffs,
            &active_scene.positions,
            &binning.sorted_entries,
            &binning.tile_ranges,
            compositor,
        );

        (img, culled_count, rendered_count, active_tiles)
    }

    /// Executes native CubeCL GPU compute kernels for volumetric deformation,
    /// projection, and shared-memory tile compositing.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_cubecl_gpu(
        scene: &GaussianSceneSoa,
        tet_mesh: Option<&TetMesh>,
        bindings: Option<&[GaussianBinding]>,
        deformed_cage_vertices: Option<&[[f32; 3]]>,
        camera: &CameraUniforms,
        compositor: &CompositorUniforms,
    ) -> Result<(RasterizedImage, usize, usize, usize), anyhow::Error> {
        let count = scene.len();
        let width = compositor.screen_width;
        let height = compositor.screen_height;
        if count == 0 || width == 0 || height == 0 {
            return Ok((
                RasterizedImage::new(width, height, compositor.background_color),
                0,
                0,
                0,
            ));
        }

        let client = {
            let dev = Device::Wgpu(cubecl::wgpu::WgpuDevice::new(
                cubecl::wgpu::WgpuDeviceKind::DiscreteGpu(0),
            ));
            std::panic::catch_unwind(|| dev.client()).unwrap_or_else(|_| Device::default().client())
        };

        // 1. Stage 1: Volumetric deformation (if cage provided)
        let (pos_bytes, cov_bytes, opa_bytes, sh_bytes) = if let (
            Some(mesh),
            Some(binds),
            Some(cage_verts),
        ) = (tet_mesh, bindings, deformed_cage_vertices)
        {
            let num_tets = mesh.num_tets() as u32;
            let flat_cage: Vec<f32> = cage_verts.iter().flat_map(|v| [v[0], v[1], v[2]]).collect();
            let flat_elems: Vec<u32> = mesh.elements.iter().flat_map(|e| e.indices).collect();
            let flat_inv_dm: Vec<f32> = mesh.precomputed.iter().flat_map(|p| p.inv_dm).collect();

            let cage_handle = client.create(cubecl_common::bytes::Bytes::from_bytes_vec(
                bytemuck::cast_slice(&flat_cage).to_vec(),
            ));
            let elems_handle = client.create(cubecl_common::bytes::Bytes::from_bytes_vec(
                bytemuck::cast_slice(&flat_elems).to_vec(),
            ));
            let inv_dm_handle = client.create(cubecl_common::bytes::Bytes::from_bytes_vec(
                bytemuck::cast_slice(&flat_inv_dm).to_vec(),
            ));

            let jac_handle = client.empty((num_tets as usize * 9 * 4).max(4));
            let rot_handle = client.empty((num_tets as usize * 9 * 4).max(4));

            let cube_dim_tet = CubeDim::new_1d(64.min(num_tets.max(1)));
            let cube_count_tet = CubeCount::Static(num_tets.div_ceil(64), 1, 1);

            let cage_arg = unsafe { BufferArg::from_raw_parts(cage_handle.clone(), flat_cage.len()) };
            let elems_arg = unsafe { BufferArg::from_raw_parts(elems_handle.clone(), flat_elems.len()) };
            let inv_dm_arg = unsafe { BufferArg::from_raw_parts(inv_dm_handle.clone(), flat_inv_dm.len()) };
            let jac_arg = unsafe { BufferArg::from_raw_parts(jac_handle.clone(), num_tets as usize * 9) };
            let rot_arg = unsafe { BufferArg::from_raw_parts(rot_handle.clone(), num_tets as usize * 9) };

            compute_tet_jacobians_kernel::launch(
                &client,
                cube_count_tet,
                cube_dim_tet,
                cage_arg,
                elems_arg,
                inv_dm_arg,
                jac_arg,
                rot_arg,
                num_tets,
            );

            // Deform Gaussians
            let bind_tets: Vec<u32> = binds.iter().map(|b| b.tet_index).collect();
            let bind_weights: Vec<f32> = binds.iter().flat_map(|b| b.weights).collect();

            let bind_tets_h = client.create(cubecl_common::bytes::Bytes::from_bytes_vec(
                bytemuck::cast_slice(&bind_tets).to_vec(),
            ));
            let bind_weights_h = client.create(cubecl_common::bytes::Bytes::from_bytes_vec(
                bytemuck::cast_slice(&bind_weights).to_vec(),
            ));
            let rest_pos_h = client.create(cubecl_common::bytes::Bytes::from_bytes_vec(
                scene.positions_bytes().to_vec(),
            ));
            let rest_cov_h = client.create(cubecl_common::bytes::Bytes::from_bytes_vec(
                scene.covariances_bytes().to_vec(),
            ));
            let rest_opa_h = client.create(cubecl_common::bytes::Bytes::from_bytes_vec(
                scene.opacities_bytes().to_vec(),
            ));
            let sh_h = client.create(cubecl_common::bytes::Bytes::from_bytes_vec(
                scene.sh_coeffs_bytes().to_vec(),
            ));

            let out_pos_h = client.empty((count * 3 * 4).max(4));
            let out_cov_h = client.empty((count * 6 * 4).max(4));
            let out_opa_h = client.empty((count * 4).max(4));
            let out_col_h = client.empty((count * 3 * 4).max(4));

            let cube_dim_g = CubeDim::new_1d(256);
            let cube_count_g = CubeCount::Static((count as u32).div_ceil(256), 1, 1);

            let cage_arg = unsafe { BufferArg::from_raw_parts(cage_handle, flat_cage.len()) };
            let elems_arg = unsafe { BufferArg::from_raw_parts(elems_handle, flat_elems.len()) };
            let jac_arg = unsafe { BufferArg::from_raw_parts(jac_handle, num_tets as usize * 9) };
            let rot_arg = unsafe { BufferArg::from_raw_parts(rot_handle, num_tets as usize * 9) };
            let bind_t_arg = unsafe { BufferArg::from_raw_parts(bind_tets_h, bind_tets.len()) };
            let bind_w_arg = unsafe { BufferArg::from_raw_parts(bind_weights_h, bind_weights.len()) };
            let r_pos_arg = unsafe { BufferArg::from_raw_parts(rest_pos_h, count * 3) };
            let r_cov_arg = unsafe { BufferArg::from_raw_parts(rest_cov_h, count * 6) };
            let r_opa_arg = unsafe { BufferArg::from_raw_parts(rest_opa_h, count) };
            let sh_arg = unsafe { BufferArg::from_raw_parts(sh_h, count * 48) };
            let o_pos_arg = unsafe { BufferArg::from_raw_parts(out_pos_h.clone(), count * 3) };
            let o_cov_arg = unsafe { BufferArg::from_raw_parts(out_cov_h.clone(), count * 6) };
            let o_opa_arg = unsafe { BufferArg::from_raw_parts(out_opa_h.clone(), count) };
            let o_col_arg = unsafe { BufferArg::from_raw_parts(out_col_h, count * 3) };

            deform_gaussians_kernel::launch(
                &client,
                cube_count_g,
                cube_dim_g,
                cage_arg,
                elems_arg,
                jac_arg,
                rot_arg,
                bind_t_arg,
                bind_w_arg,
                r_pos_arg,
                r_cov_arg,
                r_opa_arg,
                sh_arg,
                o_pos_arg,
                o_cov_arg,
                o_opa_arg,
                o_col_arg,
                camera.view_matrix[12],
                camera.view_matrix[13],
                camera.view_matrix[14],
                compositor.sh_degree,
                count as u32,
                num_tets,
            );

            let pos_bytes = client
                .read_one(out_pos_h)
                .map_err(|e| anyhow::anyhow!("{:?}", e))?
                .to_vec();
            let cov_bytes = client
                .read_one(out_cov_h)
                .map_err(|e| anyhow::anyhow!("{:?}", e))?
                .to_vec();
            let opa_bytes = client
                .read_one(out_opa_h)
                .map_err(|e| anyhow::anyhow!("{:?}", e))?
                .to_vec();
            let sh_bytes = scene.sh_coeffs_bytes().to_vec();

            (pos_bytes, cov_bytes, opa_bytes, sh_bytes)
        } else {
            (
                scene.positions_bytes().to_vec(),
                scene.covariances_bytes().to_vec(),
                scene.opacities_bytes().to_vec(),
                scene.sh_coeffs_bytes().to_vec(),
            )
        };

        // 2. Stage 2: Camera projection (project_gaussians_kernel)
        let pos_h = client.create(cubecl_common::bytes::Bytes::from_bytes_vec(pos_bytes.clone()));
        let cov_h = client.create(cubecl_common::bytes::Bytes::from_bytes_vec(cov_bytes));
        let cam_h = client.create(cubecl_common::bytes::Bytes::from_bytes_vec(
            camera.as_bytes().to_vec(),
        ));

        let (pts_h, d_h, r_h, c_h) = project_scene_cubecl(&client, &pos_h, &cov_h, &cam_h, count);

        let pts_bytes = client
            .read_one(pts_h.clone())
            .map_err(|e| anyhow::anyhow!("{:?}", e))?;
        let d_bytes = client
            .read_one(d_h)
            .map_err(|e| anyhow::anyhow!("{:?}", e))?;
        let r_bytes = client
            .read_one(r_h)
            .map_err(|e| anyhow::anyhow!("{:?}", e))?;
        let c_bytes = client
            .read_one(c_h.clone())
            .map_err(|e| anyhow::anyhow!("{:?}", e))?;

        let pts_slice: &[f32] = bytemuck::cast_slice(&pts_bytes[..count * 2 * 4]);
        let d_slice: &[f32] = bytemuck::cast_slice(&d_bytes[..count * 4]);
        let r_slice: &[i32] = bytemuck::cast_slice(&r_bytes[..count * 4]);
        let c_slice: &[f32] = bytemuck::cast_slice(&c_bytes[..count * 3 * 4]);

        let proj = ProjectionOutputs {
            points_xy: pts_slice.to_vec(),
            depths: d_slice.to_vec(),
            radii: r_slice.to_vec(),
            conics: c_slice.to_vec(),
            count,
        };
        let culled_count = proj.count.saturating_sub(proj.num_surviving());
        let rendered_count = proj.num_surviving();

        // 3. Stage 3: Binning and sorting
        let binning = proj.bin_and_sort(width, height);
        let active_tiles = binning.non_empty_tiles_count();

        // 4. Stage 4: Tile compositor kernel
        let opa_slice: &[f32] = bytemuck::cast_slice(&opa_bytes[..count * 4]);
        let sh_slice: &[f32] = bytemuck::cast_slice(&sh_bytes[..count * 48 * 4]);
        let pos_slice: &[f32] = bytemuck::cast_slice(&pos_bytes[..count * 3 * 4]);

        let sorted_entries_flat: Vec<u32> = binning
            .sorted_entries
            .iter()
            .flat_map(|e| [e.key.tile_id, e.key.depth_bits, e.gaussian_id])
            .collect();
        let tile_ranges_flat: Vec<u32> = binning
            .tile_ranges
            .iter()
            .flat_map(|r| [r.start, r.end])
            .collect();

        let sorted_h = client.create(cubecl_common::bytes::Bytes::from_bytes_vec(
            bytemuck::cast_slice(&sorted_entries_flat).to_vec(),
        ));
        let ranges_h = client.create(cubecl_common::bytes::Bytes::from_bytes_vec(
            bytemuck::cast_slice(&tile_ranges_flat).to_vec(),
        ));
        let opa_h = client.create(cubecl_common::bytes::Bytes::from_bytes_vec(
            bytemuck::cast_slice(opa_slice).to_vec(),
        ));
        let sh_h = client.create(cubecl_common::bytes::Bytes::from_bytes_vec(
            bytemuck::cast_slice(sh_slice).to_vec(),
        ));
        let pos_3d_h = client.create(cubecl_common::bytes::Bytes::from_bytes_vec(
            bytemuck::cast_slice(pos_slice).to_vec(),
        ));

        let pixel_count = (width * height) as usize;
        let out_pixels_h = client.empty((pixel_count * 4 * 4).max(4));

        let cube_dim_comp = CubeDim::new_1d(256);
        let cube_count_comp = CubeCount::Static(compositor.tiles_x, compositor.tiles_y, 1);

        let pts_arg = unsafe { BufferArg::from_raw_parts(pts_h, count * 2) };
        let con_arg = unsafe { BufferArg::from_raw_parts(c_h, count * 3) };
        let opa_arg = unsafe { BufferArg::from_raw_parts(opa_h, count) };
        let sh_arg = unsafe { BufferArg::from_raw_parts(sh_h, count * 48) };
        let pos_arg = unsafe { BufferArg::from_raw_parts(pos_3d_h, count * 3) };
        let sort_arg = unsafe { BufferArg::from_raw_parts(sorted_h, sorted_entries_flat.len()) };
        let range_arg = unsafe { BufferArg::from_raw_parts(ranges_h, tile_ranges_flat.len()) };
        let pix_arg = unsafe { BufferArg::from_raw_parts(out_pixels_h.clone(), pixel_count * 4) };

        tile_compositor_kernel::launch(
            &client,
            cube_count_comp,
            cube_dim_comp,
            pts_arg,
            con_arg,
            opa_arg,
            sh_arg,
            pos_arg,
            sort_arg,
            range_arg,
            pix_arg,
            width,
            height,
            compositor.tiles_x,
            compositor.background_color[0],
            compositor.background_color[1],
            compositor.background_color[2],
            compositor.background_color[3],
            compositor.camera_position[0],
            compositor.camera_position[1],
            compositor.camera_position[2],
            compositor.sh_degree,
        );

        let pix_bytes = client
            .read_one(out_pixels_h)
            .map_err(|e| anyhow::anyhow!("{:?}", e))?;
        let pix_slice: &[f32] = bytemuck::cast_slice(&pix_bytes[..pixel_count * 4 * 4]);

        let mut img = RasterizedImage::new(width, height, compositor.background_color);
        for i in 0..pixel_count {
            img.rgba_f32[i * 4] = pix_slice[i * 4];
            img.rgba_f32[i * 4 + 1] = pix_slice[i * 4 + 1];
            img.rgba_f32[i * 4 + 2] = pix_slice[i * 4 + 2];
            img.rgba_f32[i * 4 + 3] = pix_slice[i * 4 + 3];

            img.rgba8[i * 4] = (pix_slice[i * 4].clamp(0.0, 1.0) * 255.0).round() as u8;
            img.rgba8[i * 4 + 1] = (pix_slice[i * 4 + 1].clamp(0.0, 1.0) * 255.0).round() as u8;
            img.rgba8[i * 4 + 2] = (pix_slice[i * 4 + 2].clamp(0.0, 1.0) * 255.0).round() as u8;
            img.rgba8[i * 4 + 3] = (pix_slice[i * 4 + 3].clamp(0.0, 1.0) * 255.0).round() as u8;
        }

        Ok((img, culled_count, rendered_count, active_tiles))
    }

    /// Executes the full 3DGS render pipeline for a single frame:
    /// 1. Volumetric cage deformation (if cage present)
    /// 2. Camera frustum projection & conic computation
    /// 3. Tile intersection counting, monotonic depth sorting, and boundary extraction
    /// 4. Tile compositor rasterization into offscreen buffer (CPU or CubeCL GPU)
    /// 5. Upload to offscreen texture and fullscreen blit into swapchain surface view
    /// 6. Cage wireframe & pick handles overlay pass directly on swapchain
    #[allow(clippy::too_many_arguments)]
    pub fn execute_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        scene: &GaussianSceneSoa,
        tet_mesh: Option<&TetMesh>,
        bindings: Option<&[GaussianBinding]>,
        deformed_cage_vertices: Option<&[[f32; 3]]>,
        camera: &CameraUniforms,
        compositor: &CompositorUniforms,
        surface_view: &wgpu::TextureView,
        show_wireframe: bool,
        selected_vertex: Option<usize>,
    ) -> FrameStats {
        let width = self.offscreen_width;
        let height = self.offscreen_height;
        if width == 0 || height == 0 {
            return FrameStats::default();
        }

        // 1. Ensure GPU buffer pool capacities & update uniforms
        self.gpu_pool.ensure_gaussian_capacity(device, scene.count);
        self.gpu_pool.update_camera_uniforms(queue, camera);
        self.gpu_pool.update_compositor_uniforms(queue, compositor);

        // 2. Execute pipeline according to selected execution backend
        let (img, culled_count, rendered_count, active_tiles) = match self.backend {
            ExecutionBackend::CubeClGpu => {
                match Self::execute_cubecl_gpu(
                    scene,
                    tet_mesh,
                    bindings,
                    deformed_cage_vertices,
                    camera,
                    compositor,
                ) {
                    Ok(res) => res,
                    Err(e) => {
                        eprintln!(
                            "CubeCL GPU dispatch failed ({:?}), falling back to CPU reference",
                            e
                        );
                        Self::execute_cpu_reference(
                            scene,
                            tet_mesh,
                            bindings,
                            deformed_cage_vertices,
                            camera,
                            compositor,
                            width,
                            height,
                        )
                    }
                }
            }
            ExecutionBackend::CpuReference => Self::execute_cpu_reference(
                scene,
                tet_mesh,
                bindings,
                deformed_cage_vertices,
                camera,
                compositor,
                width,
                height,
            ),
        };

        // 3. Upload rasterized pixels to offscreen texture
        if let Some(texture) = &self.offscreen_texture {
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &img.rgba8,
                wgpu::ImageDataLayout {
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

        // 4. Fullscreen blit pass to swapchain surface
        if let Some(bind_group) = &self.blit_bind_group {
            let mut blit_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("FullscreenBlitPass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: surface_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            self.blit_pipeline.render(&mut blit_pass, bind_group);
        }

        // 5. Wireframe and pick handle overlay pass
        if let (true, Some(mesh), Some(cage_verts)) =
            (show_wireframe, tet_mesh, deformed_cage_vertices)
        {
            let mut wireframe_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("CageWireframePass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: surface_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });

                self.wireframe.render(
                    device,
                    queue,
                    &mut wireframe_pass,
                    cage_verts,
                    &mesh.elements,
                    selected_vertex,
                    camera,
                );
            }

        FrameStats {
            total_gaussians: scene.count,
            culled_gaussians: culled_count,
            rendered_gaussians: rendered_count,
            active_tiles,
        }
    }
}
