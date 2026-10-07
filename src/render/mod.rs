pub mod blit;
pub mod camera;
pub mod gpu_pool;
pub mod kernels;
pub mod pipeline;
pub mod rasterizer;
pub mod tiles;
pub mod wireframe;

pub use blit::*;
pub use camera::*;
pub use gpu_pool::*;
pub use kernels::*;
pub use pipeline::*;
pub use rasterizer::*;
pub use tiles::*;
pub use wireframe::*;


use crate::math::project_ewa_splat;
use crate::scene::GaussianSceneSoa;
use cubecl::prelude::*;
use cubecl::Device;
use glam::Vec3;
use rayon::prelude::*;

/// Output buffers containing projected 2D Gaussian splat attributes for rasterization.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectionOutputs {
    /// Screen-space 2D centers [mu_x, mu_y]: [N * 2] floats.
    pub points_xy: Vec<f32>,

    /// Camera-space depth t_z: [N] floats.
    pub depths: Vec<f32>,

    /// Clamped pixel radius r: [N] i32 (0 if culled).
    pub radii: Vec<i32>,

    /// Conic coefficients [a, b, c]: [N * 3] floats.
    pub conics: Vec<f32>,

    /// Total number of splats processed.
    pub count: usize,
}

impl ProjectionOutputs {
    /// Creates an empty `ProjectionOutputs`.
    pub fn empty() -> Self {
        Self {
            points_xy: Vec::new(),
            depths: Vec::new(),
            radii: Vec::new(),
            conics: Vec::new(),
            count: 0,
        }
    }

    /// Pre-allocates buffers for `capacity` splats.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            points_xy: Vec::with_capacity(capacity * 2),
            depths: Vec::with_capacity(capacity),
            radii: Vec::with_capacity(capacity),
            conics: Vec::with_capacity(capacity * 3),
            count: 0,
        }
    }

    /// Number of splats that survived near-plane, determinant, and frustum culling.
    pub fn num_surviving(&self) -> usize {
        self.radii.iter().filter(|&&r| r > 0).count()
    }

    /// Byte slice of 2D centers for GPU buffer upload.
    #[inline]
    pub fn points_xy_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.points_xy)
    }

    /// Byte slice of depths for GPU buffer upload.
    #[inline]
    pub fn depths_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.depths)
    }

    /// Byte slice of radii for GPU buffer upload.
    #[inline]
    pub fn radii_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.radii)
    }

    /// Byte slice of conics for GPU buffer upload.
    #[inline]
    pub fn conics_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.conics)
    }

    /// Executes Module 4 tile binning, scattering, and sorting on CPU.
    pub fn bin_and_sort(&self, screen_width: u32, screen_height: u32) -> TileBinningOutputs {
        let config = TileConfigUniforms::new(screen_width, screen_height, self.count as u32);
        bin_and_sort_gaussians_cpu(&self.points_xy, &self.depths, &self.radii, config)
    }
}

/// Dispatches the EWA projection pipeline on CPU using Rayon multi-threading.
pub fn project_scene_cpu(
    scene: &GaussianSceneSoa,
    camera: &CameraUniforms,
) -> ProjectionOutputs {
    let count = scene.len();
    if count == 0 {
        return ProjectionOutputs::empty();
    }

    let mut points_xy = vec![0.0f32; count * 2];
    let mut depths = vec![0.0f32; count];
    let mut radii = vec![0i32; count];
    let mut conics = vec![0.0f32; count * 3];

    let view_mat = camera.view_mat4();
    const CHUNK_SIZE: usize = 2048;

    points_xy
        .par_chunks_mut(CHUNK_SIZE * 2)
        .zip(depths.par_chunks_mut(CHUNK_SIZE))
        .zip(radii.par_chunks_mut(CHUNK_SIZE))
        .zip(conics.par_chunks_mut(CHUNK_SIZE * 3))
        .enumerate()
        .for_each(|(chunk_idx, (((pts_chunk, d_chunk), r_chunk), c_chunk))| {
            let base_i = chunk_idx * CHUNK_SIZE;
            for (local_i, r_val) in r_chunk.iter_mut().enumerate() {
                let i = base_i + local_i;
                let pos = Vec3::new(
                    scene.positions[i * 3],
                    scene.positions[i * 3 + 1],
                    scene.positions[i * 3 + 2],
                );
                let cov = [
                    scene.covariances_3d[i * 6],
                    scene.covariances_3d[i * 6 + 1],
                    scene.covariances_3d[i * 6 + 2],
                    scene.covariances_3d[i * 6 + 3],
                    scene.covariances_3d[i * 6 + 4],
                    scene.covariances_3d[i * 6 + 5],
                ];

                let splat = project_ewa_splat(
                    pos,
                    cov,
                    view_mat,
                    camera.focal_x,
                    camera.focal_y,
                    camera.principal_x,
                    camera.principal_y,
                    camera.viewport_width,
                    camera.viewport_height,
                    camera.near_plane,
                );

                pts_chunk[local_i * 2..local_i * 2 + 2].copy_from_slice(&splat.point_xy);
                d_chunk[local_i] = splat.depth;
                *r_val = splat.radius;
                c_chunk[local_i * 3..local_i * 3 + 3].copy_from_slice(&splat.conic);
            }
        });

    ProjectionOutputs {
        points_xy,
        depths,
        radii,
        conics,
        count,
    }
}

/// Dispatches the `project_gaussians_kernel` on the GPU using CubeCL runtime.
///
/// Returns handles to the four GPU output buffers: `(points_xy, depths, radii, conics)`.
pub fn project_scene_cubecl(
    client: &Client,
    positions_handle: &cubecl::server::Handle,
    covariances_handle: &cubecl::server::Handle,
    camera_handle: &cubecl::server::Handle,
    count: usize,
) -> (
    cubecl::server::Handle,
    cubecl::server::Handle,
    cubecl::server::Handle,
    cubecl::server::Handle,
) {
    let num_gaussians = count as u32;

    // Allocate GPU output buffers
    let points_xy_handle = client.empty((count * 2 * std::mem::size_of::<f32>()).max(4));
    let depths_handle = client.empty((count * std::mem::size_of::<f32>()).max(4));
    let radii_handle = client.empty((count * std::mem::size_of::<i32>()).max(4));
    let conics_handle = client.empty((count * 3 * std::mem::size_of::<f32>()).max(4));

    if count == 0 {
        return (points_xy_handle, depths_handle, radii_handle, conics_handle);
    }

    let cube_dim = CubeDim::new_1d(256);
    let num_cubes = num_gaussians.div_ceil(256);
    let cube_count = CubeCount::Static(num_cubes, 1, 1);

    let pos_arg = unsafe { BufferArg::from_raw_parts(positions_handle.clone(), count * 3) };
    let cov_arg = unsafe { BufferArg::from_raw_parts(covariances_handle.clone(), count * 6) };
    let cam_arg = unsafe { BufferArg::from_raw_parts(camera_handle.clone(), 24) };
    let pts_arg = unsafe { BufferArg::from_raw_parts(points_xy_handle.clone(), count * 2) };
    let d_arg = unsafe { BufferArg::from_raw_parts(depths_handle.clone(), count) };
    let r_arg = unsafe { BufferArg::from_raw_parts(radii_handle.clone(), count) };
    let c_arg = unsafe { BufferArg::from_raw_parts(conics_handle.clone(), count * 3) };

    project_gaussians_kernel::launch(
        client,
        cube_count,
        cube_dim,
        pos_arg,
        cov_arg,
        cam_arg,
        pts_arg,
        d_arg,
        r_arg,
        c_arg,
        num_gaussians,
    );

    (points_xy_handle, depths_handle, radii_handle, conics_handle)
}

/// End-to-end GPU projection using CubeCL's WGPU runtime:
/// uploads scene buffers, executes compute kernel, and downloads projection outputs.
pub fn project_scene_wgpu(
    scene: &GaussianSceneSoa,
    camera: &CameraUniforms,
) -> Result<ProjectionOutputs, anyhow::Error> {
    let count = scene.len();
    if count == 0 {
        return Ok(ProjectionOutputs::empty());
    }

    let client = Device::default().client();

    // Upload positions, covariances, and camera uniforms
    let pos_handle = client.create(cubecl_common::bytes::Bytes::from_bytes_vec(scene.positions_bytes().to_vec()));
    let cov_handle = client.create(cubecl_common::bytes::Bytes::from_bytes_vec(scene.covariances_bytes().to_vec()));
    let cam_handle = client.create(cubecl_common::bytes::Bytes::from_bytes_vec(camera.as_bytes().to_vec()));

    // Dispatch kernel
    let (pts_h, d_h, r_h, c_h) = project_scene_cubecl(
        &client,
        &pos_handle,
        &cov_handle,
        &cam_handle,
        count,
    );

    // Read back results
    let pts_bytes = client
        .read_one(pts_h)
        .map_err(|e| anyhow::anyhow!("Failed to read points_xy: {:?}", e))?;
    let d_bytes = client
        .read_one(d_h)
        .map_err(|e| anyhow::anyhow!("Failed to read depths: {:?}", e))?;
    let r_bytes = client
        .read_one(r_h)
        .map_err(|e| anyhow::anyhow!("Failed to read radii: {:?}", e))?;
    let c_bytes = client
        .read_one(c_h)
        .map_err(|e| anyhow::anyhow!("Failed to read conics: {:?}", e))?;

    let pts_slice: &[f32] = bytemuck::cast_slice(&pts_bytes[..count * 2 * 4]);
    let d_slice: &[f32] = bytemuck::cast_slice(&d_bytes[..count * 4]);
    let r_slice: &[i32] = bytemuck::cast_slice(&r_bytes[..count * 4]);
    let c_slice: &[f32] = bytemuck::cast_slice(&c_bytes[..count * 3 * 4]);

    Ok(ProjectionOutputs {
        points_xy: pts_slice.to_vec(),
        depths: d_slice.to_vec(),
        radii: r_slice.to_vec(),
        conics: c_slice.to_vec(),
        count,
    })
}

/// End-to-end CPU rendering pipeline chaining Module 3 projection, Module 4 binning, and Module 5 rasterization.
pub fn render_scene_cpu(
    scene: &GaussianSceneSoa,
    camera: &CameraUniforms,
    uniforms: &CompositorUniforms,
) -> RasterizedImage {
    let proj = project_scene_cpu(scene, camera);
    let binning = proj.bin_and_sort(uniforms.screen_width, uniforms.screen_height);
    rasterize_scene_cpu(
        &proj.points_xy,
        &proj.conics,
        &scene.opacities,
        &scene.sh_coeffs,
        &scene.positions,
        &binning.sorted_entries,
        &binning.tile_ranges,
        uniforms,
    )
}
