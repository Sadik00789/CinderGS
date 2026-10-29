pub mod blit;
pub mod camera;
pub mod camera_path;
pub mod gpu;
pub mod gpu_bridge;
pub mod gpu_pool;
pub mod headless;
pub mod kernels;
pub mod pipeline;
pub mod rasterizer;
pub mod tiles;
pub mod wireframe;

pub use blit::*;
pub use camera::*;
pub use camera_path::*;
pub use gpu::*;
pub use gpu_bridge::*;
pub use gpu_pool::*;
pub use headless::*;
pub use kernels::*;
pub use pipeline::*;
pub use rasterizer::*;
pub use tiles::*;
pub use wireframe::*;

use crate::math::project_ewa_splat;
use crate::parallel;
use crate::scene::GaussianSceneSoa;
use glam::Vec3;

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

    /// Executes tile binning, scattering, and depth sorting on the CPU.
    pub fn bin_and_sort(&self, screen_width: u32, screen_height: u32) -> TileBinningOutputs {
        let config = TileConfigUniforms::new(screen_width, screen_height, self.count as u32);
        bin_and_sort_gaussians_cpu(&self.points_xy, &self.depths, &self.radii, config)
    }
}

/// Number of splats processed per parallel chunk during projection.
const PROJECT_CHUNK_SIZE: usize = 2048;

/// Dispatches the EWA projection pipeline on CPU using the parallel shim.
pub fn project_scene_cpu(scene: &GaussianSceneSoa, camera: &CameraUniforms) -> ProjectionOutputs {
    let count = scene.len();
    if count == 0 {
        return ProjectionOutputs::empty();
    }

    let mut points_xy = vec![0.0f32; count * 2];
    let mut depths = vec![0.0f32; count];
    let mut radii = vec![0i32; count];
    let mut conics = vec![0.0f32; count * 3];

    let view_mat = camera.view_mat4();

    parallel::zip_fill_4(
        &mut points_xy,
        &mut depths,
        &mut radii,
        &mut conics,
        [
            PROJECT_CHUNK_SIZE * 2,
            PROJECT_CHUNK_SIZE,
            PROJECT_CHUNK_SIZE,
            PROJECT_CHUNK_SIZE * 3,
        ],
        |base_i, pts_chunk, d_chunk, r_chunk, c_chunk| {
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
        },
    );

    ProjectionOutputs {
        points_xy,
        depths,
        radii,
        conics,
        count,
    }
}

/// End-to-end CPU rendering pipeline chaining projection, binning, and rasterization.
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
