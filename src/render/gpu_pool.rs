use crate::deformation::DeformationUniforms;
use crate::render::camera::CameraUniforms;
use crate::render::rasterizer::CompositorUniforms;
use crate::render::tiles::TileConfigUniforms;


/// Calculates the next capacity with a 1.5x exponential growth factor.
#[inline]
pub fn calculate_exponential_growth(current: usize, needed: usize) -> usize {
    if needed <= current {
        return current;
    }
    let scaled = (current as f64 * 1.5).ceil() as usize;
    scaled.max(needed).max(64)
}

/// Helper that reallocates a GPU buffer with a 1.5x exponential growth factor when needed.
pub fn ensure_buffer_capacity(
    device: &wgpu::Device,
    buffer: &mut Option<wgpu::Buffer>,
    current_capacity: &mut usize,
    required_capacity: usize,
    element_size: usize,
    usage: wgpu::BufferUsages,
    label: &str,
) -> bool {
    if required_capacity <= *current_capacity && buffer.is_some() {
        return false;
    }

    let new_capacity = calculate_exponential_growth(*current_capacity, required_capacity);
    let byte_size = (new_capacity * element_size).max(64) as u64;

    *buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: byte_size,
        usage,
        mapped_at_creation: false,
    }));
    *current_capacity = new_capacity;
    true
}

/// Persistent dynamic GPU buffer pool maintaining storage and uniform buffers across frames
/// with zero allocations in the steady-state render loop.
pub struct GpuBufferPool {
    // 1. Deformed Gaussian SoA buffers
    pub gaussian_capacity: usize,
    pub positions_buffer: Option<wgpu::Buffer>,
    pub covariances_buffer: Option<wgpu::Buffer>,
    pub opacities_buffer: Option<wgpu::Buffer>,
    pub colors_buffer: Option<wgpu::Buffer>,

    // 2. Projection output buffers
    pub points_xy_buffer: Option<wgpu::Buffer>,
    pub depths_buffer: Option<wgpu::Buffer>,
    pub radii_buffer: Option<wgpu::Buffer>,
    pub conics_buffer: Option<wgpu::Buffer>,

    // 3. Tile binning buffers
    pub tiles_per_gaussian_buffer: Option<wgpu::Buffer>,

    pub entries_capacity: usize,
    pub sort_entries_buffer: Option<wgpu::Buffer>,

    pub tiles_capacity: usize,
    pub tile_ranges_buffer: Option<wgpu::Buffer>,

    // 4. Uniform buffers
    pub camera_uniform_buffer: wgpu::Buffer,
    pub tile_config_uniform_buffer: wgpu::Buffer,
    pub compositor_uniform_buffer: wgpu::Buffer,
    pub deformation_uniform_buffer: wgpu::Buffer,

    // 5. Volumetric cage deformation buffers
    pub cage_vertices_capacity: usize,
    pub cage_vertices_buffer: Option<wgpu::Buffer>,

    pub cage_tets_capacity: usize,
    pub tet_elements_buffer: Option<wgpu::Buffer>,
    pub tet_inv_dm_buffer: Option<wgpu::Buffer>,
    pub tet_jacobians_buffer: Option<wgpu::Buffer>,
    pub tet_rotations_buffer: Option<wgpu::Buffer>,

    pub bindings_tet_buffer: Option<wgpu::Buffer>,
    pub bindings_weights_buffer: Option<wgpu::Buffer>,
}

impl GpuBufferPool {
    /// Creates a new GPU buffer pool with uniform buffers allocated.
    pub fn new(device: &wgpu::Device) -> Self {
        let uniform_usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;


        let camera_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CameraUniformBuffer"),
            size: std::mem::size_of::<CameraUniforms>() as u64,
            usage: uniform_usage,
            mapped_at_creation: false,
        });

        let tile_config_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TileConfigUniformBuffer"),
            size: std::mem::size_of::<TileConfigUniforms>() as u64,
            usage: uniform_usage,
            mapped_at_creation: false,
        });

        let compositor_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CompositorUniformBuffer"),
            size: std::mem::size_of::<CompositorUniforms>() as u64,
            usage: uniform_usage,
            mapped_at_creation: false,
        });

        let deformation_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DeformationUniformBuffer"),
            size: std::mem::size_of::<DeformationUniforms>() as u64,
            usage: uniform_usage,
            mapped_at_creation: false,
        });

        Self {
            gaussian_capacity: 0,
            positions_buffer: None,
            covariances_buffer: None,
            opacities_buffer: None,
            colors_buffer: None,

            points_xy_buffer: None,
            depths_buffer: None,
            radii_buffer: None,
            conics_buffer: None,

            tiles_per_gaussian_buffer: None,

            entries_capacity: 0,
            sort_entries_buffer: None,

            tiles_capacity: 0,
            tile_ranges_buffer: None,

            camera_uniform_buffer,
            tile_config_uniform_buffer,
            compositor_uniform_buffer,
            deformation_uniform_buffer,

            cage_vertices_capacity: 0,
            cage_vertices_buffer: None,

            cage_tets_capacity: 0,
            tet_elements_buffer: None,
            tet_inv_dm_buffer: None,
            tet_jacobians_buffer: None,
            tet_rotations_buffer: None,

            bindings_tet_buffer: None,
            bindings_weights_buffer: None,
        }
    }

    /// Ensures all per-Gaussian buffers can hold at least `count` Gaussians.
    pub fn ensure_gaussian_capacity(&mut self, device: &wgpu::Device, count: usize) -> bool {
        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;

        let mut reallocated = false;
        if count > self.gaussian_capacity || self.positions_buffer.is_none() {
            let new_cap = calculate_exponential_growth(self.gaussian_capacity, count);

            ensure_buffer_capacity(
                device,
                &mut self.positions_buffer,
                &mut self.gaussian_capacity,
                new_cap,
                3 * std::mem::size_of::<f32>(),
                storage,
                "PositionsBuffer",
            );

            let mut cap = self.gaussian_capacity;
            ensure_buffer_capacity(
                device,
                &mut self.covariances_buffer,
                &mut cap,
                new_cap,
                6 * std::mem::size_of::<f32>(),
                storage,
                "CovariancesBuffer",
            );

            ensure_buffer_capacity(
                device,
                &mut self.opacities_buffer,
                &mut cap,
                new_cap,
                std::mem::size_of::<f32>(),
                storage,
                "OpacitiesBuffer",
            );

            ensure_buffer_capacity(
                device,
                &mut self.colors_buffer,
                &mut cap,
                new_cap,
                3 * std::mem::size_of::<f32>(),
                storage,
                "ColorsBuffer",
            );

            ensure_buffer_capacity(
                device,
                &mut self.points_xy_buffer,
                &mut cap,
                new_cap,
                2 * std::mem::size_of::<f32>(),
                storage,
                "PointsXyBuffer",
            );

            ensure_buffer_capacity(
                device,
                &mut self.depths_buffer,
                &mut cap,
                new_cap,
                std::mem::size_of::<f32>(),
                storage,
                "DepthsBuffer",
            );

            ensure_buffer_capacity(
                device,
                &mut self.radii_buffer,
                &mut cap,
                new_cap,
                std::mem::size_of::<i32>(),
                storage,
                "RadiiBuffer",
            );

            ensure_buffer_capacity(
                device,
                &mut self.conics_buffer,
                &mut cap,
                new_cap,
                3 * std::mem::size_of::<f32>(),
                storage,
                "ConicsBuffer",
            );

            ensure_buffer_capacity(
                device,
                &mut self.tiles_per_gaussian_buffer,
                &mut cap,
                new_cap,
                std::mem::size_of::<u32>(),
                storage,
                "TilesPerGaussianBuffer",
            );

            ensure_buffer_capacity(
                device,
                &mut self.bindings_tet_buffer,
                &mut cap,
                new_cap,
                std::mem::size_of::<u32>(),
                storage,
                "BindingsTetBuffer",
            );

            ensure_buffer_capacity(
                device,
                &mut self.bindings_weights_buffer,
                &mut cap,
                new_cap,
                4 * std::mem::size_of::<f32>(),
                storage,
                "BindingsWeightsBuffer",
            );

            self.gaussian_capacity = new_cap;
            reallocated = true;
        }

        reallocated
    }

    /// Ensures the tile sort entries buffer can hold at least `count` entries.
    pub fn ensure_entries_capacity(&mut self, device: &wgpu::Device, count: usize) -> bool {
        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;

        ensure_buffer_capacity(
            device,
            &mut self.sort_entries_buffer,
            &mut self.entries_capacity,
            count,
            3 * std::mem::size_of::<u32>(), // [tile_id, depth_key, gaussian_id]
            storage,
            "SortEntriesBuffer",
        )
    }

    /// Ensures the tile boundary ranges buffer can hold at least `num_tiles` ranges.
    pub fn ensure_tiles_capacity(&mut self, device: &wgpu::Device, num_tiles: usize) -> bool {
        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;

        ensure_buffer_capacity(
            device,
            &mut self.tile_ranges_buffer,
            &mut self.tiles_capacity,
            num_tiles,
            2 * std::mem::size_of::<u32>(), // [start, end]
            storage,
            "TileRangesBuffer",
        )
    }

    /// Ensures cage vertices and tetrahedra buffers can hold the given mesh sizes.
    pub fn ensure_cage_capacity(
        &mut self,
        device: &wgpu::Device,
        num_vertices: usize,
        num_tets: usize,
    ) -> bool {
        let storage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;

        let mut realloc = false;
        if ensure_buffer_capacity(
            device,
            &mut self.cage_vertices_buffer,
            &mut self.cage_vertices_capacity,
            num_vertices,
            3 * std::mem::size_of::<f32>(),
            storage,
            "CageVerticesBuffer",
        ) {
            realloc = true;
        }

        if num_tets > self.cage_tets_capacity || self.tet_elements_buffer.is_none() {
            let new_tets = calculate_exponential_growth(self.cage_tets_capacity, num_tets);
            let mut cap = self.cage_tets_capacity;

            ensure_buffer_capacity(
                device,
                &mut self.tet_elements_buffer,
                &mut cap,
                new_tets,
                4 * std::mem::size_of::<u32>(),
                storage,
                "TetElementsBuffer",
            );

            ensure_buffer_capacity(
                device,
                &mut self.tet_inv_dm_buffer,
                &mut cap,
                new_tets,
                9 * std::mem::size_of::<f32>(),
                storage,
                "TetInvDmBuffer",
            );

            ensure_buffer_capacity(
                device,
                &mut self.tet_jacobians_buffer,
                &mut cap,
                new_tets,
                9 * std::mem::size_of::<f32>(),
                storage,
                "TetJacobiansBuffer",
            );

            ensure_buffer_capacity(
                device,
                &mut self.tet_rotations_buffer,
                &mut cap,
                new_tets,
                9 * std::mem::size_of::<f32>(),
                storage,
                "TetRotationsBuffer",
            );

            self.cage_tets_capacity = new_tets;
            realloc = true;
        }

        realloc
    }

    /// Uploads camera uniforms to the persistent camera uniform buffer.
    #[inline]
    pub fn update_camera_uniforms(&self, queue: &wgpu::Queue, uniforms: &CameraUniforms) {
        queue.write_buffer(
            &self.camera_uniform_buffer,
            0,
            bytemuck::bytes_of(uniforms),
        );
    }

    /// Uploads tile config uniforms to the persistent tile config uniform buffer.
    #[inline]
    pub fn update_tile_config_uniforms(&self, queue: &wgpu::Queue, uniforms: &TileConfigUniforms) {
        queue.write_buffer(
            &self.tile_config_uniform_buffer,
            0,
            bytemuck::bytes_of(uniforms),
        );
    }

    /// Uploads compositor uniforms to the persistent compositor uniform buffer.
    #[inline]
    pub fn update_compositor_uniforms(&self, queue: &wgpu::Queue, uniforms: &CompositorUniforms) {
        queue.write_buffer(
            &self.compositor_uniform_buffer,
            0,
            bytemuck::bytes_of(uniforms),
        );
    }

    /// Uploads deformation uniforms to the persistent deformation uniform buffer.
    #[inline]
    pub fn update_deformation_uniforms(&self, queue: &wgpu::Queue, uniforms: &DeformationUniforms) {
        queue.write_buffer(
            &self.deformation_uniform_buffer,
            0,
            bytemuck::bytes_of(uniforms),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_growth_factor() {
        assert_eq!(calculate_exponential_growth(0, 10), 64);
        assert_eq!(calculate_exponential_growth(100, 100), 100);
        assert_eq!(calculate_exponential_growth(100, 120), 150); // 100 * 1.5 = 150
        assert_eq!(calculate_exponential_growth(100, 250), 250); // needed exceeds 1.5x, picks needed
    }
}
