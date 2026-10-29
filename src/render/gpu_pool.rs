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

/// Persistent uniform buffers used by the presentation-side render passes.
///
/// The compute stages (deformation, projection, compositing) allocate through
/// CubeCL's memory manager on the shared device, so this pool deliberately holds
/// only the four small uniform buffers that the WGPU passes read. The previous
/// storage-buffer fields were never bound to any pipeline and have been removed.
pub struct GpuBufferPool {
    pub camera_uniform_buffer: wgpu::Buffer,
    pub tile_config_uniform_buffer: wgpu::Buffer,
    pub compositor_uniform_buffer: wgpu::Buffer,
    pub deformation_uniform_buffer: wgpu::Buffer,
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
            camera_uniform_buffer,
            tile_config_uniform_buffer,
            compositor_uniform_buffer,
            deformation_uniform_buffer,
        }
    }

    /// Total bytes of GPU memory held by the pool's uniform buffers.
    pub fn total_bytes(&self) -> u64 {
        self.camera_uniform_buffer.size()
            + self.tile_config_uniform_buffer.size()
            + self.compositor_uniform_buffer.size()
            + self.deformation_uniform_buffer.size()
    }

    /// Uploads camera uniforms to the persistent camera uniform buffer.
    #[inline]
    pub fn update_camera_uniforms(&self, queue: &wgpu::Queue, uniforms: &CameraUniforms) {
        queue.write_buffer(&self.camera_uniform_buffer, 0, bytemuck::bytes_of(uniforms));
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
