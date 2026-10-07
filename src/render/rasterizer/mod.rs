mod cpu_reference;
pub mod sh;

pub use cpu_reference::*;
pub use sh::*;

use bytemuck::{Pod, Zeroable};

/// Uniform parameters controlling the tile compositor and rasterizer.
///
/// Total size: 48 bytes (aligned to 16 bytes for WebGPU uniform buffers).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct CompositorUniforms {
    /// Target viewport width in pixels.
    pub screen_width: u32,

    /// Target viewport height in pixels.
    pub screen_height: u32,

    /// Horizontal tile count: $\lceil \text{width} / 16.0 \rceil$.
    pub tiles_x: u32,

    /// Vertical tile count: $\lceil \text{height} / 16.0 \rceil$.
    pub tiles_y: u32,

    /// Background color [R, G, B, A] used for unblended transmittance (default black [0, 0, 0, 1]).
    pub background_color: [f32; 4],

    /// World-space camera position used for view-dependent Spherical Harmonics color evaluation.
    pub camera_position: [f32; 3],

    /// Maximum Spherical Harmonics degree to evaluate (0, 1, 2, or 3).
    pub sh_degree: u32,
}

impl CompositorUniforms {
    /// Creates compositor uniforms with default black background and camera position.
    pub fn new(
        screen_width: u32,
        screen_height: u32,
        background_color: [f32; 4],
        camera_position: [f32; 3],
        sh_degree: u32,
    ) -> Self {
        let tiles_x = screen_width.div_ceil(16);
        let tiles_y = screen_height.div_ceil(16);

        Self {
            screen_width,
            screen_height,
            tiles_x,
            tiles_y,
            background_color,
            camera_position,
            sh_degree: sh_degree.min(3),
        }
    }

    /// Default configuration for given resolution.
    pub fn default_with_resolution(width: u32, height: u32) -> Self {
        Self::new(width, height, [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0], 0)
    }

    /// Returns byte slice view for GPU uniform buffer upload.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }

    /// Returns float slice view (12 floats).
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        bytemuck::cast_slice(self.as_bytes())
    }
}

// Compile-time static assertions
const _: () = {
    assert!(std::mem::size_of::<CompositorUniforms>() == 48);
    assert!(std::mem::size_of::<CompositorUniforms>().is_multiple_of(16));
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compositor_uniforms_layout() {
        assert_eq!(std::mem::size_of::<CompositorUniforms>(), 48);
        assert!(std::mem::size_of::<CompositorUniforms>().is_multiple_of(16));

        let uniforms = CompositorUniforms::new(
            800,
            600,
            [0.1, 0.2, 0.3, 1.0],
            [1.0, 2.0, 3.0],
            3,
        );

        assert_eq!(uniforms.tiles_x, 50);
        assert_eq!(uniforms.tiles_y, 38);
        assert_eq!(uniforms.as_slice().len(), 12);
        assert_eq!(uniforms.as_bytes().len(), 48);
    }
}
