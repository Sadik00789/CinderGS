use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

/// Camera uniforms layout matching GPU compute/uniform buffers.
///
/// Total size: 96 bytes (24 floats), aligned to 16 bytes for WGPU compatibility.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct CameraUniforms {
    /// World-to-Camera View Matrix (Column-major 4x4, 16 floats = 64 bytes).
    pub view_matrix: [f32; 16],

    /// Horizontal focal length in pixels.
    pub focal_x: f32,

    /// Vertical focal length in pixels.
    pub focal_y: f32,

    /// Principal point X coordinate in pixels (screen center).
    pub principal_x: f32,

    /// Principal point Y coordinate in pixels (screen center).
    pub principal_y: f32,

    /// Viewport width in pixels.
    pub viewport_width: f32,

    /// Viewport height in pixels.
    pub viewport_height: f32,

    /// Near clipping plane distance (typically 0.2).
    pub near_plane: f32,

    /// Explicit padding to ensure 96-byte (multiple of 16) uniform buffer alignment.
    pub _pad: f32,
}

impl CameraUniforms {
    /// Total byte size of the camera uniform buffer (96 bytes).
    pub const BYTE_SIZE: usize = std::mem::size_of::<Self>();

    /// Total float count (24 floats).
    pub const FLOAT_COUNT: usize = Self::BYTE_SIZE / std::mem::size_of::<f32>();

    /// Creates camera uniforms with explicit parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        view_matrix: Mat4,
        focal_x: f32,
        focal_y: f32,
        principal_x: f32,
        principal_y: f32,
        viewport_width: f32,
        viewport_height: f32,
        near_plane: f32,
    ) -> Self {
        Self {
            view_matrix: view_matrix.to_cols_array(),
            focal_x,
            focal_y,
            principal_x,
            principal_y,
            viewport_width,
            viewport_height,
            near_plane,
            _pad: 0.0,
        }
    }

    /// Constructs camera uniforms from eye, target, up vectors and vertical field-of-view.
    pub fn from_look_at(
        eye: Vec3,
        target: Vec3,
        up: Vec3,
        fov_y_radians: f32,
        viewport_width: f32,
        viewport_height: f32,
        near_plane: f32,
    ) -> Self {
        let view_matrix = Mat4::look_at_lh(eye, target, up);
        let focal_y = 0.5 * viewport_height / (0.5 * fov_y_radians).tan();
        let focal_x = focal_y; // Square pixels assumption
        let principal_x = 0.5 * viewport_width;
        let principal_y = 0.5 * viewport_height;

        Self::new(
            view_matrix,
            focal_x,
            focal_y,
            principal_x,
            principal_y,
            viewport_width,
            viewport_height,
            near_plane,
        )
    }

    /// Returns the view matrix as a `glam::Mat4`.
    #[inline]
    pub fn view_mat4(&self) -> Mat4 {
        Mat4::from_cols_array(&self.view_matrix)
    }

    /// Returns a byte slice for GPU buffer upload.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }

    /// Returns a float slice view (24 floats).
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        bytemuck::cast_slice(self.as_bytes())
    }

    /// Transforms a world-space point into camera space.
    #[inline]
    pub fn world_to_camera(&self, world_point: Vec3) -> Vec3 {
        self.view_mat4().transform_point3(world_point)
    }

    /// Projects a world-space point to 2D pixel coordinates.
    #[inline]
    pub fn project_point(&self, world_point: Vec3) -> Option<[f32; 2]> {
        let cam_pos = self.world_to_camera(world_point);
        if cam_pos.z <= self.near_plane {
            return None;
        }
        let inv_z = 1.0 / cam_pos.z;
        Some([
            self.focal_x * cam_pos.x * inv_z + self.principal_x,
            self.focal_y * cam_pos.y * inv_z + self.principal_y,
        ])
    }
}

// Compile-time static assertions
const _: () = {
    assert!(std::mem::size_of::<CameraUniforms>() == 96);
    assert!(std::mem::size_of::<CameraUniforms>().is_multiple_of(16));
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera_uniforms_layout() {
        assert_eq!(std::mem::size_of::<CameraUniforms>(), 96);
        assert_eq!(CameraUniforms::FLOAT_COUNT, 24);
        assert_eq!(std::mem::size_of::<CameraUniforms>() % 16, 0);

        let uniforms = CameraUniforms::new(
            Mat4::IDENTITY,
            800.0,
            800.0,
            400.0,
            300.0,
            800.0,
            600.0,
            0.2,
        );

        assert_eq!(uniforms.as_slice().len(), 24);
        assert_eq!(uniforms.as_bytes().len(), 96);
        assert_eq!(uniforms.focal_x, 800.0);
        assert_eq!(uniforms.near_plane, 0.2);
    }

    #[test]
    fn test_camera_projection_point() {
        let uniforms = CameraUniforms::new(
            Mat4::IDENTITY,
            1000.0,
            1000.0,
            500.0,
            500.0,
            1000.0,
            1000.0,
            0.2,
        );

        // Point at (1, 2, 2)
        let pt = Vec3::new(1.0, 2.0, 2.0);
        let screen = uniforms.project_point(pt).expect("Should project");
        // x: 1000 * 1 / 2 + 500 = 1000
        // y: 1000 * 2 / 2 + 500 = 1500
        assert_eq!(screen, [1000.0, 1500.0]);

        // Point behind near plane
        let pt_near = Vec3::new(0.0, 0.0, 0.1);
        assert!(uniforms.project_point(pt_near).is_none());
    }
}
