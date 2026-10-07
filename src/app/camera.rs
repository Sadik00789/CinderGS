use crate::render::camera::CameraUniforms;
use glam::{Mat4, Vec3};
use std::f32::consts::{FRAC_PI_2, PI};

/// Interactive Orbit Camera Controller.
///
/// Features:
/// - Spherical orbit around pivot target $\mathbf{c}_{\text{target}}$ (Right Mouse Button Drag)
/// - Camera-plane sensor panning (Middle Mouse Button Drag)
/// - Proportional distance zooming (Scroll Wheel)
/// - Pitch clamping to avoid gimbal lock: $[-\frac{\pi}{2} + 0.01, \frac{\pi}{2} - 0.01]$
#[derive(Clone, Debug, PartialEq)]
pub struct OrbitCamera {
    /// Pivot target position in world space.
    pub target: Vec3,
    /// Distance from pivot target to eye.
    pub distance: f32,
    /// Horizontal yaw angle in radians.
    pub yaw: f32,
    /// Vertical pitch angle in radians.
    pub pitch: f32,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Near clipping plane distance.
    pub near_plane: f32,
    /// Mouse orbit sensitivity (radians per pixel).
    pub orbit_sensitivity: f32,
    /// Mouse pan sensitivity (world units per pixel).
    pub pan_sensitivity: f32,
    /// Scroll zoom sensitivity factor.
    pub zoom_sensitivity: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 4.0,
            yaw: 0.0,
            pitch: 0.2,
            fov_y: 45.0f32.to_radians(),
            near_plane: 0.2,
            orbit_sensitivity: 0.005,
            pan_sensitivity: 0.003,
            zoom_sensitivity: 0.15,
        }
    }
}

impl OrbitCamera {
    /// Constructs an orbit camera looking at `target` from `distance`.
    pub fn new(target: Vec3, distance: f32) -> Self {
        Self {
            target,
            distance,
            ..Default::default()
        }
    }

    /// Pitch lower bound to avoid gimbal lock.
    pub const PITCH_MIN: f32 = -FRAC_PI_2 + 0.01;
    /// Pitch upper bound to avoid gimbal lock.
    pub const PITCH_MAX: f32 = FRAC_PI_2 - 0.01;

    /// Computes the camera eye position in world space.
    pub fn eye_position(&self) -> Vec3 {
        let cos_pitch = self.pitch.cos();
        let sin_pitch = self.pitch.sin();
        let cos_yaw = self.yaw.cos();
        let sin_yaw = self.yaw.sin();

        let offset = Vec3::new(
            self.distance * cos_pitch * sin_yaw,
            self.distance * sin_pitch,
            self.distance * cos_pitch * cos_yaw,
        );

        self.target + offset
    }

    /// Computes the Right-Handed World-to-Camera View Matrix.
    pub fn view_matrix(&self) -> Mat4 {
        let eye = self.eye_position();
        Mat4::look_at_rh(eye, self.target, Vec3::Y)
    }

    /// Computes the camera world-space unit axes: (right $\mathbf{u}$, up $\mathbf{v}$, forward $\mathbf{f}$).
    pub fn camera_axes(&self) -> (Vec3, Vec3, Vec3) {
        let eye = self.eye_position();
        let forward = (self.target - eye).normalize_or_zero();
        let right = forward.cross(Vec3::Y).normalize_or_zero();
        let up = right.cross(forward).normalize_or_zero();
        (right, up, forward)
    }

    /// Processes mouse orbit rotation delta $(\Delta x, \Delta y)$ in pixels.
    pub fn on_mouse_orbit(&mut self, delta_x: f32, delta_y: f32) {
        self.yaw -= delta_x * self.orbit_sensitivity;
        // Wrap yaw to [-PI, PI] to prevent float overflow
        while self.yaw > PI {
            self.yaw -= 2.0 * PI;
        }
        while self.yaw < -PI {
            self.yaw += 2.0 * PI;
        }

        self.pitch = (self.pitch + delta_y * self.orbit_sensitivity).clamp(Self::PITCH_MIN, Self::PITCH_MAX);
    }

    /// Processes mouse panning delta $(\Delta x, \Delta y)$ along camera view plane.
    pub fn on_mouse_pan(&mut self, delta_x: f32, delta_y: f32) {
        let (right, up, _) = self.camera_axes();
        let speed = self.pan_sensitivity * self.distance * 0.25;
        self.target -= right * (delta_x * speed);
        self.target += up * (delta_y * speed);
    }

    /// Processes scroll wheel zoom delta.
    pub fn on_scroll(&mut self, delta_y: f32) {
        let zoom_factor = 1.0 - delta_y * self.zoom_sensitivity * 0.1;
        self.distance = (self.distance * zoom_factor).clamp(0.05, 5000.0);
    }

    /// Constructs standard GPU `CameraUniforms` matching viewport dimensions.
    pub fn build_camera_uniforms(&self, width: f32, height: f32) -> CameraUniforms {
        let view_matrix = self.view_matrix();
        let focal_y = 0.5 * height / (0.5 * self.fov_y).tan();
        let focal_x = focal_y;
        let principal_x = 0.5 * width;
        let principal_y = 0.5 * height;

        CameraUniforms::new(
            view_matrix,
            focal_x,
            focal_y,
            principal_x,
            principal_y,
            width,
            height,
            self.near_plane,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orbit_camera_clamping() {
        let mut cam = OrbitCamera::default();
        cam.on_mouse_orbit(0.0, 10000.0);
        assert!(cam.pitch <= OrbitCamera::PITCH_MAX);

        cam.on_mouse_orbit(0.0, -20000.0);
        assert!(cam.pitch >= OrbitCamera::PITCH_MIN);
    }

    #[test]
    fn test_orbit_camera_distance_zoom() {
        let mut cam = OrbitCamera::default();
        let init_dist = cam.distance;
        cam.on_scroll(10.0);
        assert!(cam.distance < init_dist);

        cam.on_scroll(-100.0);
        assert!(cam.distance > init_dist);
    }
}
