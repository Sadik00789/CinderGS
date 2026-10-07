use crate::render::camera::CameraUniforms;
use glam::Vec3;

/// Maximum screen-space Euclidean pixel distance to register a vertex pick.
pub const PICKING_RADIUS_PIXELS: f32 = 12.0;

/// Screen-space cage vertex picker and eye-plane dragging controller.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CagePicker {
    /// Currently selected cage vertex index, if any.
    pub selected_vertex: Option<usize>,
    /// Whether an active dragging interaction is occurring.
    pub is_dragging: bool,
    /// Last recorded mouse position [x, y] in screen pixels.
    pub last_mouse_pos: Option<[f32; 2]>,
}

impl CagePicker {
    /// Creates a new cage picker with no selection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Projects all cage vertices into screen coordinates and picks the closest vertex
    /// within `PICKING_RADIUS_PIXELS` (12.0 px) of `mouse_pos`.
    pub fn pick_vertex(
        &mut self,
        mouse_pos: [f32; 2],
        cage_vertices: &[[f32; 3]],
        camera: &CameraUniforms,
    ) -> Option<usize> {
        let mut closest_idx = None;
        let mut min_dist_sq = PICKING_RADIUS_PIXELS * PICKING_RADIUS_PIXELS;

        for (i, &v_arr) in cage_vertices.iter().enumerate() {
            let v_world = Vec3::from(v_arr);
            let t = camera.world_to_camera(v_world);

            // Ignore vertices behind or at near clipping plane
            if t.z <= camera.near_plane {
                continue;
            }

            let sx = camera.focal_x * t.x / t.z + camera.principal_x;
            let sy = camera.focal_y * t.y / t.z + camera.principal_y;

            let dx = sx - mouse_pos[0];
            let dy = sy - mouse_pos[1];
            let dist_sq = dx * dx + dy * dy;

            if dist_sq < min_dist_sq {
                min_dist_sq = dist_sq;
                closest_idx = Some(i);
            }
        }

        self.selected_vertex = closest_idx;
        closest_idx
    }

    /// Computes eye-plane translation delta and updates the selected cage vertex position.
    ///
    /// Displacement formula along camera-space plane at vertex depth $t_z$:
    /// $$\Delta \mathbf{X}_{\text{world}} = \left(\frac{\Delta x}{f_x} \cdot t_z\right) \mathbf{u} - \left(\frac{\Delta y}{f_y} \cdot t_z\right) \mathbf{v}$$
    /// where $\mathbf{u}$ (right) and $\mathbf{v}$ (up) are the camera world-space unit axes.
    ///
    /// Returns `true` if the vertex was updated.
    pub fn drag_selected(
        &self,
        delta: [f32; 2],
        cage_vertices: &mut [[f32; 3]],
        camera: &CameraUniforms,
    ) -> bool {
        let Some(idx) = self.selected_vertex else {
            return false;
        };

        if idx >= cage_vertices.len() {
            return false;
        }

        let cur_pos = Vec3::from(cage_vertices[idx]);
        let t = camera.world_to_camera(cur_pos);

        // Cannot translate vertices behind near plane
        if t.z <= camera.near_plane {
            return false;
        }

        let t_z = t.z;

        // Camera world-space axes extracted from view matrix rows
        let w = camera.view_matrix;
        let u = Vec3::new(w[0], w[4], w[8]).normalize_or_zero(); // Camera Right
        let v = Vec3::new(w[1], w[5], w[9]).normalize_or_zero(); // Camera Up

        let dx_world = (delta[0] / camera.focal_x * t_z) * u;
        let dy_world = (delta[1] / camera.focal_y * t_z) * v;

        let delta_world = dx_world - dy_world;
        let new_pos = cur_pos + delta_world;

        cage_vertices[idx] = [new_pos.x, new_pos.y, new_pos.z];
        true
    }

    /// Clears any selected vertex and ends active drag.
    pub fn clear_selection(&mut self) {
        self.selected_vertex = None;
        self.is_dragging = false;
        self.last_mouse_pos = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Mat4;

    #[test]
    fn test_picker_selection_distance() {
        let camera = CameraUniforms::new(
            Mat4::IDENTITY,
            1000.0,
            1000.0,
            500.0,
            500.0,
            1000.0,
            1000.0,
            0.2,
        );

        // Vertex at (0, 0, 2.0) projects to screen center (500, 500)
        let vertices = vec![[0.0, 0.0, 2.0], [1.0, 1.0, 2.0]];
        let mut picker = CagePicker::new();

        // Mouse within 5 pixels of (500, 500)
        let picked = picker.pick_vertex([503.0, 504.0], &vertices, &camera);
        assert_eq!(picked, Some(0));

        // Mouse far away at (0, 0)
        let picked_far = picker.pick_vertex([0.0, 0.0], &vertices, &camera);
        assert_eq!(picked_far, None);
    }

    #[test]
    fn test_picker_eye_plane_drag_math() {
        let camera = CameraUniforms::new(
            Mat4::IDENTITY,
            1000.0,
            1000.0,
            500.0,
            500.0,
            1000.0,
            1000.0,
            0.2,
        );

        let mut vertices = vec![[0.0, 0.0, 2.0]];
        let mut picker = CagePicker::new();
        picker.selected_vertex = Some(0);

        // Drag 100 pixels in X: delta_X = (100 / 1000) * 2.0 = 0.2
        let changed = picker.drag_selected([100.0, 0.0], &mut vertices, &camera);
        assert!(changed);
        assert!((vertices[0][0] - 0.2).abs() < 1e-5);
        assert!((vertices[0][1] - 0.0).abs() < 1e-5);
        assert!((vertices[0][2] - 2.0).abs() < 1e-5);
    }
}
