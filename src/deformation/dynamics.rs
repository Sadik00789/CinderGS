use glam::Vec3;

/// Rigid planar half-space obstacle enforcing $\mathbf{n} \cdot \mathbf{x} \ge d$.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CollisionPlane {
    /// Unit outward plane normal pointing into the allowed half-space.
    pub normal: Vec3,
    /// Plane signed offset along `normal`.
    pub offset: f32,
    /// Coefficient of restitution in `[0, 1]` along the plane normal.
    pub restitution: f32,
    /// Tangential velocity damping in `[0, 1]` applied on contact.
    pub friction: f32,
}

impl CollisionPlane {
    /// Creates a collision plane from an arbitrary normal, signed offset, restitution, and friction.
    #[inline]
    pub fn new(normal: Vec3, offset: f32, restitution: f32, friction: f32) -> Self {
        let unit = if normal.length_squared() > 1e-8 {
            normal.normalize()
        } else {
            Vec3::Y
        };
        Self {
            normal: unit,
            offset,
            restitution: restitution.clamp(0.0, 1.0),
            friction: friction.clamp(0.0, 1.0),
        }
    }

    /// Creates a horizontal ground plane blocking motion below `y`.
    #[inline]
    pub fn ground_y(y: f32) -> Self {
        Self::new(Vec3::Y, y, 0.25, 0.2)
    }

    /// Signed distance from `point` to the allowed half-space (negative means penetration).
    #[inline]
    pub fn signed_distance(&self, point: Vec3) -> f32 {
        self.normal.dot(point) - self.offset
    }

    /// Projects `point` onto the allowed half-space without changing valid points.
    #[inline]
    pub fn clamp_point(&self, point: Vec3) -> Vec3 {
        let d = self.signed_distance(point);
        if d < 0.0 {
            point - self.normal * d
        } else {
            point
        }
    }

    /// Resolves half-space penetration and reflects/damps incoming velocity.
    pub fn resolve_contact(&self, pos: &mut Vec3, vel: &mut Vec3) -> bool {
        let d = self.signed_distance(*pos);
        if d >= 0.0 {
            return false;
        }
        *pos -= self.normal * d;
        let vn = vel.dot(self.normal);
        if vn < 0.0 {
            let v_norm = self.normal * vn;
            let v_tan = *vel - v_norm;
            *vel = v_tan * (1.0 - self.friction) - v_norm * self.restitution;
        }
        true
    }
}

/// Real-time damped spring physics simulator for interactive volumetric cage elasticity.
///
/// Features:
/// - Timestep clamping to $\Delta t_{\text{sim}} \le 0.033$ to prevent numerical explosion during frame hitches.
/// - Hookean restorative spring force towards rest cage configuration:
///   $$\mathbf{a}_k = \frac{k_{\text{stiffness}}}{m} (\mathbf{X}_{\text{rest}, k} - \mathbf{x}_k)$$
/// - Frame-rate independent exponential velocity damping:
///   $$\gamma(\Delta t) = e^{-c \Delta t}, \qquad c = -\ln(\gamma_{60}) \cdot 60$$
///   where $\gamma_{60}$ is the velocity retention per 1/60 s frame. Stepping at
///   30 Hz, 60 Hz or 144 Hz produces the same physical trajectory.
/// - Transient drag pin (`pinned_index`) and persistent multi-handle pinning (`pinned_vertices`).
/// - Rigid collision plane half-space constraints (`collision_planes`).
/// - Outward radial jiggle impulse perturbation.
#[derive(Clone, Debug, PartialEq)]
pub struct CageSpringSimulator {
    /// Rest configuration of cage vertices in world space.
    pub rest_positions: Vec<Vec3>,
    /// Current deformed positions of cage vertices.
    pub current_positions: Vec<Vec3>,
    /// Instantaneous velocities of cage vertices.
    pub velocities: Vec<Vec3>,
    /// Spring stiffness constant $k$ (default 180.0).
    pub stiffness: f32,
    /// Velocity retention per 1/60 s frame, in (0, 1) (default 0.92).
    pub damping: f32,
    /// Vertex point mass $m$ (default 1.0).
    pub mass: f32,
    /// Currently pinned vertex index (e.g. held by cursor).
    pub pinned_index: Option<usize>,
    /// Persistent multi-handle pin mask per cage vertex.
    pub pinned_vertices: Vec<bool>,
    /// Planar obstacles restricting vertex motion during simulation.
    pub collision_planes: Vec<CollisionPlane>,
}

impl CageSpringSimulator {
    /// Maximum simulation timestep in seconds (30 Hz minimum floor) to avoid instability.
    pub const MAX_DT: f32 = 0.033;

    /// Reference frame rate at which [`Self::damping`] is defined.
    pub const DAMPING_REFERENCE_HZ: f32 = 60.0;

    /// Creates a new spring simulator from rest positions.
    pub fn new(rest_positions: Vec<[f32; 3]>) -> Self {
        let count = rest_positions.len();
        let rest_vec3: Vec<Vec3> = rest_positions.into_iter().map(Vec3::from).collect();
        let current_vec3 = rest_vec3.clone();
        let velocities = vec![Vec3::ZERO; count];

        Self {
            rest_positions: rest_vec3,
            current_positions: current_vec3,
            velocities,
            stiffness: 180.0,
            damping: 0.92,
            mass: 1.0,
            pinned_index: None,
            pinned_vertices: vec![false; count],
            collision_planes: Vec::new(),
        }
    }

    /// Converts the 60 Hz retention ratio into a continuous damping coefficient `c`.
    ///
    /// `gamma = exp(-c * dt)` for any `dt`, so the decay per unit of simulated
    /// time no longer depends on how often [`Self::step`] is called.
    pub fn damping_coefficient(damping: f32) -> f32 {
        let retention = damping.clamp(1.0e-4, 0.9999);
        -retention.ln() * Self::DAMPING_REFERENCE_HZ
    }

    /// Velocity retention factor for a single step of `dt` seconds.
    #[inline]
    pub fn frame_damping(damping: f32, dt: f32) -> f32 {
        (-Self::damping_coefficient(damping) * dt).exp()
    }

    /// Returns true if `index` is fixed either by the active cursor drag or persistent pin mask.
    #[inline]
    pub fn is_vertex_pinned(&self, index: usize) -> bool {
        self.pinned_index == Some(index)
            || self.pinned_vertices.get(index).copied().unwrap_or(false)
    }

    /// Persistently pins cage vertex `index` at its current position.
    #[inline]
    pub fn pin_vertex(&mut self, index: usize) {
        if index < self.pinned_vertices.len() {
            self.pinned_vertices[index] = true;
            self.velocities[index] = Vec3::ZERO;
        }
    }

    /// Removes persistent pin on cage vertex `index`.
    #[inline]
    pub fn unpin_vertex(&mut self, index: usize) {
        if index < self.pinned_vertices.len() {
            self.pinned_vertices[index] = false;
        }
    }

    /// Toggles persistent pin state on cage vertex `index`, returning the new state.
    #[inline]
    pub fn toggle_pin_vertex(&mut self, index: usize) -> bool {
        if let Some(slot) = self.pinned_vertices.get_mut(index) {
            *slot = !*slot;
            if *slot {
                self.velocities[index] = Vec3::ZERO;
            }
            *slot
        } else {
            false
        }
    }

    /// Clears all persistent multi-handle pins.
    #[inline]
    pub fn clear_pins(&mut self) {
        self.pinned_vertices.fill(false);
    }

    /// Counts distinct pinned vertices across both persistent pins and active cursor drag.
    #[inline]
    pub fn pinned_count(&self) -> usize {
        self.pinned_vertices
            .iter()
            .enumerate()
            .filter(|&(i, &p)| p || self.pinned_index == Some(i))
            .count()
    }

    /// Appends a rigid planar obstacle to this simulator.
    #[inline]
    pub fn add_collision_plane(&mut self, plane: CollisionPlane) {
        self.collision_planes.push(plane);
    }

    /// Removes all collision planes.
    #[inline]
    pub fn clear_collision_planes(&mut self) {
        self.collision_planes.clear();
    }

    /// Registered collision planes.
    #[inline]
    pub fn collision_planes(&self) -> &[CollisionPlane] {
        &self.collision_planes
    }

    /// Clamps `point` to the valid side of every configured collision plane.
    pub fn project_point_to_collision_planes(&self, mut point: Vec3) -> Vec3 {
        for plane in &self.collision_planes {
            point = plane.clamp_point(point);
        }
        point
    }

    /// Sets which vertex is currently pinned by user mouse interaction.
    #[inline]
    pub fn set_pinned(&mut self, pinned: Option<usize>) {
        self.pinned_index = pinned;
        if let Some(idx) = pinned.filter(|&i| i < self.velocities.len()) {
            self.velocities[idx] = Vec3::ZERO;
        }
    }

    /// Explicitly updates a pinned vertex position (e.g., from eye-plane dragging).
    #[inline]
    pub fn set_pinned_position(&mut self, index: usize, pos: [f32; 3]) {
        if index < self.current_positions.len() {
            self.current_positions[index] = self.project_point_to_collision_planes(Vec3::from(pos));
            self.velocities[index] = Vec3::ZERO;
        }
    }

    /// Advances the spring simulation by `dt` seconds.
    pub fn step(&mut self, dt: f32) {
        let dt_sim = dt.min(Self::MAX_DT);
        if dt_sim <= 0.0 {
            return;
        }

        let inv_mass = 1.0 / self.mass.max(0.001);
        let k = self.stiffness;
        let gamma = Self::frame_damping(self.damping, dt_sim);

        for i in 0..self.current_positions.len() {
            if self.is_vertex_pinned(i) {
                self.velocities[i] = Vec3::ZERO;
                continue;
            }

            // Hooke's Law: F = -k * (x - x_rest) => a = (x_rest - x) * (k / m)
            let displacement = self.rest_positions[i] - self.current_positions[i];
            let accel = displacement * (k * inv_mass);

            // Semi-implicit Euler integration with frame-rate independent damping
            self.velocities[i] = (self.velocities[i] + accel * dt_sim) * gamma;
            self.current_positions[i] += self.velocities[i] * dt_sim;

            for plane in &self.collision_planes {
                plane.resolve_contact(&mut self.current_positions[i], &mut self.velocities[i]);
            }
        }
    }

    /// Perturbs all unpinned cage vertices with an outward radial velocity pulse.
    pub fn trigger_jiggle_impulse(&mut self, magnitude: f32) {
        if self.current_positions.is_empty() {
            return;
        }

        // Compute centroid of rest cage
        let mut centroid = Vec3::ZERO;
        for pos in &self.rest_positions {
            centroid += *pos;
        }
        centroid /= self.rest_positions.len() as f32;

        for i in 0..self.current_positions.len() {
            if self.is_vertex_pinned(i) {
                continue;
            }

            let radial_dir = (self.rest_positions[i] - centroid).normalize_or_zero();
            let impulse = if radial_dir.length_squared() > 1e-4 {
                radial_dir * magnitude
            } else {
                Vec3::Y * magnitude
            };

            self.velocities[i] += impulse;
        }
    }

    /// Resets all cage vertices to rest positions with zero velocity.
    pub fn reset(&mut self) {
        self.current_positions = self.rest_positions.clone();
        self.velocities.fill(Vec3::ZERO);
        self.pinned_index = None;
    }

    /// Returns current deformed positions as flattened `[f32; 3]` slice.
    pub fn positions_array(&self) -> Vec<[f32; 3]> {
        self.current_positions
            .iter()
            .map(|v| [v.x, v.y, v.z])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spring_convergence_to_rest() {
        let rest = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let mut sim = CageSpringSimulator::new(rest);

        // Perturb vertex 0 by +2.0 in X
        sim.current_positions[0].x = 2.0;

        // Simulate 2.0 seconds at 60 Hz
        for _ in 0..120 {
            sim.step(0.016);
        }

        // Must have converged back to rest position (0, 0, 0)
        let diff = (sim.current_positions[0] - sim.rest_positions[0]).length();
        assert!(
            diff < 0.05,
            "Spring did not converge to rest, residual: {}",
            diff
        );
    }

    #[test]
    fn test_timestep_clamping_prevents_explosion() {
        let rest = vec![[0.0, 0.0, 0.0]];
        let mut sim = CageSpringSimulator::new(rest);
        sim.current_positions[0].x = 1.0;

        // Frame hitch of 2.0 seconds
        sim.step(2.0);

        // Clamping ensures position does not shoot to NaN or infinity
        assert!(!sim.current_positions[0].x.is_nan());
        assert!(sim.current_positions[0].x.abs() < 10.0);
    }

    #[test]
    fn test_pinned_vertex_remains_static() {
        let rest = vec![[0.0, 0.0, 0.0]];
        let mut sim = CageSpringSimulator::new(rest);
        sim.current_positions[0] = Vec3::new(5.0, 5.0, 5.0);
        sim.set_pinned(Some(0));

        sim.step(0.016);
        assert_eq!(sim.current_positions[0], Vec3::new(5.0, 5.0, 5.0));
        assert_eq!(sim.velocities[0], Vec3::ZERO);
    }

    #[test]
    fn test_damping_is_frame_rate_independent() {
        let rest = vec![[0.0, 0.0, 0.0]];

        let mut at_60 = CageSpringSimulator::new(rest.clone());
        let mut at_120 = CageSpringSimulator::new(rest.clone());
        let mut at_30 = CageSpringSimulator::new(rest);

        at_60.current_positions[0].x = 1.0;
        at_120.current_positions[0].x = 1.0;
        at_30.current_positions[0].x = 1.0;

        // One simulated second at three different frame rates.
        for _ in 0..60 {
            at_60.step(1.0 / 60.0);
        }
        for _ in 0..120 {
            at_120.step(1.0 / 120.0);
        }
        for _ in 0..30 {
            at_30.step(1.0 / 30.0);
        }

        let reference = at_60.current_positions[0].x;
        assert!(
            (at_120.current_positions[0].x - reference).abs() < 2e-2,
            "120 Hz trajectory diverged: {} vs {}",
            at_120.current_positions[0].x,
            reference
        );
        assert!(
            (at_30.current_positions[0].x - reference).abs() < 2e-2,
            "30 Hz trajectory diverged: {} vs {}",
            at_30.current_positions[0].x,
            reference
        );
    }

    #[test]
    fn test_damping_coefficient_matches_reference_frame() {
        // A retention factor of 0.92 per 60 Hz frame means one second of
        // simulation retains 0.92^60 of the velocity.
        let coefficient = CageSpringSimulator::damping_coefficient(0.92);
        let one_second = (-coefficient * 1.0).exp();
        let expected = 0.92f32.powi(60);
        assert!(
            (one_second - expected).abs() < 1e-5,
            "expected {expected}, got {one_second}"
        );
    }

    #[test]
    fn test_multi_handle_pinning_and_collision_plane() {
        let rest = vec![[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]];
        let mut sim = CageSpringSimulator::new(rest);
        sim.pin_vertex(0);
        assert!(sim.toggle_pin_vertex(2));
        assert_eq!(sim.pinned_count(), 2);

        // Move vertex 0 upward and vertex 1 below ground plane y = 0.25.
        sim.set_pinned_position(0, [0.0, 2.5, 0.0]);
        sim.current_positions[1] = Vec3::new(1.0, -1.0, 0.0);
        sim.velocities[1] = Vec3::new(0.0, -4.0, 0.0);
        sim.add_collision_plane(CollisionPlane::ground_y(0.25));

        sim.step(0.016);

        // Persistently pinned handle remains fixed while unpinned handle cannot go below y = 0.25.
        assert!((sim.current_positions[0].y - 2.5).abs() < 1e-6);
        assert!(sim.current_positions[1].y >= 0.25 - 1e-5);
        assert!(sim.is_vertex_pinned(2));

        sim.clear_pins();
        assert_eq!(sim.pinned_count(), 0);
    }
}
