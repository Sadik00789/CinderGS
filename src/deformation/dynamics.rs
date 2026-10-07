use glam::Vec3;

/// Real-time damped spring physics simulator for interactive volumetric cage elasticity.
///
/// Features:
/// - Timestep clamping to $\Delta t_{\text{sim}} \le 0.033$ to prevent numerical explosion during frame hitches.
/// - Hookean restorative spring force towards rest cage configuration:
///   $$\mathbf{a}_k = \frac{k_{\text{stiffness}}}{m} (\mathbf{X}_{\text{rest}, k} - \mathbf{x}_k)$$
/// - Velocity damping for smooth settling:
///   $$\mathbf{v}_k^{t+1} = \gamma \cdot (\mathbf{v}_k^t + \mathbf{a}_k \Delta t_{\text{sim}})$$
///   $$\mathbf{x}_k^{t+1} = \mathbf{x}_k^t + \mathbf{v}_k^{t+1} \Delta t_{\text{sim}}$$
/// - Pinning support: Dragged vertex maintains mouse position with zero velocity.
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
    /// Damping coefficient $\gamma \in (0, 1)$ (default 0.92).
    pub damping: f32,
    /// Vertex point mass $m$ (default 1.0).
    pub mass: f32,
    /// Currently pinned vertex index (e.g. held by cursor).
    pub pinned_index: Option<usize>,
}

impl CageSpringSimulator {
    /// Maximum simulation timestep in seconds (30 Hz minimum floor) to avoid instability.
    pub const MAX_DT: f32 = 0.033;

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
        }
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
            self.current_positions[index] = Vec3::from(pos);
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
        let gamma = self.damping.clamp(0.0, 0.999);

        for i in 0..self.current_positions.len() {
            if self.pinned_index == Some(i) {
                self.velocities[i] = Vec3::ZERO;
                continue;
            }

            // Hooke's Law: F = -k * (x - x_rest) => a = (x_rest - x) * (k / m)
            let displacement = self.rest_positions[i] - self.current_positions[i];
            let accel = displacement * (k * inv_mass);

            // Semi-implicit Euler integration
            self.velocities[i] = (self.velocities[i] + accel * dt_sim) * gamma;
            self.current_positions[i] += self.velocities[i] * dt_sim;
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
            if self.pinned_index == Some(i) {
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
        assert!(diff < 0.05, "Spring did not converge to rest, residual: {}", diff);
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
}
