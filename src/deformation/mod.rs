pub mod cpu_reference;
pub mod dynamics;
pub mod polar;

pub use cpu_reference::*;
pub use dynamics::*;
pub use polar::*;

use crate::parallel;
use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Vec3};

/// Indices of the 4 cage vertices defining a single tetrahedron.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct TetElement {
    pub indices: [u32; 4], // Vertex indices into cage positions
}

impl TetElement {
    #[inline]
    pub const fn new(v0: u32, v1: u32, v2: u32, v3: u32) -> Self {
        Self {
            indices: [v0, v1, v2, v3],
        }
    }
}

/// Barycentric binding mapping a single Gaussian to its enclosing tetrahedron.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct GaussianBinding {
    pub tet_index: u32,    // Parent tetrahedron index
    pub weights: [f32; 4], // Barycentric weights (w0, w1, w2, w3)
}

impl GaussianBinding {
    /// Sentinel tetrahedron index indicating an unbound, static Gaussian (outside cage).
    pub const UNBOUND: u32 = u32::MAX;

    #[inline]
    pub const fn new(tet_index: u32, weights: [f32; 4]) -> Self {
        Self { tet_index, weights }
    }

    /// Constructs an unbound binding with zero weights and `tet_index = u32::MAX`.
    #[inline]
    pub const fn unbound() -> Self {
        Self {
            tet_index: Self::UNBOUND,
            weights: [0.0; 4],
        }
    }

    /// Returns true if this Gaussian is bound to an active cage tetrahedron.
    #[inline]
    pub fn is_bound(&self) -> bool {
        self.tet_index != Self::UNBOUND
    }
}

/// Binding of a Gaussian to one of several independent cages.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct CageBinding {
    /// Index of the owning cage, or [`Self::UNBOUND`].
    pub cage: u32,
    /// Tetrahedron index inside that cage.
    pub tet: u32,
    /// Barycentric weights `(w0, w1, w2, w3)`.
    pub weights: [f32; 4],
}

impl CageBinding {
    /// Sentinel cage index for a Gaussian that no cage deforms.
    pub const UNBOUND: u32 = u32::MAX;

    /// Constructs an unbound binding.
    #[inline]
    pub const fn unbound() -> Self {
        Self {
            cage: Self::UNBOUND,
            tet: 0,
            weights: [0.0; 4],
        }
    }

    /// Constructs a binding to `cage`'s tetrahedron `tet`.
    #[inline]
    pub const fn new(cage: u32, tet: u32, weights: [f32; 4]) -> Self {
        Self { cage, tet, weights }
    }

    /// Returns true if a cage owns this Gaussian.
    #[inline]
    pub fn is_bound(&self) -> bool {
        self.cage != Self::UNBOUND
    }
}

/// Precomputed inverse rest shape matrix $D_m^{-1}$ per tetrahedron.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct TetPrecomputed {
    pub inv_dm: [f32; 9], // Inverse rest shape matrix (column-major 3x3)
}

impl TetPrecomputed {
    #[inline]
    pub const fn new(inv_dm: [f32; 9]) -> Self {
        Self { inv_dm }
    }

    #[inline]
    pub fn matrix(&self) -> Mat3 {
        Mat3::from_cols_array(&self.inv_dm)
    }
}

/// Uniforms buffer for deformation and SH rotation.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct DeformationUniforms {
    /// Legacy camera position slot (kept for buffer layout stability).
    pub camera_position: [f32; 3],
    /// Maximum SH degree whose coefficients are rotated under deformation.
    pub sh_degree: u32,
    pub num_gaussians: u32,
    pub num_tets: u32,
    pub _pad: [f32; 2],
}

impl DeformationUniforms {
    pub const BYTE_SIZE: usize = std::mem::size_of::<Self>();

    #[inline]
    pub fn new(
        camera_position: [f32; 3],
        sh_degree: u32,
        num_gaussians: u32,
        num_tets: u32,
    ) -> Self {
        Self {
            camera_position,
            sh_degree,
            num_gaussians,
            num_tets,
            _pad: [0.0; 2],
        }
    }
}

/// Volumetric tetrahedral cage mesh containing rest vertices, topology, and precomputed $D_m^{-1}$.
#[derive(Clone, Debug, PartialEq)]
pub struct TetMesh {
    /// Rest positions of cage vertices: `[V]` points in 3D.
    pub rest_vertices: Vec<[f32; 3]>,
    /// Connectivity of tetrahedra: `[T]` elements.
    pub elements: Vec<TetElement>,
    /// Precomputed rest shape inverse matrices $D_m^{-1}$: `[T]` elements.
    pub precomputed: Vec<TetPrecomputed>,
}

impl TetMesh {
    /// Constructs a tetrahedral mesh and precomputes $D_m^{-1}$ for every tetrahedron.
    pub fn new(rest_vertices: Vec<[f32; 3]>, elements: Vec<TetElement>) -> Self {
        let mut precomputed = Vec::with_capacity(elements.len());

        for elem in &elements {
            let x0 = Vec3::from(rest_vertices[elem.indices[0] as usize]);
            let x1 = Vec3::from(rest_vertices[elem.indices[1] as usize]);
            let x2 = Vec3::from(rest_vertices[elem.indices[2] as usize]);
            let x3 = Vec3::from(rest_vertices[elem.indices[3] as usize]);

            // D_m = [X1 - X0, X2 - X0, X3 - X0] (column-major)
            let dm = Mat3::from_cols(x1 - x0, x2 - x0, x3 - x0);
            let inv_dm = if dm.determinant().abs() > 1e-8 {
                dm.inverse()
            } else {
                Mat3::IDENTITY
            };

            precomputed.push(TetPrecomputed::new(inv_dm.to_cols_array()));
        }

        Self {
            rest_vertices,
            elements,
            precomputed,
        }
    }

    /// Number of vertices in the cage.
    #[inline]
    pub fn num_vertices(&self) -> usize {
        self.rest_vertices.len()
    }

    /// Number of tetrahedra in the cage.
    #[inline]
    pub fn num_tets(&self) -> usize {
        self.elements.len()
    }

    /// Axis-aligned bounds of the rest cage.
    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        if self.rest_vertices.is_empty() {
            return ([-0.5; 3], [0.5; 3]);
        }
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for v in &self.rest_vertices {
            let p = Vec3::from(*v);
            min = min.min(p);
            max = max.max(p);
        }
        // Small padding so surface splats are not excluded by float rounding.
        let pad = (max - min).max(Vec3::splat(1e-3)) * 1e-3;
        ((min - pad).to_array(), (max + pad).to_array())
    }

    /// Computes barycentric binding coordinates for an array of 3D points.
    ///
    /// For each point $\mathbf{p}$:
    /// 1. Finds the tetrahedron where $w_k \ge -1e-4$ for all $k \in \{0, 1, 2, 3\}$.
    /// 2. If the point lies outside all tetrahedra, selects the tetrahedron minimizing the barycentric violation penalty.
    pub fn bind_gaussians(&self, positions: &[[f32; 3]]) -> Vec<GaussianBinding> {
        parallel::map_slice(positions, |&p_arr| self.bind_point(p_arr, None))
    }

    fn bind_point(&self, p_arr: [f32; 3], bounds: Option<([f32; 3], [f32; 3])>) -> GaussianBinding {
        if let Some((min, max)) = bounds {
            let outside = (0..3).any(|axis| p_arr[axis] < min[axis] || p_arr[axis] > max[axis]);
            if outside {
                return GaussianBinding::unbound();
            }
        }

        let p = Vec3::from(p_arr);
        let mut best_tet = 0u32;
        let mut best_penalty = f32::INFINITY;
        let mut best_weights = [1.0f32, 0.0, 0.0, 0.0];

        for (t_idx, elem) in self.elements.iter().enumerate() {
            let x0 = Vec3::from(self.rest_vertices[elem.indices[0] as usize]);
            let inv_dm = self.precomputed[t_idx].matrix();

            // [w1, w2, w3]^T = D_m^{-1} * (p - X0)
            let delta = p - x0;
            let w123 = inv_dm * delta;
            let w1 = w123.x;
            let w2 = w123.y;
            let w3 = w123.z;
            let w0 = 1.0 - (w1 + w2 + w3);

            let penalty = (-w0).max(0.0) + (-w1).max(0.0) + (-w2).max(0.0) + (-w3).max(0.0);

            if penalty <= 1e-4 {
                return GaussianBinding::new(t_idx as u32, [w0, w1, w2, w3]);
            }

            if penalty < best_penalty {
                best_penalty = penalty;
                best_tet = t_idx as u32;
                best_weights = [w0, w1, w2, w3];
            }
        }

        if bounds.is_some() && best_penalty > 0.15 {
            GaussianBinding::unbound()
        } else {
            GaussianBinding::new(best_tet, best_weights)
        }
    }

    /// Computes barycentric binding coordinates, restricting active deformation strictly
    /// to Gaussians within `[hero_min, hero_max]`.
    /// Any Gaussian outside hero bounds receives `tet_index = u32::MAX` and zero weights.
    pub fn bind_gaussians_with_bounds(
        &self,
        positions: &[[f32; 3]],
        hero_min: [f32; 3],
        hero_max: [f32; 3],
    ) -> Vec<GaussianBinding> {
        parallel::map_slice(positions, |&p_arr| {
            self.bind_point(p_arr, Some((hero_min, hero_max)))
        })
    }

    /// Computes barycentric binding coordinates for flattened `[N * 3]` positions.
    pub fn bind_flat_positions(&self, positions: &[f32]) -> Vec<GaussianBinding> {
        assert_eq!(positions.len() % 3, 0);
        let count = positions.len() / 3;
        let points: Vec<[f32; 3]> = (0..count)
            .map(|i| [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]])
            .collect();
        self.bind_gaussians(&points)
    }

    /// Computes barycentric binding coordinates for flattened `[N * 3]` positions with hero bounds.
    pub fn bind_flat_positions_with_bounds(
        &self,
        positions: &[f32],
        hero_min: [f32; 3],
        hero_max: [f32; 3],
    ) -> Vec<GaussianBinding> {
        assert_eq!(positions.len() % 3, 0);
        let count = positions.len() / 3;
        let points: Vec<[f32; 3]> = (0..count)
            .map(|i| [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]])
            .collect();
        self.bind_gaussians_with_bounds(&points, hero_min, hero_max)
    }

    /// Creates a tetrahedral bounding box cage partitioning the axis-aligned box `[min, max]`
    /// into 5 non-overlapping tetrahedra (standard 5-tet cube subdivision).
    pub fn create_box_cage(min: [f32; 3], max: [f32; 3]) -> Self {
        let p0 = [min[0], min[1], min[2]];
        let p1 = [max[0], min[1], min[2]];
        let p2 = [max[0], max[1], min[2]];
        let p3 = [min[0], max[1], min[2]];
        let p4 = [min[0], min[1], max[2]];
        let p5 = [max[0], min[1], max[2]];
        let p6 = [max[0], max[1], max[2]];
        let p7 = [min[0], max[1], max[2]];

        let vertices = vec![p0, p1, p2, p3, p4, p5, p6, p7];

        let elements = vec![
            TetElement::new(0, 1, 3, 4),
            TetElement::new(1, 2, 3, 6),
            TetElement::new(1, 4, 5, 6),
            TetElement::new(3, 4, 6, 7),
            TetElement::new(1, 3, 4, 6),
        ];

        Self::new(vertices, elements)
    }
}

/// A single volumetric cage with its own spring parameters and wireframe colour.
#[derive(Clone, Debug, PartialEq)]
pub struct Cage {
    /// Human-readable cage name (shown in the HUD).
    pub name: String,
    /// Rest topology and precomputed inverse shape matrices.
    pub mesh: TetMesh,
    /// Independent spring physics for this cage.
    pub simulator: CageSpringSimulator,
    /// RGBA colour used by the wireframe overlay.
    pub edge_color: [f32; 4],
}

impl Cage {
    /// Creates a cage with its own simulator initialised at the rest pose.
    pub fn new(name: impl Into<String>, mesh: TetMesh, edge_color: [f32; 4]) -> Self {
        let simulator = CageSpringSimulator::new(mesh.rest_vertices.clone());
        Self {
            name: name.into(),
            mesh,
            simulator,
            edge_color,
        }
    }

    /// Current (possibly deformed) cage vertices.
    pub fn vertices(&self) -> Vec<[f32; 3]> {
        self.simulator.positions_array()
    }

    /// Restores the rest pose and clears velocities.
    pub fn reset(&mut self) {
        self.simulator.reset();
    }

    /// Applies an outward radial velocity pulse to this cage.
    pub fn jiggle(&mut self, magnitude: f32) {
        self.simulator.trigger_jiggle_impulse(magnitude);
    }
}

/// A scene-wide set of independent cages plus per-Gaussian ownership.
#[derive(Clone, Debug, PartialEq)]
pub struct CageScene {
    /// Independent cages, each with its own physics parameters.
    pub cages: Vec<Cage>,
    /// One binding per Gaussian (by scene index), `UNBOUND` when static.
    pub bindings: Vec<CageBinding>,
}

impl Default for CageScene {
    fn default() -> Self {
        Self::new(Vec::new(), Vec::new())
    }
}

impl CageScene {
    /// Creates a cage scene from explicit cages and bindings.
    pub fn new(cages: Vec<Cage>, bindings: Vec<CageBinding>) -> Self {
        Self { cages, bindings }
    }

    /// Binds every cage to the Gaussians inside its rest bounds.
    ///
    /// When cages overlap, the first cage in the list owns the Gaussian; later
    /// cages claim only the splats that are still unbound. This keeps the
    /// assignment deterministic and makes overlapping cages behave predictably.
    pub fn auto_bind(cages: Vec<Cage>, positions: &[f32]) -> Self {
        let count = positions.len() / 3;
        let mut bindings = vec![CageBinding::unbound(); count];

        for (cage_index, cage) in cages.iter().enumerate() {
            let (min, max) = cage.mesh.bounds();
            let bound = cage
                .mesh
                .bind_flat_positions_with_bounds(positions, min, max);

            for (i, binding) in bound.iter().enumerate() {
                if bindings[i].is_bound() || !binding.is_bound() {
                    continue;
                }
                bindings[i] =
                    CageBinding::new(cage_index as u32, binding.tet_index, binding.weights);
            }
        }

        Self { cages, bindings }
    }

    /// Number of cages.
    #[inline]
    pub fn len(&self) -> usize {
        self.cages.len()
    }

    /// Returns true when there are no cages.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cages.is_empty()
    }

    /// Number of Gaussians that at least one cage deforms.
    pub fn bound_count(&self) -> usize {
        self.bindings.iter().filter(|b| b.is_bound()).count()
    }

    /// Advances every cage simulator by `dt` seconds.
    pub fn step(&mut self, dt: f32) {
        for cage in &mut self.cages {
            cage.simulator.step(dt);
        }
    }

    /// Returns the current vertex positions of every cage.
    pub fn deformed_vertices(&self) -> Vec<Vec<[f32; 3]>> {
        self.cages.iter().map(Cage::vertices).collect()
    }

    /// Resets every cage to its rest pose.
    pub fn reset(&mut self) {
        for cage in &mut self.cages {
            cage.reset();
        }
    }

    /// Applies a jiggle impulse to every cage.
    pub fn jiggle(&mut self, magnitude: f32) {
        for cage in &mut self.cages {
            cage.jiggle(magnitude);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniforms_layout() {
        assert_eq!(std::mem::size_of::<DeformationUniforms>(), 32);
        assert_eq!(std::mem::size_of::<TetElement>(), 16);
        assert_eq!(std::mem::size_of::<GaussianBinding>(), 20);
        assert_eq!(std::mem::size_of::<TetPrecomputed>(), 36);
        assert_eq!(std::mem::size_of::<CageBinding>(), 24);
    }

    #[test]
    fn test_box_cage_binding_inside() {
        let cage = TetMesh::create_box_cage([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        assert_eq!(cage.num_vertices(), 8);
        assert_eq!(cage.num_tets(), 5);

        // Center point (0, 0, 0) is inside tet 4
        let pts = vec![[0.0, 0.0, 0.0], [0.5, -0.5, 0.5]];
        let bindings = cage.bind_gaussians(&pts);

        assert_eq!(bindings.len(), 2);
        for b in &bindings {
            let sum_w: f32 = b.weights.iter().sum();
            assert!((sum_w - 1.0).abs() < 1e-5);
            for &w in &b.weights {
                assert!(w >= -1e-4);
            }
        }
    }

    #[test]
    fn test_rest_position_reconstruction() {
        let cage = TetMesh::create_box_cage([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        let test_pt = [0.7, 1.3, 0.4];
        let binding = cage.bind_gaussians(&[test_pt])[0];

        let elem = cage.elements[binding.tet_index as usize];
        let mut reconstructed = Vec3::ZERO;
        for k in 0..4 {
            let v = Vec3::from(cage.rest_vertices[elem.indices[k] as usize]);
            reconstructed += binding.weights[k] * v;
        }

        assert!((reconstructed.x - test_pt[0]).abs() < 1e-4);
        assert!((reconstructed.y - test_pt[1]).abs() < 1e-4);
        assert!((reconstructed.z - test_pt[2]).abs() < 1e-4);
    }

    #[test]
    fn test_multi_cage_auto_binding_is_deterministic_and_disjoint() {
        let cage_a = Cage::new(
            "left",
            TetMesh::create_box_cage([-2.0, -1.0, -1.0], [-0.5, 1.0, 1.0]),
            [1.0, 0.0, 0.0, 1.0],
        );
        let cage_b = Cage::new(
            "right",
            TetMesh::create_box_cage([0.5, -1.0, -1.0], [2.0, 1.0, 1.0]),
            [0.0, 1.0, 0.0, 1.0],
        );

        // Two points inside each cage and one outside both.
        let positions = [
            -1.0f32, 0.0, 0.0, // cage A
            -0.8, 0.2, 0.1, // cage A
            1.0, 0.0, 0.0, // cage B
            0.9, -0.3, 0.2, // cage B
            5.0, 5.0, 5.0, // unbound
        ];

        let scene = CageScene::auto_bind(vec![cage_a, cage_b], &positions);
        assert_eq!(scene.len(), 2);
        assert_eq!(scene.bindings.len(), 5);
        assert_eq!(scene.bindings[0].cage, 0);
        assert_eq!(scene.bindings[1].cage, 0);
        assert_eq!(scene.bindings[2].cage, 1);
        assert_eq!(scene.bindings[3].cage, 1);
        assert!(!scene.bindings[4].is_bound());
        assert_eq!(scene.bound_count(), 4);
    }

    #[test]
    fn test_cages_have_independent_physics_parameters() {
        let cage_a = Cage::new(
            "a",
            TetMesh::create_box_cage([-1.0, -1.0, -1.0], [0.0, 0.0, 0.0]),
            [1.0, 0.0, 0.0, 1.0],
        );
        let mut cage_b = Cage::new(
            "b",
            TetMesh::create_box_cage([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            [0.0, 1.0, 0.0, 1.0],
        );
        cage_b.simulator.stiffness = 42.0;

        let mut scene = CageScene::new(vec![cage_a, cage_b], Vec::new());
        scene.cages[0].simulator.stiffness = 500.0;

        assert_eq!(scene.cages[0].simulator.stiffness, 500.0);
        assert_eq!(scene.cages[1].simulator.stiffness, 42.0);
    }

    #[test]
    fn test_cage_scene_steps_and_resets() {
        let cage = Cage::new(
            "hero",
            TetMesh::create_box_cage([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]),
            [0.0, 1.0, 1.0, 0.5],
        );
        let mut scene = CageScene::new(vec![cage], Vec::new());
        scene.cages[0].simulator.current_positions[0].x += 3.0;

        scene.step(1.0 / 60.0);
        let moved = scene.deformed_vertices()[0][0][0];
        assert!(moved < 2.0, "spring should pull the vertex back");

        scene.reset();
        assert_eq!(
            scene.deformed_vertices()[0][0],
            scene.cages[0].mesh.rest_vertices[0]
        );
    }

    #[test]
    fn test_bounds_cover_vertices() {
        let mesh = TetMesh::create_box_cage([-2.0, -3.0, -4.0], [2.0, 3.0, 4.0]);
        let (min, max) = mesh.bounds();
        assert!(min[0] <= -2.0 && min[1] <= -3.0 && min[2] <= -4.0);
        assert!(max[0] >= 2.0 && max[1] >= 3.0 && max[2] >= 4.0);
    }
}
