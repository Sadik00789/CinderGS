pub mod cpu_reference;
pub mod dynamics;
pub mod polar;

pub use cpu_reference::*;
pub use dynamics::*;
pub use polar::*;


use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Vec3};
use rayon::prelude::*;

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
    #[inline]
    pub const fn new(tet_index: u32, weights: [f32; 4]) -> Self {
        Self { tet_index, weights }
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

/// Uniforms buffer for deformation and inverse-ray radiance evaluation.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct DeformationUniforms {
    pub camera_position: [f32; 3],
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

    /// Computes barycentric binding coordinates for an array of 3D points.
    ///
    /// For each point $\mathbf{p}$:
    /// 1. Finds the tetrahedron where $w_k \ge -1e-4$ for all $k \in \{0, 1, 2, 3\}$.
    /// 2. If the point lies outside all tetrahedra, selects the tetrahedron minimizing the barycentric violation penalty.
    pub fn bind_gaussians(&self, positions: &[[f32; 3]]) -> Vec<GaussianBinding> {
        positions
            .par_iter()
            .map(|&p_arr| {
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

                    let penalty = (-w0).max(0.0)
                        + (-w1).max(0.0)
                        + (-w2).max(0.0)
                        + (-w3).max(0.0);

                    if penalty <= 1e-4 {
                        return GaussianBinding::new(t_idx as u32, [w0, w1, w2, w3]);
                    }

                    if penalty < best_penalty {
                        best_penalty = penalty;
                        best_tet = t_idx as u32;
                        best_weights = [w0, w1, w2, w3];
                    }
                }

                GaussianBinding::new(best_tet, best_weights)
            })
            .collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniforms_layout() {
        assert_eq!(std::mem::size_of::<DeformationUniforms>(), 32);
        assert_eq!(std::mem::size_of::<TetElement>(), 16);
        assert_eq!(std::mem::size_of::<GaussianBinding>(), 20);
        assert_eq!(std::mem::size_of::<TetPrecomputed>(), 36);
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
}
