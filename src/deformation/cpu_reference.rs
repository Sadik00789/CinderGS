use crate::deformation::polar::higham_polar_decomposition;
use crate::deformation::{DeformationUniforms, GaussianBinding, TetMesh};
use crate::math::covariance_matrix_from_slice;
use crate::render::rasterizer::sh::evaluate_sh;
use crate::scene::soa::GaussianSceneSoa;
use glam::{Mat3, Vec3};
use rayon::prelude::*;

/// Output of CPU reference volumetric cage deformation pipeline.
#[derive(Clone, Debug, PartialEq)]
pub struct DeformedSceneOutput {
    /// Deformed Gaussian scene with updated positions, transformed covariances, and scaled opacities.
    pub scene: GaussianSceneSoa,
    /// Evaluated directional RGB colors [N * 3] along deformed inverse rays.
    pub colors: Vec<f32>,
}

/// Computes deformation gradient matrices $J = D_s D_m^{-1}$ and polar rotations $R$ for all tetrahedra.
pub fn compute_tet_transforms_cpu(
    tet_mesh: &TetMesh,
    deformed_vertices: &[[f32; 3]],
) -> (Vec<Mat3>, Vec<Mat3>) {
    let num_tets = tet_mesh.elements.len();
    let mut jacobians = Vec::with_capacity(num_tets);
    let mut rotations = Vec::with_capacity(num_tets);

    for (t_idx, elem) in tet_mesh.elements.iter().enumerate() {
        let x0 = Vec3::from(deformed_vertices[elem.indices[0] as usize]);
        let x1 = Vec3::from(deformed_vertices[elem.indices[1] as usize]);
        let x2 = Vec3::from(deformed_vertices[elem.indices[2] as usize]);
        let x3 = Vec3::from(deformed_vertices[elem.indices[3] as usize]);

        let ds = Mat3::from_cols(x1 - x0, x2 - x0, x3 - x0);
        let inv_dm = tet_mesh.precomputed[t_idx].matrix();

        let j = ds * inv_dm;
        let (r, _) = higham_polar_decomposition(j);

        jacobians.push(j);
        rotations.push(r);
    }

    (jacobians, rotations)
}

/// CPU reference implementation of the volumetric cage deformation pipeline.
///
/// Multi-threaded using Rayon:
/// 1. Evaluates piecewise-constant deformation gradients $J_t = D_s D_m^{-1}$ and rotations $R_t \in \text{SO}(3)$.
/// 2. Deforms positions: $\mathbf{x}'_i = \sum_{k=0}^3 w_k \mathbf{x}_{v_k}$.
/// 3. Transforms covariances: $M = J \Sigma_{\text{rest}} J^T$, enforces symmetry, regularizes diagonal ($\ge 1e-4$).
/// 4. Scales opacities for energy conservation: $\alpha' = 1 - (1 - \alpha)^{1 / \det(J)}$.
/// 5. Evaluates inverse-ray SH colors: $\mathbf{d}_{\text{local}} = R^T \mathbf{d}_{\text{world}}$.
pub fn deform_scene_cpu(
    scene: &GaussianSceneSoa,
    tet_mesh: &TetMesh,
    bindings: &[GaussianBinding],
    deformed_vertices: &[[f32; 3]],
    uniforms: &DeformationUniforms,
) -> DeformedSceneOutput {
    let count = scene.count;
    assert_eq!(bindings.len(), count);

    // 1. Precompute per-tet Jacobians and rotations
    let (jacobians, rotations) = compute_tet_transforms_cpu(tet_mesh, deformed_vertices);

    let cam_pos = Vec3::from(uniforms.camera_position);
    let sh_degree = uniforms.sh_degree;

    // Parallel transformation per Gaussian
    struct DeformedGaussianItem {
        pos: [f32; 3],
        cov: [f32; 6],
        alpha: f32,
        color: [f32; 3],
    }

    let deformed_items: Vec<DeformedGaussianItem> = (0..count)
        .into_par_iter()
        .map(|i| {
            let binding = &bindings[i];
            let num_tets = tet_mesh.elements.len();

            // Sentinel handling for unbound background splats: static and preserved opacity
            if binding.tet_index == GaussianBinding::UNBOUND || binding.tet_index as usize >= num_tets {
                let rest_pos = [
                    scene.positions[i * 3],
                    scene.positions[i * 3 + 1],
                    scene.positions[i * 3 + 2],
                ];
                let rest_cov = [
                    scene.covariances_3d[i * 6],
                    scene.covariances_3d[i * 6 + 1],
                    scene.covariances_3d[i * 6 + 2],
                    scene.covariances_3d[i * 6 + 3],
                    scene.covariances_3d[i * 6 + 4],
                    scene.covariances_3d[i * 6 + 5],
                ];
                let rest_alpha = scene.opacities[i];
                let to_cam = Vec3::from(rest_pos) - cam_pos;
                let dist = to_cam.length();
                let d_world = if dist > 1e-6 {
                    to_cam / dist
                } else {
                    Vec3::Z
                };
                let sh_slice = &scene.sh_coeffs[i * 48..(i + 1) * 48];
                let rgb = evaluate_sh(sh_slice, d_world, sh_degree);

                return DeformedGaussianItem {
                    pos: rest_pos,
                    cov: rest_cov,
                    alpha: rest_alpha,
                    color: rgb,
                };
            }

            let t_idx = binding.tet_index as usize;
            let elem = &tet_mesh.elements[t_idx];
            let w = binding.weights;

            // A. Deform position
            let x0 = Vec3::from(deformed_vertices[elem.indices[0] as usize]);
            let x1 = Vec3::from(deformed_vertices[elem.indices[1] as usize]);
            let x2 = Vec3::from(deformed_vertices[elem.indices[2] as usize]);
            let x3 = Vec3::from(deformed_vertices[elem.indices[3] as usize]);

            let def_pos = w[0] * x0 + w[1] * x1 + w[2] * x2 + w[3] * x3;

            // B. Transform covariance M = J * Sigma_rest * J^T
            let j = jacobians[t_idx];
            let rest_cov_slice: [f32; 6] = [
                scene.covariances_3d[i * 6],
                scene.covariances_3d[i * 6 + 1],
                scene.covariances_3d[i * 6 + 2],
                scene.covariances_3d[i * 6 + 3],
                scene.covariances_3d[i * 6 + 4],
                scene.covariances_3d[i * 6 + 5],
            ];
            let rest_sigma = covariance_matrix_from_slice(&rest_cov_slice);
            let m = j * rest_sigma * j.transpose();

            // Extract upper-triangular components and enforce strict symmetry
            let def_cov = [
                m.x_axis.x.max(1e-4),                         // xx
                0.5 * (m.x_axis.y + m.y_axis.x),              // xy
                0.5 * (m.x_axis.z + m.z_axis.x),              // xz
                m.y_axis.y.max(1e-4),                         // yy
                0.5 * (m.y_axis.z + m.z_axis.y),              // yz
                m.z_axis.z.max(1e-4),                         // zz
            ];

            // C. Energy-conserving opacity scaling with hard-bounded volume dilation
            let rest_alpha = scene.opacities[i];
            let det_j = j.determinant();
            let s_det = det_j.clamp(0.6, 1.8);
            let safe_base = (1.0 - rest_alpha).clamp(1e-4, 0.999);
            let def_alpha = (1.0 - safe_base.powf(1.0 / s_det)).clamp(0.05, 0.99);

            // D. Inverse-ray directional SH evaluation
            let to_cam = def_pos - cam_pos;
            let dist = to_cam.length();
            let d_world = if dist > 1e-6 {
                to_cam / dist
            } else {
                Vec3::Z
            };

            let r = rotations[t_idx];
            let d_local = r.transpose() * d_world;

            let sh_slice = &scene.sh_coeffs[i * 48..(i + 1) * 48];
            let rgb = evaluate_sh(sh_slice, d_local, sh_degree);

            DeformedGaussianItem {
                pos: [def_pos.x, def_pos.y, def_pos.z],
                cov: def_cov,
                alpha: def_alpha,
                color: rgb,
            }
        })
        .collect();

    let mut out_positions = Vec::with_capacity(count * 3);
    let mut out_covariances = Vec::with_capacity(count * 6);
    let mut out_opacities = Vec::with_capacity(count);
    let mut out_colors = Vec::with_capacity(count * 3);

    for item in deformed_items {
        out_positions.extend_from_slice(&item.pos);
        out_covariances.extend_from_slice(&item.cov);
        out_opacities.push(item.alpha);
        out_colors.extend_from_slice(&item.color);
    }

    let out_scene = GaussianSceneSoa {
        positions: out_positions,
        covariances_3d: out_covariances,
        opacities: out_opacities,
        sh_coeffs: scene.sh_coeffs.clone(),
        count,
    };

    DeformedSceneOutput {
        scene: out_scene,
        colors: out_colors,
    }
}
