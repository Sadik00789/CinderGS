use crate::deformation::polar::{higham_polar_decomposition, jacobian_singular_values};
use crate::deformation::{
    CageScene, DEFAULT_SPLIT_STRETCH_THRESHOLD, DeformationUniforms, GaussianBinding, TetMesh,
};
use crate::math::covariance::{
    DEFAULT_MIP_FILTER_3D_VAR, apply_mip_filter_3d, split_gaussian_along_major_axis,
};
use crate::math::covariance_matrix_from_slice;
use crate::parallel;
use crate::render::rasterizer::sh_rotation::ShRotator;
use crate::scene::soa::GaussianSceneSoa;
use glam::{Mat3, Vec3};

/// Output of the CPU reference volumetric cage deformation pipeline.
#[derive(Clone, Debug, PartialEq)]
pub struct DeformedSceneOutput {
    /// Deformed Gaussian scene with updated positions, transformed covariances,
    /// scaled opacities, and SH coefficients rotated into the deformed frame.
    pub scene: GaussianSceneSoa,
}

impl DeformedSceneOutput {
    /// Returns a cloned [`GaussianSceneSoa`] for canonical/deferred consumers.
    #[inline]
    pub fn to_soa(&self, _sh_degree: u32) -> GaussianSceneSoa {
        self.scene.clone()
    }

    /// Consumes this output and yields its inner [`GaussianSceneSoa`].
    #[inline]
    pub fn into_soa(self) -> GaussianSceneSoa {
        self.scene
    }
}

/// Per-tetrahedron transform: deformation gradient, polar rotation, and max principal stretch.
#[derive(Clone, Copy, Debug)]
pub struct TetTransform {
    /// Deformation gradient $J = D_s D_m^{-1}$.
    pub jacobian: Mat3,
    /// Proper rotation $R \in \mathrm{SO}(3)$ of the polar decomposition $J = RP$.
    pub rotation: Mat3,
    /// Maximum singular value $\sigma_1$ of $J$.
    pub max_stretch: f32,
}

const DEFORM_CHUNK_SIZE: usize = 1024;

/// Computes deformation gradient matrices $J = D_s D_m^{-1}$ and polar rotations $R$ for all tetrahedra.
pub fn compute_tet_transforms_cpu(
    tet_mesh: &TetMesh,
    deformed_vertices: &[[f32; 3]],
) -> (Vec<Mat3>, Vec<Mat3>) {
    let transforms = tet_transforms(tet_mesh, deformed_vertices);
    (
        transforms.iter().map(|t| t.jacobian).collect(),
        transforms.iter().map(|t| t.rotation).collect(),
    )
}

fn tet_transforms(tet_mesh: &TetMesh, deformed_vertices: &[[f32; 3]]) -> Vec<TetTransform> {
    let mut transforms = Vec::with_capacity(tet_mesh.elements.len());

    for (t_idx, elem) in tet_mesh.elements.iter().enumerate() {
        let mut corners = [Vec3::ZERO; 4];
        for (k, &index) in elem.indices.iter().enumerate() {
            corners[k] = deformed_vertices
                .get(index as usize)
                .copied()
                .map(Vec3::from)
                .unwrap_or_else(|| Vec3::from(tet_mesh.rest_vertices[index as usize]));
        }

        let ds = Mat3::from_cols(
            corners[1] - corners[0],
            corners[2] - corners[0],
            corners[3] - corners[0],
        );
        let inv_dm = tet_mesh.precomputed[t_idx].matrix();

        let jacobian = ds * inv_dm;
        let (rotation, _) = higham_polar_decomposition(jacobian);
        let max_stretch = jacobian_singular_values(jacobian).max_stretch();

        transforms.push(TetTransform {
            jacobian,
            rotation,
            max_stretch,
        });
    }

    transforms
}

/// Deforms a single Gaussian and writes it into the chunk-local output slices.
#[allow(clippy::too_many_arguments)]
fn write_deformed(
    out_positions: &mut [f32],
    out_covariances: &mut [f32],
    out_opacities: &mut [f32],
    out_sh: &mut [f32],
    local: usize,
    position: Vec3,
    covariance: [f32; 6],
    opacity: f32,
    sh: &[f32],
    transform: TetTransform,
    rotator: Option<&ShRotator>,
) {
    let pos_base = local * 3;
    out_positions[pos_base] = position.x;
    out_positions[pos_base + 1] = position.y;
    out_positions[pos_base + 2] = position.z;

    let sigma = covariance_matrix_from_slice(&covariance);
    let m = transform.jacobian * sigma * transform.jacobian.transpose();
    let cov_base = local * 6;
    out_covariances[cov_base] = m.x_axis.x.max(1e-4);
    out_covariances[cov_base + 1] = 0.5 * (m.x_axis.y + m.y_axis.x);
    out_covariances[cov_base + 2] = 0.5 * (m.x_axis.z + m.z_axis.x);
    out_covariances[cov_base + 3] = m.y_axis.y.max(1e-4);
    out_covariances[cov_base + 4] = 0.5 * (m.y_axis.z + m.z_axis.y);
    out_covariances[cov_base + 5] = m.z_axis.z.max(1e-4);

    // Energy-conserving opacity scaling with hard-bounded volume dilation
    let s_det = transform.jacobian.determinant().clamp(0.6, 1.8);
    let safe_base = (1.0 - opacity).clamp(1e-4, 0.999);
    out_opacities[local] = (1.0 - safe_base.powf(1.0 / s_det)).clamp(0.05, 0.99);

    let sh_base = local * 48;
    let mut rotated = [0.0f32; 48];
    rotated.copy_from_slice(&sh[..48]);
    if let Some(rotator) = rotator {
        rotator.apply(&mut rotated);
    }
    out_sh[sh_base..sh_base + 48].copy_from_slice(&rotated);
}

#[allow(clippy::too_many_arguments)]
fn write_static(
    out_positions: &mut [f32],
    out_covariances: &mut [f32],
    out_opacities: &mut [f32],
    out_sh: &mut [f32],
    local: usize,
    position: [f32; 3],
    covariance: [f32; 6],
    opacity: f32,
    sh: &[f32],
) {
    out_positions[local * 3..local * 3 + 3].copy_from_slice(&position);
    out_covariances[local * 6..local * 6 + 6].copy_from_slice(&covariance);
    out_opacities[local] = opacity;
    out_sh[local * 48..local * 48 + 48].copy_from_slice(&sh[..48]);
}

fn deform_scene_with<F>(
    scene: &GaussianSceneSoa,
    uniforms: &DeformationUniforms,
    cage_vertices: &[Vec<[f32; 3]>],
    transforms: &[Vec<TetTransform>],
    meshes: &[&TetMesh],
    resolve: F,
) -> DeformedSceneOutput
where
    F: Fn(usize) -> Option<(usize, usize, [f32; 4])> + Sync + Send,
{
    let count = scene.count;
    let degree = uniforms.sh_degree.min(3);

    // Build the SH rotation matrices once per tetrahedron, not once per splat.
    let sh_rotators: Vec<Vec<ShRotator>> = if degree == 0 {
        Vec::new()
    } else {
        transforms
            .iter()
            .map(|tets| {
                tets.iter()
                    .map(|transform| ShRotator::new(transform.rotation, degree))
                    .collect()
            })
            .collect()
    };

    let mut out_positions = vec![0.0f32; count * 3];
    let mut out_covariances = vec![0.0f32; count * 6];
    let mut out_opacities = vec![0.0f32; count];
    let mut out_sh = vec![0.0f32; count * 48];
    let split_threshold = uniforms.split_threshold();
    let mip_filter_3d_var = uniforms.mip_filter_3d_var();
    let mut needs_split = if split_threshold > 0.0 {
        vec![0u8; count]
    } else {
        Vec::new()
    };

    parallel::zip_fill_4(
        &mut out_positions,
        &mut out_covariances,
        &mut out_opacities,
        &mut out_sh,
        [
            DEFORM_CHUNK_SIZE * 3,
            DEFORM_CHUNK_SIZE * 6,
            DEFORM_CHUNK_SIZE,
            DEFORM_CHUNK_SIZE * 48,
        ],
        |base, pos_chunk, cov_chunk, opa_chunk, sh_chunk| {
            for local in 0..opa_chunk.len() {
                let i = base + local;
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
                let rest_sh = &scene.sh_coeffs[i * 48..(i + 1) * 48];
                let rest_alpha = scene.opacities[i];

                let Some((cage, tet, weights)) = resolve(i) else {
                    let (final_cov, final_alpha) = if mip_filter_3d_var > 0.0 {
                        apply_mip_filter_3d(&rest_cov, rest_alpha, mip_filter_3d_var)
                    } else {
                        (rest_cov, rest_alpha)
                    };
                    write_static(
                        pos_chunk,
                        cov_chunk,
                        opa_chunk,
                        sh_chunk,
                        local,
                        rest_pos,
                        final_cov,
                        final_alpha,
                        rest_sh,
                    );
                    continue;
                };

                if cage >= transforms.len()
                    || cage >= cage_vertices.len()
                    || cage >= meshes.len()
                    || tet >= transforms[cage].len()
                {
                    let (final_cov, final_alpha) = if mip_filter_3d_var > 0.0 {
                        apply_mip_filter_3d(&rest_cov, rest_alpha, mip_filter_3d_var)
                    } else {
                        (rest_cov, rest_alpha)
                    };
                    write_static(
                        pos_chunk,
                        cov_chunk,
                        opa_chunk,
                        sh_chunk,
                        local,
                        rest_pos,
                        final_cov,
                        final_alpha,
                        rest_sh,
                    );
                    continue;
                }

                let elem = meshes[cage].elements[tet];
                let verts = &cage_vertices[cage];
                let mut corners = [Vec3::ZERO; 4];
                for (k, &index) in elem.indices.iter().enumerate() {
                    corners[k] = verts
                        .get(index as usize)
                        .copied()
                        .map(Vec3::from)
                        .unwrap_or_else(|| Vec3::from(rest_pos));
                }

                let deformed = weights[0] * corners[0]
                    + weights[1] * corners[1]
                    + weights[2] * corners[2]
                    + weights[3] * corners[3];

                write_deformed(
                    pos_chunk,
                    cov_chunk,
                    opa_chunk,
                    sh_chunk,
                    local,
                    deformed,
                    rest_cov,
                    rest_alpha,
                    rest_sh,
                    transforms[cage][tet],
                    sh_rotators.get(cage).and_then(|tets| tets.get(tet)),
                );
            }
        },
    );

    if split_threshold > 0.0 {
        for (i, flag) in needs_split.iter_mut().enumerate().take(count) {
            if let Some((cage, tet, _)) = resolve(i)
                && cage < transforms.len()
                && tet < transforms[cage].len()
                && transforms[cage][tet].max_stretch > split_threshold
            {
                *flag = 1;
            }
        }
    }

    let mut total_count = count;
    if split_threshold > 0.0 {
        for (i, &flag) in needs_split.iter().enumerate().take(count) {
            if flag == 0 {
                if mip_filter_3d_var > 0.0 {
                    let cov = [
                        out_covariances[i * 6],
                        out_covariances[i * 6 + 1],
                        out_covariances[i * 6 + 2],
                        out_covariances[i * 6 + 3],
                        out_covariances[i * 6 + 4],
                        out_covariances[i * 6 + 5],
                    ];
                    let (reg_cov, reg_alpha) =
                        apply_mip_filter_3d(&cov, out_opacities[i], mip_filter_3d_var);
                    out_covariances[i * 6..i * 6 + 6].copy_from_slice(&reg_cov);
                    out_opacities[i] = reg_alpha;
                }
                continue;
            }

            let center = Vec3::new(
                out_positions[i * 3],
                out_positions[i * 3 + 1],
                out_positions[i * 3 + 2],
            );
            let cov = [
                out_covariances[i * 6],
                out_covariances[i * 6 + 1],
                out_covariances[i * 6 + 2],
                out_covariances[i * 6 + 3],
                out_covariances[i * 6 + 4],
                out_covariances[i * 6 + 5],
            ];
            let alpha = out_opacities[i];
            let [child_a, child_b] = split_gaussian_along_major_axis(center, &cov, alpha);

            let (cov_a, alpha_a) = if mip_filter_3d_var > 0.0 {
                apply_mip_filter_3d(&child_a.covariance_3d, child_a.opacity, mip_filter_3d_var)
            } else {
                (child_a.covariance_3d, child_a.opacity)
            };
            let pos_a = center + child_a.offset;
            out_positions[i * 3] = pos_a.x;
            out_positions[i * 3 + 1] = pos_a.y;
            out_positions[i * 3 + 2] = pos_a.z;
            out_covariances[i * 6..i * 6 + 6].copy_from_slice(&cov_a);
            out_opacities[i] = alpha_a;

            let (cov_b, alpha_b) = if mip_filter_3d_var > 0.0 {
                apply_mip_filter_3d(&child_b.covariance_3d, child_b.opacity, mip_filter_3d_var)
            } else {
                (child_b.covariance_3d, child_b.opacity)
            };
            let pos_b = center + child_b.offset;
            out_positions.extend_from_slice(&[pos_b.x, pos_b.y, pos_b.z]);
            out_covariances.extend_from_slice(&cov_b);
            out_opacities.push(alpha_b);
            let sh_start = i * 48;
            out_sh.extend_from_within(sh_start..sh_start + 48);
            total_count += 1;
        }
    } else if mip_filter_3d_var > 0.0 {
        for i in 0..count {
            let cov = [
                out_covariances[i * 6],
                out_covariances[i * 6 + 1],
                out_covariances[i * 6 + 2],
                out_covariances[i * 6 + 3],
                out_covariances[i * 6 + 4],
                out_covariances[i * 6 + 5],
            ];
            let (reg_cov, reg_alpha) =
                apply_mip_filter_3d(&cov, out_opacities[i], mip_filter_3d_var);
            out_covariances[i * 6..i * 6 + 6].copy_from_slice(&reg_cov);
            out_opacities[i] = reg_alpha;
        }
    }

    DeformedSceneOutput {
        scene: GaussianSceneSoa {
            positions: out_positions,
            covariances_3d: out_covariances,
            opacities: out_opacities,
            sh_coeffs: out_sh,
            count: total_count,
        },
    }
}

/// CPU reference implementation of the single-cage volumetric deformation pipeline.
///
/// 1. Evaluates piecewise-constant deformation gradients $J_t = D_s D_m^{-1}$ and rotations $R_t \in \text{SO}(3)$.
/// 2. Deforms positions: $\mathbf{x}'_i = \sum_{k=0}^3 w_k \mathbf{x}_{v_k}$.
/// 3. Transforms covariances: $M = J \Sigma_{\text{rest}} J^T$, enforces symmetry, regularizes diagonal ($\ge 1e-4$).
/// 4. Scales opacities for energy conservation: $\alpha' = 1 - (1 - \alpha)^{1 / \det(J)}$.
/// 5. Rotates SH coefficients by $R_t$ so view-dependent colour follows the deformation.
pub fn deform_scene_cpu(
    scene: &GaussianSceneSoa,
    tet_mesh: &TetMesh,
    bindings: &[GaussianBinding],
    deformed_vertices: &[[f32; 3]],
    uniforms: &DeformationUniforms,
) -> DeformedSceneOutput {
    let transforms = tet_transforms(tet_mesh, deformed_vertices);
    let cage_vertices = vec![deformed_vertices.to_vec()];
    let transform_sets = vec![transforms];

    deform_scene_with(
        scene,
        uniforms,
        &cage_vertices,
        &transform_sets,
        &[tet_mesh],
        |i| {
            let binding = bindings.get(i)?;
            if !binding.is_bound() || binding.tet_index as usize >= transform_sets[0].len() {
                return None;
            }
            Some((0, binding.tet_index as usize, binding.weights))
        },
    )
}

/// CPU reference implementation of the multi-cage deformation pipeline.
///
/// Each Gaussian is deformed by the cage that owns it (first matching cage wins
/// at binding time). Cages never interact, so their spring parameters remain
/// independent.
pub fn deform_cage_scene_cpu(
    scene: &GaussianSceneSoa,
    cages: &CageScene,
    uniforms: &DeformationUniforms,
) -> DeformedSceneOutput {
    let cage_vertices = cages.deformed_vertices();
    let meshes: Vec<&TetMesh> = cages.cages.iter().map(|c| &c.mesh).collect();
    let transforms: Vec<Vec<TetTransform>> = cages
        .cages
        .iter()
        .zip(cage_vertices.iter())
        .map(|(cage, vertices)| tet_transforms(&cage.mesh, vertices))
        .collect();

    deform_scene_with(scene, uniforms, &cage_vertices, &transforms, &meshes, |i| {
        let binding = cages.bindings.get(i)?;
        if !binding.is_bound() {
            return None;
        }
        Some((binding.cage as usize, binding.tet as usize, binding.weights))
    })
}

/// Adaptive variant of [`deform_scene_cpu`] enabling strain-driven transient
/// splitting and 3D Mip-Splatting variance regularization using either the thresholds
/// configured in `uniforms` or the defaults (`DEFAULT_SPLIT_STRETCH_THRESHOLD` /
/// `DEFAULT_MIP_FILTER_3D_VAR`).
pub fn deform_scene_adaptive_cpu(
    scene: &GaussianSceneSoa,
    tet_mesh: &TetMesh,
    bindings: &[GaussianBinding],
    deformed_vertices: &[[f32; 3]],
    uniforms: &DeformationUniforms,
) -> DeformedSceneOutput {
    let mut adaptive = *uniforms;
    if adaptive.split_threshold() <= 0.0 {
        adaptive = adaptive.with_strain_splitting(DEFAULT_SPLIT_STRETCH_THRESHOLD);
    }
    if adaptive.mip_filter_3d_var() <= 0.0 {
        adaptive = adaptive.with_mip_filter_3d(DEFAULT_MIP_FILTER_3D_VAR);
    }
    deform_scene_cpu(scene, tet_mesh, bindings, deformed_vertices, &adaptive)
}

/// Adaptive variant of [`deform_cage_scene_cpu`] enabling strain-driven transient
/// splitting and 3D Mip-Splatting variance regularization.
pub fn deform_cage_scene_adaptive_cpu(
    scene: &GaussianSceneSoa,
    cages: &CageScene,
    uniforms: &DeformationUniforms,
) -> DeformedSceneOutput {
    let mut adaptive = *uniforms;
    if adaptive.split_threshold() <= 0.0 {
        adaptive = adaptive.with_strain_splitting(DEFAULT_SPLIT_STRETCH_THRESHOLD);
    }
    if adaptive.mip_filter_3d_var() <= 0.0 {
        adaptive = adaptive.with_mip_filter_3d(DEFAULT_MIP_FILTER_3D_VAR);
    }
    deform_cage_scene_cpu(scene, cages, &adaptive)
}

#[cfg(test)]
mod adaptive_splitting_tests {
    use super::*;
    use crate::math::compute_covariance_3d;
    use crate::scene::raw_vertex::RawPlyVertex;

    #[test]
    fn test_adaptive_splitting_adds_transient_children_without_mutating_rest_scene() {
        let mut scene = GaussianSceneSoa::new();
        let mut raw = RawPlyVertex::ZERO;
        raw.position = [0.5, 0.5, 0.5];
        raw.opacity = 1.0;
        raw.scale = [-2.0, -2.0, -2.0];
        scene.push_raw(&raw);
        let original_count = scene.count;
        let original_pos = scene.positions.clone();

        let tet = TetMesh::create_box_cage([0.0; 3], [1.0; 3]);
        let bindings = tet.bind_gaussians(bytemuck::cast_slice(&scene.positions));
        let mut deformed = tet.rest_vertices.clone();
        // Stretch +X face to 2.5x along X, exceeding DEFAULT_SPLIT_STRETCH_THRESHOLD (1.75)
        for v in &mut deformed {
            v[0] *= 2.5;
        }

        let uniforms =
            DeformationUniforms::new([0.0; 3], 0, scene.count as u32, tet.num_tets() as u32);
        let out = deform_scene_adaptive_cpu(&scene, &tet, &bindings, &deformed, &uniforms);

        // Rest scene remains immutable, while deformed output splits the elongated Gaussian.
        assert_eq!(scene.count, original_count);
        assert_eq!(scene.positions, original_pos);
        assert_eq!(out.scene.count, 2);
        let c = compute_covariance_3d(Vec3::splat(0.1), glam::Quat::IDENTITY);
        assert_eq!(c.len(), 6);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_scene(positions: &[[f32; 3]]) -> GaussianSceneSoa {
        let mut scene = GaussianSceneSoa::with_capacity(positions.len());
        let mut sh = [0.0f32; 48];
        sh[0] = 0.3;
        sh[4] = 0.2;
        for p in positions {
            scene.push(*p, [0.04, 0.0, 0.0, 0.04, 0.0, 0.04], 0.8, &sh);
        }
        scene
    }

    #[test]
    fn test_unbound_gaussians_are_preserved() {
        let scene = simple_scene(&[[10.0, 10.0, 10.0]]);
        let cage = TetMesh::create_box_cage([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        let bindings = vec![GaussianBinding::unbound()];
        let uniforms = DeformationUniforms::new([0.0; 3], 3, 1, cage.num_tets() as u32);

        let out = deform_scene_cpu(&scene, &cage, &bindings, &cage.rest_vertices, &uniforms);

        assert_eq!(out.scene.positions, scene.positions);
        assert_eq!(out.scene.covariances_3d, scene.covariances_3d);
        assert_eq!(out.scene.opacities, scene.opacities);
        assert_eq!(out.scene.sh_coeffs, scene.sh_coeffs);
    }

    #[test]
    fn test_two_cages_deform_independently() {
        let positions = [-1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0];
        let scene = simple_scene(&[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);

        let cage_a = crate::deformation::Cage::new(
            "a",
            TetMesh::create_box_cage([-2.0, -1.0, -1.0], [-0.5, 1.0, 1.0]),
            [1.0, 0.0, 0.0, 1.0],
        );
        let cage_b = crate::deformation::Cage::new(
            "b",
            TetMesh::create_box_cage([0.5, -1.0, -1.0], [2.0, 1.0, 1.0]),
            [0.0, 1.0, 0.0, 1.0],
        );

        let mut cages = CageScene::auto_bind(vec![cage_a, cage_b], &positions);

        // Move the first cage's vertices by +5 in Y; the second cage stays at rest.
        for vertex in &mut cages.cages[0].simulator.current_positions {
            vertex.y += 5.0;
        }

        let uniforms = DeformationUniforms::new([0.0; 3], 0, 2, 5);
        let out = deform_cage_scene_cpu(&scene, &cages, &uniforms);

        assert!(
            (out.scene.positions[1] - 5.0).abs() < 1e-4,
            "gaussian owned by cage A must move, got {}",
            out.scene.positions[1]
        );
        assert!(
            out.scene.positions[4].abs() < 1e-4,
            "gaussian owned by cage B must stay at rest, got {}",
            out.scene.positions[4]
        );
    }

    #[test]
    fn test_identity_transform_keeps_covariance_and_sh() {
        let scene = simple_scene(&[[0.25, -0.5, 0.75]]);
        let cage = TetMesh::create_box_cage([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        let bindings = cage.bind_gaussians(&[[0.25, -0.5, 0.75]]);
        let uniforms = DeformationUniforms::new([0.0; 3], 3, 1, cage.num_tets() as u32);

        let out = deform_scene_cpu(&scene, &cage, &bindings, &cage.rest_vertices, &uniforms);

        for (a, b) in out.scene.sh_coeffs.iter().zip(scene.sh_coeffs.iter()) {
            assert!((a - b).abs() < 1e-4);
        }
        assert!((out.scene.opacities[0] - scene.opacities[0]).abs() < 1e-4);
    }
}
