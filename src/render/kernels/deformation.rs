use cubecl::prelude::*;

/// CubeCL GPU kernel evaluating the piecewise-constant deformation gradient
/// $J = D_s D_m^{-1}$ per tetrahedron.
///
/// Rotation extraction (polar decomposition) and SH rotation are deliberately not
/// performed here: deformation is a geometry stage on the GPU, and SH coefficients
/// are rotated on the CPU where the exact Wigner-D matrices are cheap to build for
/// a handful of tetrahedra.
///
/// Dispatched with one thread per tetrahedron.
#[allow(clippy::unnecessary_cast)]
#[cube(launch)]
pub fn compute_tet_jacobians_kernel(
    cage_vertices: &[f32],
    tet_elements: &[u32],
    tet_inv_dm: &[f32],
    out_jacobians: &mut [f32],
    num_tets: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= num_tets as usize {
        terminate!();
    }

    // 1. Read vertex indices
    let elem_base = idx * 4usize;
    let v0 = tet_elements[elem_base] as usize;
    let v1 = tet_elements[elem_base + 1usize] as usize;
    let v2 = tet_elements[elem_base + 2usize] as usize;
    let v3 = tet_elements[elem_base + 3usize] as usize;

    // 2. Read deformed cage vertices
    let x0_x = cage_vertices[v0 * 3usize];
    let x0_y = cage_vertices[v0 * 3usize + 1usize];
    let x0_z = cage_vertices[v0 * 3usize + 2usize];

    let x1_x = cage_vertices[v1 * 3usize];
    let x1_y = cage_vertices[v1 * 3usize + 1usize];
    let x1_z = cage_vertices[v1 * 3usize + 2usize];

    let x2_x = cage_vertices[v2 * 3usize];
    let x2_y = cage_vertices[v2 * 3usize + 1usize];
    let x2_z = cage_vertices[v2 * 3usize + 2usize];

    let x3_x = cage_vertices[v3 * 3usize];
    let x3_y = cage_vertices[v3 * 3usize + 1usize];
    let x3_z = cage_vertices[v3 * 3usize + 2usize];

    // 3. Form deformed shape matrix D_s = [x1 - x0, x2 - x0, x3 - x0] (column-major)
    // Column 0
    let ds_00 = x1_x - x0_x;
    let ds_10 = x1_y - x0_y;
    let ds_20 = x1_z - x0_z;
    // Column 1
    let ds_01 = x2_x - x0_x;
    let ds_11 = x2_y - x0_y;
    let ds_21 = x2_z - x0_z;
    // Column 2
    let ds_02 = x3_x - x0_x;
    let ds_12 = x3_y - x0_y;
    let ds_22 = x3_z - x0_z;

    // 4. Read D_m^{-1} (column-major)
    let inv_base = idx * 9usize;
    let a_00 = tet_inv_dm[inv_base];
    let a_10 = tet_inv_dm[inv_base + 1usize];
    let a_20 = tet_inv_dm[inv_base + 2usize];

    let a_01 = tet_inv_dm[inv_base + 3usize];
    let a_11 = tet_inv_dm[inv_base + 4usize];
    let a_21 = tet_inv_dm[inv_base + 5usize];

    let a_02 = tet_inv_dm[inv_base + 6usize];
    let a_12 = tet_inv_dm[inv_base + 7usize];
    let a_22 = tet_inv_dm[inv_base + 8usize];

    // 5. Compute J = D_s * D_m^{-1} (column-major)
    // Column 0
    let j_00 = ds_00 * a_00 + ds_01 * a_10 + ds_02 * a_20;
    let j_10 = ds_10 * a_00 + ds_11 * a_10 + ds_12 * a_20;
    let j_20 = ds_20 * a_00 + ds_21 * a_10 + ds_22 * a_20;
    // Column 1
    let j_01 = ds_00 * a_01 + ds_01 * a_11 + ds_02 * a_21;
    let j_11 = ds_10 * a_01 + ds_11 * a_11 + ds_12 * a_21;
    let j_21 = ds_20 * a_01 + ds_21 * a_11 + ds_22 * a_21;
    // Column 2
    let j_02 = ds_00 * a_02 + ds_01 * a_12 + ds_02 * a_22;
    let j_12 = ds_10 * a_02 + ds_11 * a_12 + ds_12 * a_22;
    let j_22 = ds_20 * a_02 + ds_21 * a_12 + ds_22 * a_22;

    // Store Jacobian
    let out_j_base = idx * 9usize;
    out_jacobians[out_j_base] = j_00;
    out_jacobians[out_j_base + 1usize] = j_10;
    out_jacobians[out_j_base + 2usize] = j_20;
    out_jacobians[out_j_base + 3usize] = j_01;
    out_jacobians[out_j_base + 4usize] = j_11;
    out_jacobians[out_j_base + 5usize] = j_21;
    out_jacobians[out_j_base + 6usize] = j_02;
    out_jacobians[out_j_base + 7usize] = j_12;
    out_jacobians[out_j_base + 8usize] = j_22;
}

/// CubeCL GPU kernel transforming Gaussian geometry under volumetric cage deformation:
/// 1. Deforms positions: $\mathbf{x}'_i = \sum_{k=0}^3 w_k \mathbf{x}_{v_k}$
/// 2. Transforms covariances: $M = J \Sigma_{\text{rest}} J^T$ (symmetrized and regularized)
/// 3. Energy-conserving opacity scaling: $\alpha' = 1 - (1 - \alpha)^{1 / \det(J)}$
///
/// SH coefficients are not evaluated here. The compositor evaluates them in world
/// space; rotating the coefficients themselves is done by the CPU deformation stage.
#[allow(clippy::unnecessary_cast)]
#[cube(launch)]
pub fn deform_gaussians_kernel(
    cage_vertices: &[f32],
    tet_elements: &[u32],
    tet_jacobians: &[f32],
    bindings_tet: &[u32],
    bindings_weights: &[f32],
    rest_positions: &[f32],
    rest_covariances: &[f32],
    rest_opacities: &[f32],
    out_positions: &mut [f32],
    out_covariances: &mut [f32],
    out_opacities: &mut [f32],
    num_gaussians: u32,
    num_tets: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= num_gaussians as usize {
        terminate!();
    }

    // 0. Sentinel check for unbound background splats (outside cage)
    let tet_raw = bindings_tet[idx];
    if tet_raw >= num_tets {
        let pos_base = idx * 3usize;
        out_positions[pos_base] = rest_positions[pos_base];
        out_positions[pos_base + 1usize] = rest_positions[pos_base + 1usize];
        out_positions[pos_base + 2usize] = rest_positions[pos_base + 2usize];

        let cov_base = idx * 6usize;
        out_covariances[cov_base] = rest_covariances[cov_base];
        out_covariances[cov_base + 1usize] = rest_covariances[cov_base + 1usize];
        out_covariances[cov_base + 2usize] = rest_covariances[cov_base + 2usize];
        out_covariances[cov_base + 3usize] = rest_covariances[cov_base + 3usize];
        out_covariances[cov_base + 4usize] = rest_covariances[cov_base + 4usize];
        out_covariances[cov_base + 5usize] = rest_covariances[cov_base + 5usize];

        out_opacities[idx] = rest_opacities[idx];
        terminate!();
    }

    // 1. Barycentric binding lookup
    let tet_idx = tet_raw as usize;
    let w_base = idx * 4usize;
    let w0 = bindings_weights[w_base];
    let w1 = bindings_weights[w_base + 1usize];
    let w2 = bindings_weights[w_base + 2usize];
    let w3 = bindings_weights[w_base + 3usize];

    // 2. Cage vertex positions
    let elem_base = tet_idx * 4usize;
    let v0 = tet_elements[elem_base] as usize;
    let v1 = tet_elements[elem_base + 1usize] as usize;
    let v2 = tet_elements[elem_base + 2usize] as usize;
    let v3 = tet_elements[elem_base + 3usize] as usize;

    let x0_x = cage_vertices[v0 * 3usize];
    let x0_y = cage_vertices[v0 * 3usize + 1usize];
    let x0_z = cage_vertices[v0 * 3usize + 2usize];

    let x1_x = cage_vertices[v1 * 3usize];
    let x1_y = cage_vertices[v1 * 3usize + 1usize];
    let x1_z = cage_vertices[v1 * 3usize + 2usize];

    let x2_x = cage_vertices[v2 * 3usize];
    let x2_y = cage_vertices[v2 * 3usize + 1usize];
    let x2_z = cage_vertices[v2 * 3usize + 2usize];

    let x3_x = cage_vertices[v3 * 3usize];
    let x3_y = cage_vertices[v3 * 3usize + 1usize];
    let x3_z = cage_vertices[v3 * 3usize + 2usize];

    // A. Deformed position
    let px = w0 * x0_x + w1 * x1_x + w2 * x2_x + w3 * x3_x;
    let py = w0 * x0_y + w1 * x1_y + w2 * x2_y + w3 * x3_y;
    let pz = w0 * x0_z + w1 * x1_z + w2 * x2_z + w3 * x3_z;

    let pos_base = idx * 3usize;
    out_positions[pos_base] = px;
    out_positions[pos_base + 1usize] = py;
    out_positions[pos_base + 2usize] = pz;

    // B. Transform covariance M = J * Sigma_rest * J^T
    let j_base = tet_idx * 9usize;
    let j_00 = tet_jacobians[j_base];
    let j_10 = tet_jacobians[j_base + 1usize];
    let j_20 = tet_jacobians[j_base + 2usize];

    let j_01 = tet_jacobians[j_base + 3usize];
    let j_11 = tet_jacobians[j_base + 4usize];
    let j_21 = tet_jacobians[j_base + 5usize];

    let j_02 = tet_jacobians[j_base + 6usize];
    let j_12 = tet_jacobians[j_base + 7usize];
    let j_22 = tet_jacobians[j_base + 8usize];

    let cov_base = idx * 6usize;
    let s_xx = rest_covariances[cov_base];
    let s_xy = rest_covariances[cov_base + 1usize];
    let s_xz = rest_covariances[cov_base + 2usize];
    let s_yy = rest_covariances[cov_base + 3usize];
    let s_yz = rest_covariances[cov_base + 4usize];
    let s_zz = rest_covariances[cov_base + 5usize];

    // Intermediate matrix K = J * Sigma_rest
    let k_00 = j_00 * s_xx + j_01 * s_xy + j_02 * s_xz;
    let k_01 = j_00 * s_xy + j_01 * s_yy + j_02 * s_yz;
    let k_02 = j_00 * s_xz + j_01 * s_yz + j_02 * s_zz;

    let k_10 = j_10 * s_xx + j_11 * s_xy + j_12 * s_xz;
    let k_11 = j_10 * s_xy + j_11 * s_yy + j_12 * s_yz;
    let k_12 = j_10 * s_xz + j_11 * s_yz + j_12 * s_zz;

    let k_20 = j_20 * s_xx + j_21 * s_xy + j_22 * s_xz;
    let k_21 = j_20 * s_xy + j_21 * s_yy + j_22 * s_yz;
    let k_22 = j_20 * s_xz + j_21 * s_yz + j_22 * s_zz;

    // M = K * J^T
    let m_00 = k_00 * j_00 + k_01 * j_01 + k_02 * j_02;
    let m_01 = k_00 * j_10 + k_01 * j_11 + k_02 * j_12;
    let m_02 = k_00 * j_20 + k_01 * j_21 + k_02 * j_22;

    let m_10 = k_10 * j_00 + k_11 * j_01 + k_12 * j_02;
    let m_11 = k_10 * j_10 + k_11 * j_11 + k_12 * j_12;
    let m_12 = k_10 * j_20 + k_11 * j_21 + k_12 * j_22;

    let m_20 = k_20 * j_00 + k_21 * j_01 + k_22 * j_02;
    let m_21 = k_20 * j_10 + k_21 * j_11 + k_22 * j_12;
    let m_22 = k_20 * j_20 + k_21 * j_21 + k_22 * j_22;

    // Symmetrize and regularize diagonal (max(1e-4))
    out_covariances[cov_base] = f32::max(1e-4f32, m_00);
    out_covariances[cov_base + 1usize] = 0.5f32 * (m_01 + m_10);
    out_covariances[cov_base + 2usize] = 0.5f32 * (m_02 + m_20);
    out_covariances[cov_base + 3usize] = f32::max(1e-4f32, m_11);
    out_covariances[cov_base + 4usize] = 0.5f32 * (m_12 + m_21);
    out_covariances[cov_base + 5usize] = f32::max(1e-4f32, m_22);

    // C. Energy-conserving opacity scaling with bounded volume dilation
    let det_j = j_00 * (j_11 * j_22 - j_12 * j_21) - j_01 * (j_10 * j_22 - j_12 * j_20)
        + j_02 * (j_10 * j_21 - j_11 * j_20);

    let s_det = f32::clamp(det_j, 0.6f32, 1.8f32);
    let safe_base = f32::clamp(1.0f32 - rest_opacities[idx], 1e-4f32, 0.999f32);
    let alpha_prime = f32::clamp(
        1.0f32 - f32::powf(safe_base, 1.0f32 / s_det),
        0.05f32,
        0.99f32,
    );
    out_opacities[idx] = alpha_prime;
}
