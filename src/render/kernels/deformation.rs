use cubecl::prelude::*;

/// CubeCL GPU kernel evaluating piecewise-constant deformation gradient $J = D_s D_m^{-1}$
/// and polar decomposition rotation $R \in \text{SO}(3)$ per tetrahedron.
///
/// Dispatched with 1 thread per tetrahedron.
#[allow(clippy::unnecessary_cast)]
#[cube(launch)]
pub fn compute_tet_jacobians_kernel(
    cage_vertices: &[f32],
    tet_elements: &[u32],
    tet_inv_dm: &[f32],
    out_jacobians: &mut [f32],
    out_rotations: &mut [f32],
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

    // 6. Higham Polar Decomposition using 5-step Newton-Schulz iteration
    let det_j = j_00 * (j_11 * j_22 - j_12 * j_21)
        - j_01 * (j_10 * j_22 - j_12 * j_20)
        + j_02 * (j_10 * j_21 - j_11 * j_20);

    let abs_det = f32::abs(det_j);
    let mut r_00 = 1.0f32;
    let mut r_10 = 0.0f32;
    let mut r_20 = 0.0f32;
    let mut r_01 = 0.0f32;
    let mut r_11 = 1.0f32;
    let mut r_21 = 0.0f32;
    let mut r_02 = 0.0f32;
    let mut r_12 = 0.0f32;
    let mut r_22 = 1.0f32;

    if abs_det >= 1e-6f32 {
        let gamma = f32::powf(abs_det, -0.33333334f32);
        r_00 = j_00 * gamma;
        r_10 = j_10 * gamma;
        r_20 = j_20 * gamma;
        r_01 = j_01 * gamma;
        r_11 = j_11 * gamma;
        r_21 = j_21 * gamma;
        r_02 = j_02 * gamma;
        r_12 = j_12 * gamma;
        r_22 = j_22 * gamma;

        // 5 Newton-Schulz iterations: R_{k+1} = 0.5 * (R_k + R_k^{-T})
        // For 3x3, R^{-T} = C / det(R) where C is the cofactor matrix
        for _k in 0u32..5u32 {
            let c_00 = r_11 * r_22 - r_12 * r_21;
            let c_01 = -(r_10 * r_22 - r_12 * r_20);
            let c_02 = r_10 * r_21 - r_11 * r_20;

            let c_10 = -(r_01 * r_22 - r_02 * r_21);
            let c_11 = r_00 * r_22 - r_02 * r_20;
            let c_12 = -(r_00 * r_21 - r_01 * r_20);

            let c_20 = r_01 * r_12 - r_02 * r_11;
            let c_21 = -(r_00 * r_12 - r_02 * r_10);
            let c_22 = r_00 * r_11 - r_01 * r_10;

            let cur_det = r_00 * c_00 + r_01 * c_01 + r_02 * c_02;
            let inv_cur_det = 1.0f32 / f32::max(1e-6f32, f32::abs(cur_det));
            let sign = if cur_det < 0.0f32 { -1.0f32 } else { 1.0f32 };
            let scale = inv_cur_det * sign;

            r_00 = 0.5f32 * (r_00 + c_00 * scale);
            r_10 = 0.5f32 * (r_10 + c_10 * scale);
            r_20 = 0.5f32 * (r_20 + c_20 * scale);

            r_01 = 0.5f32 * (r_01 + c_01 * scale);
            r_11 = 0.5f32 * (r_11 + c_11 * scale);
            r_21 = 0.5f32 * (r_21 + c_21 * scale);

            r_02 = 0.5f32 * (r_02 + c_02 * scale);
            r_12 = 0.5f32 * (r_12 + c_12 * scale);
            r_22 = 0.5f32 * (r_22 + c_22 * scale);
        }

        // Chirality check: Enforce det(R) = +1 by negating column 2
        let det_r5 = r_00 * (r_11 * r_22 - r_12 * r_21)
            - r_01 * (r_10 * r_22 - r_12 * r_20)
            + r_02 * (r_10 * r_21 - r_11 * r_20);
        if det_r5 < 0.0f32 {
            r_02 = -r_02;
            r_12 = -r_12;
            r_22 = -r_22;
        }
    }

    // Store Rotation
    let out_r_base = idx * 9usize;
    out_rotations[out_r_base] = r_00;
    out_rotations[out_r_base + 1usize] = r_10;
    out_rotations[out_r_base + 2usize] = r_20;
    out_rotations[out_r_base + 3usize] = r_01;
    out_rotations[out_r_base + 4usize] = r_11;
    out_rotations[out_r_base + 5usize] = r_21;
    out_rotations[out_r_base + 6usize] = r_02;
    out_rotations[out_r_base + 7usize] = r_12;
    out_rotations[out_r_base + 8usize] = r_22;
}

/// CubeCL GPU kernel transforming Gaussians under volumetric tetrahedral cage deformation:
/// 1. Deforms positions: $\mathbf{x}'_i = \sum_{k=0}^3 w_k \mathbf{x}_{v_k}$
/// 2. Transforms covariances: $M = J \Sigma_{\text{rest}} J^T$ (symmetrized and regularized)
/// 3. Energy-conserving opacity scaling: $\alpha' = 1 - (1 - \alpha)^{1 / \det(J)}$
/// 4. Inverse-ray directional SH evaluation: $\mathbf{d}_{\text{local}} = R^T \mathbf{d}_{\text{world}}$
#[allow(clippy::unnecessary_cast)]
#[cube(launch)]
pub fn deform_gaussians_kernel(
    cage_vertices: &[f32],
    tet_elements: &[u32],
    tet_jacobians: &[f32],
    tet_rotations: &[f32],
    bindings_tet: &[u32],
    bindings_weights: &[f32],
    rest_covariances: &[f32],
    rest_opacities: &[f32],
    sh_coeffs: &[f32],
    out_positions: &mut [f32],
    out_covariances: &mut [f32],
    out_opacities: &mut [f32],
    out_colors: &mut [f32],
    cam_pos_x: f32,
    cam_pos_y: f32,
    cam_pos_z: f32,
    sh_degree: u32,
    num_gaussians: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= num_gaussians as usize {
        terminate!();
    }

    // 1. Barycentric binding lookup
    let tet_idx = bindings_tet[idx] as usize;
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

    // C. Energy-conserving opacity scaling
    let det_j = j_00 * (j_11 * j_22 - j_12 * j_21)
        - j_01 * (j_10 * j_22 - j_12 * j_20)
        + j_02 * (j_10 * j_21 - j_11 * j_20);

    let s_det = f32::clamp(det_j, 0.01f32, 100.0f32);
    let base = f32::clamp(1.0f32 - rest_opacities[idx], 1e-4f32, 1.0f32);
    let def_alpha = f32::clamp(
        1.0f32 - f32::powf(base, 1.0f32 / s_det),
        0.0f32,
        0.99f32,
    );
    out_opacities[idx] = def_alpha;

    // D. Inverse-ray directional SH evaluation
    let r_base = tet_idx * 9usize;
    let r_00 = tet_rotations[r_base];
    let r_10 = tet_rotations[r_base + 1usize];
    let r_20 = tet_rotations[r_base + 2usize];

    let r_01 = tet_rotations[r_base + 3usize];
    let r_11 = tet_rotations[r_base + 4usize];
    let r_21 = tet_rotations[r_base + 5usize];

    let r_02 = tet_rotations[r_base + 6usize];
    let r_12 = tet_rotations[r_base + 7usize];
    let r_22 = tet_rotations[r_base + 8usize];

    let vx = px - cam_pos_x;
    let vy = py - cam_pos_y;
    let vz = pz - cam_pos_z;
    let dist_sq = vx * vx + vy * vy + vz * vz;
    let inv_dist = 1.0f32 / f32::max(1e-4f32, f32::sqrt(dist_sq));
    let dw_x = vx * inv_dist;
    let dw_y = vy * inv_dist;
    let dw_z = vz * inv_dist;

    // d_local = R^T * d_world
    let dl_x = r_00 * dw_x + r_10 * dw_y + r_20 * dw_z;
    let dl_y = r_01 * dw_x + r_11 * dw_y + r_21 * dw_z;
    let dl_z = r_02 * dw_x + r_12 * dw_y + r_22 * dw_z;

    // Spherical harmonics evaluation along d_local
    let sh_base = idx * 48usize;
    let mut color_r = 0.2820948f32 * sh_coeffs[sh_base];
    let mut color_g = 0.2820948f32 * sh_coeffs[sh_base + 1usize];
    let mut color_b = 0.2820948f32 * sh_coeffs[sh_base + 2usize];

    if sh_degree >= 1u32 {
        let y1_m1 = -0.4886025f32 * dl_y;
        let y1_0 = 0.4886025f32 * dl_z;
        let y1_p1 = -0.4886025f32 * dl_x;

        color_r += y1_m1 * sh_coeffs[sh_base + 3usize]
            + y1_0 * sh_coeffs[sh_base + 4usize]
            + y1_p1 * sh_coeffs[sh_base + 5usize];
        color_g += y1_m1 * sh_coeffs[sh_base + 18usize]
            + y1_0 * sh_coeffs[sh_base + 19usize]
            + y1_p1 * sh_coeffs[sh_base + 20usize];
        color_b += y1_m1 * sh_coeffs[sh_base + 33usize]
            + y1_0 * sh_coeffs[sh_base + 34usize]
            + y1_p1 * sh_coeffs[sh_base + 35usize];
    }

    let col_base = idx * 3usize;
    out_colors[col_base] = f32::clamp(color_r + 0.5f32, 0.0f32, 1.0f32);
    out_colors[col_base + 1usize] = f32::clamp(color_g + 0.5f32, 0.0f32, 1.0f32);
    out_colors[col_base + 2usize] = f32::clamp(color_b + 0.5f32, 0.0f32, 1.0f32);
}
