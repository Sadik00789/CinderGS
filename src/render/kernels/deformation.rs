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

/// Evaluates the maximum principal stretch ($\sigma_1$) per tetrahedron from
/// its $3\times 3$ deformation gradient $J$ using closed-form Cardano eigenvalues
/// of the right Cauchy-Green tensor $C = J^T J$.
#[allow(clippy::unnecessary_cast)]
#[cube(launch)]
pub fn compute_tet_strain_kernel(
    tet_jacobians: &[f32],
    out_max_stretch: &mut [f32],
    num_tets: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= num_tets as usize {
        terminate!();
    }

    let j_base = idx * 9usize;
    let j00 = tet_jacobians[j_base];
    let j10 = tet_jacobians[j_base + 1usize];
    let j20 = tet_jacobians[j_base + 2usize];
    let j01 = tet_jacobians[j_base + 3usize];
    let j11 = tet_jacobians[j_base + 4usize];
    let j21 = tet_jacobians[j_base + 5usize];
    let j02 = tet_jacobians[j_base + 6usize];
    let j12 = tet_jacobians[j_base + 7usize];
    let j22 = tet_jacobians[j_base + 8usize];

    // Right Cauchy-Green symmetric matrix C = J^T J
    let c00 = j00 * j00 + j10 * j10 + j20 * j20;
    let c01 = j00 * j01 + j10 * j11 + j20 * j21;
    let c02 = j00 * j02 + j10 * j12 + j20 * j22;
    let c11 = j01 * j01 + j11 * j11 + j21 * j21;
    let c12 = j01 * j02 + j11 * j12 + j21 * j22;
    let c22 = j02 * j02 + j12 * j12 + j22 * j22;

    let p1 = c01 * c01 + c02 * c02 + c12 * c12;
    if p1 <= 1e-10f32 {
        let diag_max = f32::max(c00, f32::max(c11, c22));
        out_max_stretch[idx] = f32::sqrt(f32::max(diag_max, 0.0f32));
        terminate!();
    }

    let q = (c00 + c11 + c22) / 3.0f32;
    let d0 = c00 - q;
    let d1 = c11 - q;
    let d2 = c22 - q;
    let p2 = d0 * d0 + d1 * d1 + d2 * d2 + 2.0f32 * p1;
    let p = f32::sqrt(f32::max(p2 / 6.0f32, 1e-12f32));
    let inv_p = 1.0f32 / p;

    let b00 = d0 * inv_p;
    let b11 = d1 * inv_p;
    let b22 = d2 * inv_p;
    let b01 = c01 * inv_p;
    let b02 = c02 * inv_p;
    let b12 = c12 * inv_p;

    let det_b = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02)
        + b02 * (b01 * b12 - b11 * b02);
    let r = f32::clamp(0.5f32 * det_b, -1.0f32, 1.0f32);
    let phi = f32::acos(r) / 3.0f32;
    let eig_max = q + 2.0f32 * p * f32::cos(phi);
    out_max_stretch[idx] = f32::sqrt(f32::max(eig_max, 0.0f32));
}

/// Deforms Gaussians on the GPU with optional 3D Mip-Splatting variance regularization
/// and transient strain-driven splitting along the deformed covariance major axis.
///
/// Output arrays must be pre-allocated with capacity for `2 * num_gaussians` splats;
/// slot `idx` receives either the unsplit splat or split child 0, while slot
/// `num_gaussians + idx` receives split child 1 when `tet_max_stretch[tet] > split_threshold`.
#[allow(clippy::unnecessary_cast)]
#[cube(launch)]
pub fn deform_and_split_gaussians_kernel(
    cage_vertices: &[f32],
    tet_elements: &[u32],
    tet_jacobians: &[f32],
    tet_max_stretch: &[f32],
    bindings_tet: &[u32],
    bindings_weights: &[f32],
    rest_positions: &[f32],
    rest_covariances: &[f32],
    rest_opacities: &[f32],
    out_positions: &mut [f32],
    out_covariances: &mut [f32],
    out_opacities: &mut [f32],
    out_split_flags: &mut [u32],
    split_threshold: f32,
    mip_filter_3d_var: f32,
    num_gaussians: u32,
    num_tets: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= num_gaussians as usize {
        terminate!();
    }

    let child_idx = num_gaussians as usize + idx;
    out_split_flags[idx] = 0u32;
    out_opacities[child_idx] = 0.0f32;

    let tet_raw = bindings_tet[idx];
    if tet_raw >= num_tets {
        let pos_base = idx * 3usize;
        out_positions[pos_base] = rest_positions[pos_base];
        out_positions[pos_base + 1usize] = rest_positions[pos_base + 1usize];
        out_positions[pos_base + 2usize] = rest_positions[pos_base + 2usize];

        let cov_base = idx * 6usize;
        let s_xx = rest_covariances[cov_base];
        let s_xy = rest_covariances[cov_base + 1usize];
        let s_xz = rest_covariances[cov_base + 2usize];
        let s_yy = rest_covariances[cov_base + 3usize];
        let s_yz = rest_covariances[cov_base + 4usize];
        let s_zz = rest_covariances[cov_base + 5usize];
        let alpha = rest_opacities[idx];

        if mip_filter_3d_var > 0.0f32 {
            let det_orig = f32::max(
                s_xx * (s_yy * s_zz - s_yz * s_yz) - s_xy * (s_xy * s_zz - s_xz * s_yz)
                    + s_xz * (s_xy * s_yz - s_xz * s_yy),
                1e-24f32,
            );
            let r_xx = s_xx + mip_filter_3d_var;
            let r_yy = s_yy + mip_filter_3d_var;
            let r_zz = s_zz + mip_filter_3d_var;
            let det_reg = f32::max(
                r_xx * (r_yy * r_zz - s_yz * s_yz) - s_xy * (s_xy * r_zz - s_xz * s_yz)
                    + s_xz * (s_xy * s_yz - s_xz * r_yy),
                det_orig,
            );
            let comp = f32::sqrt(f32::clamp(det_orig / det_reg, 0.0f32, 1.0f32));
            out_covariances[cov_base] = r_xx;
            out_covariances[cov_base + 1usize] = s_xy;
            out_covariances[cov_base + 2usize] = s_xz;
            out_covariances[cov_base + 3usize] = r_yy;
            out_covariances[cov_base + 4usize] = s_yz;
            out_covariances[cov_base + 5usize] = r_zz;
            out_opacities[idx] = f32::clamp(alpha * comp, 0.0f32, 1.0f32);
        } else {
            out_covariances[cov_base] = s_xx;
            out_covariances[cov_base + 1usize] = s_xy;
            out_covariances[cov_base + 2usize] = s_xz;
            out_covariances[cov_base + 3usize] = s_yy;
            out_covariances[cov_base + 4usize] = s_yz;
            out_covariances[cov_base + 5usize] = s_zz;
            out_opacities[idx] = alpha;
        }
        terminate!();
    }

    let tet_idx = tet_raw as usize;
    let w_base = idx * 4usize;
    let w0 = bindings_weights[w_base];
    let w1 = bindings_weights[w_base + 1usize];
    let w2 = bindings_weights[w_base + 2usize];
    let w3 = bindings_weights[w_base + 3usize];

    let elem_base = tet_idx * 4usize;
    let v0 = tet_elements[elem_base] as usize;
    let v1 = tet_elements[elem_base + 1usize] as usize;
    let v2 = tet_elements[elem_base + 2usize] as usize;
    let v3 = tet_elements[elem_base + 3usize] as usize;

    let px = w0 * cage_vertices[v0 * 3usize]
        + w1 * cage_vertices[v1 * 3usize]
        + w2 * cage_vertices[v2 * 3usize]
        + w3 * cage_vertices[v3 * 3usize];
    let py = w0 * cage_vertices[v0 * 3usize + 1usize]
        + w1 * cage_vertices[v1 * 3usize + 1usize]
        + w2 * cage_vertices[v2 * 3usize + 1usize]
        + w3 * cage_vertices[v3 * 3usize + 1usize];
    let pz = w0 * cage_vertices[v0 * 3usize + 2usize]
        + w1 * cage_vertices[v1 * 3usize + 2usize]
        + w2 * cage_vertices[v2 * 3usize + 2usize]
        + w3 * cage_vertices[v3 * 3usize + 2usize];

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

    let k_00 = j_00 * s_xx + j_01 * s_xy + j_02 * s_xz;
    let k_01 = j_00 * s_xy + j_01 * s_yy + j_02 * s_yz;
    let k_02 = j_00 * s_xz + j_01 * s_yz + j_02 * s_zz;
    let k_10 = j_10 * s_xx + j_11 * s_xy + j_12 * s_xz;
    let k_11 = j_10 * s_xy + j_11 * s_yy + j_12 * s_yz;
    let k_12 = j_10 * s_xz + j_11 * s_yz + j_12 * s_zz;
    let k_20 = j_20 * s_xx + j_21 * s_xy + j_22 * s_xz;
    let k_21 = j_20 * s_xy + j_21 * s_yy + j_22 * s_yz;
    let k_22 = j_20 * s_xz + j_21 * s_yz + j_22 * s_zz;

    let m00 = f32::max(1e-4f32, k_00 * j_00 + k_01 * j_01 + k_02 * j_02);
    let m01 = 0.5f32
        * ((k_00 * j_10 + k_01 * j_11 + k_02 * j_12) + (k_10 * j_00 + k_11 * j_01 + k_12 * j_02));
    let m02 = 0.5f32
        * ((k_00 * j_20 + k_01 * j_21 + k_02 * j_22) + (k_20 * j_00 + k_21 * j_01 + k_22 * j_02));
    let m11 = f32::max(1e-4f32, k_10 * j_10 + k_11 * j_11 + k_12 * j_12);
    let m12 = 0.5f32
        * ((k_10 * j_20 + k_11 * j_21 + k_12 * j_22) + (k_20 * j_10 + k_21 * j_11 + k_22 * j_12));
    let m22 = f32::max(1e-4f32, k_20 * j_20 + k_21 * j_21 + k_22 * j_22);

    let det_j = j_00 * (j_11 * j_22 - j_12 * j_21) - j_01 * (j_10 * j_22 - j_12 * j_20)
        + j_02 * (j_10 * j_21 - j_11 * j_20);
    let s_det = f32::clamp(det_j, 0.6f32, 1.8f32);
    let safe_base = f32::clamp(1.0f32 - rest_opacities[idx], 1e-4f32, 0.999f32);
    let alpha_prime = f32::clamp(
        1.0f32 - f32::powf(safe_base, 1.0f32 / s_det),
        0.05f32,
        0.99f32,
    );

    let should_split = split_threshold > 0.0f32 && tet_max_stretch[tet_idx] > split_threshold;

    if !should_split {
        let pos_base = idx * 3usize;
        out_positions[pos_base] = px;
        out_positions[pos_base + 1usize] = py;
        out_positions[pos_base + 2usize] = pz;

        if mip_filter_3d_var > 0.0f32 {
            let det_orig = f32::max(
                m00 * (m11 * m22 - m12 * m12) - m01 * (m01 * m22 - m12 * m02)
                    + m02 * (m01 * m12 - m11 * m02),
                1e-24f32,
            );
            let r00 = m00 + mip_filter_3d_var;
            let r11 = m11 + mip_filter_3d_var;
            let r22 = m22 + mip_filter_3d_var;
            let det_reg = f32::max(
                r00 * (r11 * r22 - m12 * m12) - m01 * (m01 * r22 - m12 * m02)
                    + m02 * (m01 * m12 - r11 * m02),
                det_orig,
            );
            let comp = f32::sqrt(f32::clamp(det_orig / det_reg, 0.0f32, 1.0f32));
            out_covariances[cov_base] = r00;
            out_covariances[cov_base + 1usize] = m01;
            out_covariances[cov_base + 2usize] = m02;
            out_covariances[cov_base + 3usize] = r11;
            out_covariances[cov_base + 4usize] = m12;
            out_covariances[cov_base + 5usize] = r22;
            out_opacities[idx] = f32::clamp(alpha_prime * comp, 0.0f32, 1.0f32);
        } else {
            out_covariances[cov_base] = m00;
            out_covariances[cov_base + 1usize] = m01;
            out_covariances[cov_base + 2usize] = m02;
            out_covariances[cov_base + 3usize] = m11;
            out_covariances[cov_base + 4usize] = m12;
            out_covariances[cov_base + 5usize] = m22;
            out_opacities[idx] = alpha_prime;
        }
        terminate!();
    }

    // Compute largest eigenvalue and major eigenvector of symmetric M
    let p1 = m01 * m01 + m02 * m02 + m12 * m12;
    let mut lambda0 = f32::max(m00, f32::max(m11, m22));
    let mut ux = 1.0f32;
    let mut uy = 0.0f32;
    let mut uz = 0.0f32;

    if p1 <= 1e-10f32 {
        if m11 >= m00 && m11 >= m22 {
            ux = 0.0f32;
            uy = 1.0f32;
            uz = 0.0f32;
        } else if m22 >= m00 && m22 >= m11 {
            ux = 0.0f32;
            uy = 0.0f32;
            uz = 1.0f32;
        }
    } else {
        let q = (m00 + m11 + m22) / 3.0f32;
        let d0 = m00 - q;
        let d1 = m11 - q;
        let d2 = m22 - q;
        let p2 = d0 * d0 + d1 * d1 + d2 * d2 + 2.0f32 * p1;
        let p = f32::sqrt(f32::max(p2 / 6.0f32, 1e-12f32));
        let inv_p = 1.0f32 / p;
        let b00 = d0 * inv_p;
        let b11 = d1 * inv_p;
        let b22 = d2 * inv_p;
        let b01 = m01 * inv_p;
        let b02 = m02 * inv_p;
        let b12 = m12 * inv_p;
        let det_b = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02)
            + b02 * (b01 * b12 - b11 * b02);
        let r = f32::clamp(0.5f32 * det_b, -1.0f32, 1.0f32);
        let phi = f32::acos(r) / 3.0f32;
        lambda0 = q + 2.0f32 * p * f32::cos(phi);

        // Major eigenvector from cross products of rows of (M - lambda0 * I)
        let r0x = m00 - lambda0;
        let r0y = m01;
        let r0z = m02;
        let r1x = m01;
        let r1y = m11 - lambda0;
        let r1z = m12;
        let r2x = m02;
        let r2y = m12;
        let r2z = m22 - lambda0;

        let c01x = r0y * r1z - r0z * r1y;
        let c01y = r0z * r1x - r0x * r1z;
        let c01z = r0x * r1y - r0y * r1x;
        let n01 = c01x * c01x + c01y * c01y + c01z * c01z;

        let c12x = r1y * r2z - r1z * r2y;
        let c12y = r1z * r2x - r1x * r2z;
        let c12z = r1x * r2y - r1y * r2x;
        let n12 = c12x * c12x + c12y * c12y + c12z * c12z;

        let c02x = r0y * r2z - r0z * r2y;
        let c02y = r0z * r2x - r0x * r2z;
        let c02z = r0x * r2y - r0y * r2x;
        let n02 = c02x * c02x + c02y * c02y + c02z * c02z;

        if n01 >= n12 && n01 >= n02 && n01 > 1e-12f32 {
            let inv = 1.0f32 / f32::sqrt(n01);
            ux = c01x * inv;
            uy = c01y * inv;
            uz = c01z * inv;
        } else if n12 >= n02 && n12 > 1e-12f32 {
            let inv = 1.0f32 / f32::sqrt(n12);
            ux = c12x * inv;
            uy = c12y * inv;
            uz = c12z * inv;
        } else if n02 > 1e-12f32 {
            let inv = 1.0f32 / f32::sqrt(n02);
            ux = c02x * inv;
            uy = c02y * inv;
            uz = c02z * inv;
        }
    }

    let safe_l0 = f32::max(lambda0, 1e-8f32);
    let offset_mag = 0.5f32 * f32::sqrt(safe_l0);
    let dx = ux * offset_mag;
    let dy = uy * offset_mag;
    let dz = uz * offset_mag;

    // Shrink variance along major axis to 1/4: Sigma_child = M - 0.75 * lambda0 * u u^T
    let shrink = 0.75f32 * safe_l0;
    let mut c00 = f32::max(1e-8f32, m00 - shrink * ux * ux);
    let c01 = m01 - shrink * ux * uy;
    let c02 = m02 - shrink * ux * uz;
    let mut c11 = f32::max(1e-8f32, m11 - shrink * uy * uy);
    let c12 = m12 - shrink * uy * uz;
    let mut c22 = f32::max(1e-8f32, m22 - shrink * uz * uz);
    let mut child_alpha = alpha_prime;

    if mip_filter_3d_var > 0.0f32 {
        let det_orig = f32::max(
            c00 * (c11 * c22 - c12 * c12) - c01 * (c01 * c22 - c12 * c02)
                + c02 * (c01 * c12 - c11 * c02),
            1e-24f32,
        );
        c00 += mip_filter_3d_var;
        c11 += mip_filter_3d_var;
        c22 += mip_filter_3d_var;
        let det_reg = f32::max(
            c00 * (c11 * c22 - c12 * c12) - c01 * (c01 * c22 - c12 * c02)
                + c02 * (c01 * c12 - c11 * c02),
            det_orig,
        );
        let comp = f32::sqrt(f32::clamp(det_orig / det_reg, 0.0f32, 1.0f32));
        child_alpha = f32::clamp(child_alpha * comp, 0.0f32, 1.0f32);
    }

    let pos_base0 = idx * 3usize;
    out_positions[pos_base0] = px - dx;
    out_positions[pos_base0 + 1usize] = py - dy;
    out_positions[pos_base0 + 2usize] = pz - dz;

    out_covariances[cov_base] = c00;
    out_covariances[cov_base + 1usize] = c01;
    out_covariances[cov_base + 2usize] = c02;
    out_covariances[cov_base + 3usize] = c11;
    out_covariances[cov_base + 4usize] = c12;
    out_covariances[cov_base + 5usize] = c22;
    out_opacities[idx] = child_alpha;

    let pos_base1 = child_idx * 3usize;
    out_positions[pos_base1] = px + dx;
    out_positions[pos_base1 + 1usize] = py + dy;
    out_positions[pos_base1 + 2usize] = pz + dz;

    let cov_base1 = child_idx * 6usize;
    out_covariances[cov_base1] = c00;
    out_covariances[cov_base1 + 1usize] = c01;
    out_covariances[cov_base1 + 2usize] = c02;
    out_covariances[cov_base1 + 3usize] = c11;
    out_covariances[cov_base1 + 4usize] = c12;
    out_covariances[cov_base1 + 5usize] = c22;
    out_opacities[child_idx] = child_alpha;

    out_split_flags[idx] = 1u32;
}

/// Evaluates each tetrahedron's deformation gradient $J$, pure rigid polar rotation
/// $R \in \text{SO}(3)$ (5-step Higham Newton-Schulz iteration with chirality fix),
/// and maximum principal stretch $\sigma_1$ in a single pass.
///
/// Both `out_jacobians` and `out_rotations` are stored column-major (`9` floats per tet).
#[allow(clippy::unnecessary_cast)]
#[cube(launch)]
pub fn compute_tet_transforms_kernel(
    cage_vertices: &[f32],
    tet_elements: &[u32],
    tet_inv_dm: &[f32],
    out_jacobians: &mut [f32],
    out_rotations: &mut [f32],
    out_max_stretch: &mut [f32],
    num_tets: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= num_tets as usize {
        terminate!();
    }

    let elem_base = idx * 4usize;
    let v0 = tet_elements[elem_base] as usize;
    let v1 = tet_elements[elem_base + 1usize] as usize;
    let v2 = tet_elements[elem_base + 2usize] as usize;
    let v3 = tet_elements[elem_base + 3usize] as usize;

    let x0_x = cage_vertices[v0 * 3usize];
    let x0_y = cage_vertices[v0 * 3usize + 1usize];
    let x0_z = cage_vertices[v0 * 3usize + 2usize];

    let ds_00 = cage_vertices[v1 * 3usize] - x0_x;
    let ds_10 = cage_vertices[v1 * 3usize + 1usize] - x0_y;
    let ds_20 = cage_vertices[v1 * 3usize + 2usize] - x0_z;

    let ds_01 = cage_vertices[v2 * 3usize] - x0_x;
    let ds_11 = cage_vertices[v2 * 3usize + 1usize] - x0_y;
    let ds_21 = cage_vertices[v2 * 3usize + 2usize] - x0_z;

    let ds_02 = cage_vertices[v3 * 3usize] - x0_x;
    let ds_12 = cage_vertices[v3 * 3usize + 1usize] - x0_y;
    let ds_22 = cage_vertices[v3 * 3usize + 2usize] - x0_z;

    let mat_base = idx * 9usize;
    let a_00 = tet_inv_dm[mat_base];
    let a_10 = tet_inv_dm[mat_base + 1usize];
    let a_20 = tet_inv_dm[mat_base + 2usize];
    let a_01 = tet_inv_dm[mat_base + 3usize];
    let a_11 = tet_inv_dm[mat_base + 4usize];
    let a_21 = tet_inv_dm[mat_base + 5usize];
    let a_02 = tet_inv_dm[mat_base + 6usize];
    let a_12 = tet_inv_dm[mat_base + 7usize];
    let a_22 = tet_inv_dm[mat_base + 8usize];

    let j_00 = ds_00 * a_00 + ds_01 * a_10 + ds_02 * a_20;
    let j_10 = ds_10 * a_00 + ds_11 * a_10 + ds_12 * a_20;
    let j_20 = ds_20 * a_00 + ds_21 * a_10 + ds_22 * a_20;

    let j_01 = ds_00 * a_01 + ds_01 * a_11 + ds_02 * a_21;
    let j_11 = ds_10 * a_01 + ds_11 * a_11 + ds_12 * a_21;
    let j_21 = ds_20 * a_01 + ds_21 * a_11 + ds_22 * a_21;

    let j_02 = ds_00 * a_02 + ds_01 * a_12 + ds_02 * a_22;
    let j_12 = ds_10 * a_02 + ds_11 * a_12 + ds_12 * a_22;
    let j_22 = ds_20 * a_02 + ds_21 * a_12 + ds_22 * a_22;

    out_jacobians[mat_base] = j_00;
    out_jacobians[mat_base + 1usize] = j_10;
    out_jacobians[mat_base + 2usize] = j_20;
    out_jacobians[mat_base + 3usize] = j_01;
    out_jacobians[mat_base + 4usize] = j_11;
    out_jacobians[mat_base + 5usize] = j_21;
    out_jacobians[mat_base + 6usize] = j_02;
    out_jacobians[mat_base + 7usize] = j_12;
    out_jacobians[mat_base + 8usize] = j_22;

    // 1. Max principal stretch via right Cauchy-Green C = J^T J
    let c00 = j_00 * j_00 + j_10 * j_10 + j_20 * j_20;
    let c01 = j_00 * j_01 + j_10 * j_11 + j_20 * j_21;
    let c02 = j_00 * j_02 + j_10 * j_12 + j_20 * j_22;
    let c11 = j_01 * j_01 + j_11 * j_11 + j_21 * j_21;
    let c12 = j_01 * j_02 + j_11 * j_12 + j_21 * j_22;
    let c22 = j_02 * j_02 + j_12 * j_12 + j_22 * j_22;

    let p1 = c01 * c01 + c02 * c02 + c12 * c12;
    if p1 <= 1e-10f32 {
        let diag_max = f32::max(c00, f32::max(c11, c22));
        out_max_stretch[idx] = f32::sqrt(f32::max(diag_max, 0.0f32));
    } else {
        let q = (c00 + c11 + c22) / 3.0f32;
        let d0 = c00 - q;
        let d1 = c11 - q;
        let d2 = c22 - q;
        let p2 = d0 * d0 + d1 * d1 + d2 * d2 + 2.0f32 * p1;
        let p = f32::sqrt(f32::max(p2 / 6.0f32, 1e-12f32));
        let inv_p = 1.0f32 / p;

        let b00 = d0 * inv_p;
        let b11 = d1 * inv_p;
        let b22 = d2 * inv_p;
        let b01 = c01 * inv_p;
        let b02 = c02 * inv_p;
        let b12 = c12 * inv_p;

        let det_b = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02)
            + b02 * (b01 * b12 - b11 * b02);
        let r_eig = f32::clamp(0.5f32 * det_b, -1.0f32, 1.0f32);
        let phi = f32::acos(r_eig) / 3.0f32;
        let eig_max = q + 2.0f32 * p * f32::cos(phi);
        out_max_stretch[idx] = f32::sqrt(f32::max(eig_max, 0.0f32));
    }

    // 2. Higham Polar Decomposition J = R * S via 5-step Newton-Schulz
    let det_j = j_00 * (j_11 * j_22 - j_12 * j_21) - j_01 * (j_10 * j_22 - j_12 * j_20)
        + j_02 * (j_10 * j_21 - j_11 * j_20);
    let abs_det = f32::abs(det_j);

    if abs_det < 1e-8f32 {
        out_rotations[mat_base] = 1.0f32;
        out_rotations[mat_base + 1usize] = 0.0f32;
        out_rotations[mat_base + 2usize] = 0.0f32;
        out_rotations[mat_base + 3usize] = 0.0f32;
        out_rotations[mat_base + 4usize] = 1.0f32;
        out_rotations[mat_base + 5usize] = 0.0f32;
        out_rotations[mat_base + 6usize] = 0.0f32;
        out_rotations[mat_base + 7usize] = 0.0f32;
        out_rotations[mat_base + 8usize] = 1.0f32;
        terminate!();
    }

    let gamma = f32::exp(-f32::ln(abs_det) / 3.0f32);
    let mut r00 = j_00 * gamma;
    let mut r10 = j_10 * gamma;
    let mut r20 = j_20 * gamma;
    let mut r01 = j_01 * gamma;
    let mut r11 = j_11 * gamma;
    let mut r21 = j_21 * gamma;
    let mut r02 = j_02 * gamma;
    let mut r12 = j_12 * gamma;
    let mut r22 = j_22 * gamma;

    for _ in 0u32..5u32 {
        // Cofactor columns of R: (c1 x c2), (c2 x c0), (c0 x c1)
        let cof00 = r11 * r22 - r21 * r12;
        let cof10 = r21 * r02 - r01 * r22;
        let cof20 = r01 * r12 - r11 * r02;

        let cof01 = r12 * r20 - r22 * r10;
        let cof11 = r22 * r00 - r02 * r20;
        let cof21 = r02 * r10 - r12 * r00;

        let cof02 = r10 * r21 - r20 * r11;
        let cof12 = r20 * r01 - r00 * r21;
        let cof22 = r00 * r11 - r10 * r01;

        let det_r = r00 * cof00 + r10 * cof10 + r20 * cof20;
        if f32::abs(det_r) > 1e-12f32 {
            let inv_det = 1.0f32 / det_r;
            r00 = 0.5f32 * (r00 + cof00 * inv_det);
            r10 = 0.5f32 * (r10 + cof10 * inv_det);
            r20 = 0.5f32 * (r20 + cof20 * inv_det);
            r01 = 0.5f32 * (r01 + cof01 * inv_det);
            r11 = 0.5f32 * (r11 + cof11 * inv_det);
            r21 = 0.5f32 * (r21 + cof21 * inv_det);
            r02 = 0.5f32 * (r02 + cof02 * inv_det);
            r12 = 0.5f32 * (r12 + cof12 * inv_det);
            r22 = 0.5f32 * (r22 + cof22 * inv_det);
        }
    }

    let det_final = r00 * (r11 * r22 - r21 * r12)
        + r10 * (r21 * r02 - r01 * r22)
        + r20 * (r01 * r12 - r11 * r02);
    if det_final < 0.0f32 {
        r02 = -r02;
        r12 = -r12;
        r22 = -r22;
    }

    out_rotations[mat_base] = r00;
    out_rotations[mat_base + 1usize] = r10;
    out_rotations[mat_base + 2usize] = r20;
    out_rotations[mat_base + 3usize] = r01;
    out_rotations[mat_base + 4usize] = r11;
    out_rotations[mat_base + 5usize] = r21;
    out_rotations[mat_base + 6usize] = r02;
    out_rotations[mat_base + 7usize] = r12;
    out_rotations[mat_base + 8usize] = r22;
}

/// Deforms Gaussians on the GPU with optional 3D Mip-Splatting variance regularization
/// and atomically compacts split child 1 Gaussians at `num_gaussians + child_offset`.
///
/// `split_counter[0]` must be initialized to `0` prior to launch. Each split writes a
/// `[parent_idx, tet_raw]` metadata pair into `out_split_meta` at `2 * child_offset`.
#[allow(clippy::unnecessary_cast)]
#[cube(launch)]
pub fn deform_and_split_compact_kernel(
    cage_vertices: &[f32],
    tet_elements: &[u32],
    tet_jacobians: &[f32],
    tet_max_stretch: &[f32],
    bindings_tet: &[u32],
    bindings_weights: &[f32],
    rest_positions: &[f32],
    rest_covariances: &[f32],
    rest_opacities: &[f32],
    out_positions: &mut [f32],
    out_covariances: &mut [f32],
    out_opacities: &mut [f32],
    out_split_meta: &mut [u32],
    split_counter: &mut [Atomic<u32>],
    split_threshold: f32,
    mip_filter_3d_var: f32,
    num_gaussians: u32,
    num_tets: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= num_gaussians as usize {
        terminate!();
    }

    let tet_raw = bindings_tet[idx];
    if tet_raw >= num_tets {
        let pos_base = idx * 3usize;
        out_positions[pos_base] = rest_positions[pos_base];
        out_positions[pos_base + 1usize] = rest_positions[pos_base + 1usize];
        out_positions[pos_base + 2usize] = rest_positions[pos_base + 2usize];

        let cov_base = idx * 6usize;
        let s_xx = rest_covariances[cov_base];
        let s_xy = rest_covariances[cov_base + 1usize];
        let s_xz = rest_covariances[cov_base + 2usize];
        let s_yy = rest_covariances[cov_base + 3usize];
        let s_yz = rest_covariances[cov_base + 4usize];
        let s_zz = rest_covariances[cov_base + 5usize];
        let alpha = rest_opacities[idx];

        if mip_filter_3d_var > 0.0f32 {
            let det_orig = f32::max(
                s_xx * (s_yy * s_zz - s_yz * s_yz) - s_xy * (s_xy * s_zz - s_xz * s_yz)
                    + s_xz * (s_xy * s_yz - s_xz * s_yy),
                1e-24f32,
            );
            let r_xx = s_xx + mip_filter_3d_var;
            let r_yy = s_yy + mip_filter_3d_var;
            let r_zz = s_zz + mip_filter_3d_var;
            let det_reg = f32::max(
                r_xx * (r_yy * r_zz - s_yz * s_yz) - s_xy * (s_xy * r_zz - s_xz * s_yz)
                    + s_xz * (s_xy * s_yz - s_xz * r_yy),
                det_orig,
            );
            let comp = f32::sqrt(f32::clamp(det_orig / det_reg, 0.0f32, 1.0f32));
            out_covariances[cov_base] = r_xx;
            out_covariances[cov_base + 1usize] = s_xy;
            out_covariances[cov_base + 2usize] = s_xz;
            out_covariances[cov_base + 3usize] = r_yy;
            out_covariances[cov_base + 4usize] = s_yz;
            out_covariances[cov_base + 5usize] = r_zz;
            out_opacities[idx] = f32::clamp(alpha * comp, 0.0f32, 1.0f32);
        } else {
            out_covariances[cov_base] = s_xx;
            out_covariances[cov_base + 1usize] = s_xy;
            out_covariances[cov_base + 2usize] = s_xz;
            out_covariances[cov_base + 3usize] = s_yy;
            out_covariances[cov_base + 4usize] = s_yz;
            out_covariances[cov_base + 5usize] = s_zz;
            out_opacities[idx] = alpha;
        }
        terminate!();
    }

    let tet_idx = tet_raw as usize;
    let w_base = idx * 4usize;
    let w0 = bindings_weights[w_base];
    let w1 = bindings_weights[w_base + 1usize];
    let w2 = bindings_weights[w_base + 2usize];
    let w3 = bindings_weights[w_base + 3usize];

    let elem_base = tet_idx * 4usize;
    let v0 = tet_elements[elem_base] as usize;
    let v1 = tet_elements[elem_base + 1usize] as usize;
    let v2 = tet_elements[elem_base + 2usize] as usize;
    let v3 = tet_elements[elem_base + 3usize] as usize;

    let px = w0 * cage_vertices[v0 * 3usize]
        + w1 * cage_vertices[v1 * 3usize]
        + w2 * cage_vertices[v2 * 3usize]
        + w3 * cage_vertices[v3 * 3usize];
    let py = w0 * cage_vertices[v0 * 3usize + 1usize]
        + w1 * cage_vertices[v1 * 3usize + 1usize]
        + w2 * cage_vertices[v2 * 3usize + 1usize]
        + w3 * cage_vertices[v3 * 3usize + 1usize];
    let pz = w0 * cage_vertices[v0 * 3usize + 2usize]
        + w1 * cage_vertices[v1 * 3usize + 2usize]
        + w2 * cage_vertices[v2 * 3usize + 2usize]
        + w3 * cage_vertices[v3 * 3usize + 2usize];

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

    let k_00 = j_00 * s_xx + j_01 * s_xy + j_02 * s_xz;
    let k_01 = j_00 * s_xy + j_01 * s_yy + j_02 * s_yz;
    let k_02 = j_00 * s_xz + j_01 * s_yz + j_02 * s_zz;
    let k_10 = j_10 * s_xx + j_11 * s_xy + j_12 * s_xz;
    let k_11 = j_10 * s_xy + j_11 * s_yy + j_12 * s_yz;
    let k_12 = j_10 * s_xz + j_11 * s_yz + j_12 * s_zz;
    let k_20 = j_20 * s_xx + j_21 * s_xy + j_22 * s_xz;
    let k_21 = j_20 * s_xy + j_21 * s_yy + j_22 * s_yz;
    let k_22 = j_20 * s_xz + j_21 * s_yz + j_22 * s_zz;

    let m00 = f32::max(1e-4f32, k_00 * j_00 + k_01 * j_01 + k_02 * j_02);
    let m01 = 0.5f32
        * ((k_00 * j_10 + k_01 * j_11 + k_02 * j_12) + (k_10 * j_00 + k_11 * j_01 + k_12 * j_02));
    let m02 = 0.5f32
        * ((k_00 * j_20 + k_01 * j_21 + k_02 * j_22) + (k_20 * j_00 + k_21 * j_01 + k_22 * j_02));
    let m11 = f32::max(1e-4f32, k_10 * j_10 + k_11 * j_11 + k_12 * j_12);
    let m12 = 0.5f32
        * ((k_10 * j_20 + k_11 * j_21 + k_12 * j_22) + (k_20 * j_10 + k_21 * j_11 + k_22 * j_12));
    let m22 = f32::max(1e-4f32, k_20 * j_20 + k_21 * j_21 + k_22 * j_22);

    let det_j = j_00 * (j_11 * j_22 - j_12 * j_21) - j_01 * (j_10 * j_22 - j_12 * j_20)
        + j_02 * (j_10 * j_21 - j_11 * j_20);
    let s_det = f32::clamp(det_j, 0.6f32, 1.8f32);
    let safe_base = f32::clamp(1.0f32 - rest_opacities[idx], 1e-4f32, 0.999f32);
    let alpha_prime = f32::clamp(
        1.0f32 - f32::powf(safe_base, 1.0f32 / s_det),
        0.05f32,
        0.99f32,
    );

    let should_split = split_threshold > 0.0f32 && tet_max_stretch[tet_idx] > split_threshold;

    if !should_split {
        let pos_base = idx * 3usize;
        out_positions[pos_base] = px;
        out_positions[pos_base + 1usize] = py;
        out_positions[pos_base + 2usize] = pz;

        if mip_filter_3d_var > 0.0f32 {
            let det_orig = f32::max(
                m00 * (m11 * m22 - m12 * m12) - m01 * (m01 * m22 - m12 * m02)
                    + m02 * (m01 * m12 - m11 * m02),
                1e-24f32,
            );
            let r00 = m00 + mip_filter_3d_var;
            let r11 = m11 + mip_filter_3d_var;
            let r22 = m22 + mip_filter_3d_var;
            let det_reg = f32::max(
                r00 * (r11 * r22 - m12 * m12) - m01 * (m01 * r22 - m12 * m02)
                    + m02 * (m01 * m12 - r11 * m02),
                det_orig,
            );
            let comp = f32::sqrt(f32::clamp(det_orig / det_reg, 0.0f32, 1.0f32));
            out_covariances[cov_base] = r00;
            out_covariances[cov_base + 1usize] = m01;
            out_covariances[cov_base + 2usize] = m02;
            out_covariances[cov_base + 3usize] = r11;
            out_covariances[cov_base + 4usize] = m12;
            out_covariances[cov_base + 5usize] = r22;
            out_opacities[idx] = f32::clamp(alpha_prime * comp, 0.0f32, 1.0f32);
        } else {
            out_covariances[cov_base] = m00;
            out_covariances[cov_base + 1usize] = m01;
            out_covariances[cov_base + 2usize] = m02;
            out_covariances[cov_base + 3usize] = m11;
            out_covariances[cov_base + 4usize] = m12;
            out_covariances[cov_base + 5usize] = m22;
            out_opacities[idx] = alpha_prime;
        }
        terminate!();
    }

    let p1 = m01 * m01 + m02 * m02 + m12 * m12;
    let mut lambda0 = f32::max(m00, f32::max(m11, m22));
    let mut ux = 1.0f32;
    let mut uy = 0.0f32;
    let mut uz = 0.0f32;

    if p1 <= 1e-10f32 {
        if m11 >= m00 && m11 >= m22 {
            ux = 0.0f32;
            uy = 1.0f32;
            uz = 0.0f32;
        } else if m22 >= m00 && m22 >= m11 {
            ux = 0.0f32;
            uy = 0.0f32;
            uz = 1.0f32;
        }
    } else {
        let q = (m00 + m11 + m22) / 3.0f32;
        let d0 = m00 - q;
        let d1 = m11 - q;
        let d2 = m22 - q;
        let p2 = d0 * d0 + d1 * d1 + d2 * d2 + 2.0f32 * p1;
        let p = f32::sqrt(f32::max(p2 / 6.0f32, 1e-12f32));
        let inv_p = 1.0f32 / p;
        let b00 = d0 * inv_p;
        let b11 = d1 * inv_p;
        let b22 = d2 * inv_p;
        let b01 = m01 * inv_p;
        let b02 = m02 * inv_p;
        let b12 = m12 * inv_p;
        let det_b = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02)
            + b02 * (b01 * b12 - b11 * b02);
        let r = f32::clamp(0.5f32 * det_b, -1.0f32, 1.0f32);
        let phi = f32::acos(r) / 3.0f32;
        lambda0 = q + 2.0f32 * p * f32::cos(phi);

        let r0x = m00 - lambda0;
        let r0y = m01;
        let r0z = m02;
        let r1x = m01;
        let r1y = m11 - lambda0;
        let r1z = m12;
        let r2x = m02;
        let r2y = m12;
        let r2z = m22 - lambda0;

        let c01x = r0y * r1z - r0z * r1y;
        let c01y = r0z * r1x - r0x * r1z;
        let c01z = r0x * r1y - r0y * r1x;
        let n01 = c01x * c01x + c01y * c01y + c01z * c01z;

        let c12x = r1y * r2z - r1z * r2y;
        let c12y = r1z * r2x - r1x * r2z;
        let c12z = r1x * r2y - r1y * r2x;
        let n12 = c12x * c12x + c12y * c12y + c12z * c12z;

        let c02x = r0y * r2z - r0z * r2y;
        let c02y = r0z * r2x - r0x * r2z;
        let c02z = r0x * r2y - r0y * r2x;
        let n02 = c02x * c02x + c02y * c02y + c02z * c02z;

        if n01 >= n12 && n01 >= n02 && n01 > 1e-12f32 {
            let inv = 1.0f32 / f32::sqrt(n01);
            ux = c01x * inv;
            uy = c01y * inv;
            uz = c01z * inv;
        } else if n12 >= n02 && n12 > 1e-12f32 {
            let inv = 1.0f32 / f32::sqrt(n12);
            ux = c12x * inv;
            uy = c12y * inv;
            uz = c12z * inv;
        } else if n02 > 1e-12f32 {
            let inv = 1.0f32 / f32::sqrt(n02);
            ux = c02x * inv;
            uy = c02y * inv;
            uz = c02z * inv;
        }
    }

    let safe_l0 = f32::max(lambda0, 1e-8f32);
    let offset_mag = 0.5f32 * f32::sqrt(safe_l0);
    let dx = ux * offset_mag;
    let dy = uy * offset_mag;
    let dz = uz * offset_mag;

    let shrink = 0.75f32 * safe_l0;
    let mut c00 = f32::max(1e-8f32, m00 - shrink * ux * ux);
    let c01 = m01 - shrink * ux * uy;
    let c02 = m02 - shrink * ux * uz;
    let mut c11 = f32::max(1e-8f32, m11 - shrink * uy * uy);
    let c12 = m12 - shrink * uy * uz;
    let mut c22 = f32::max(1e-8f32, m22 - shrink * uz * uz);
    let mut child_alpha = alpha_prime;

    if mip_filter_3d_var > 0.0f32 {
        let det_orig = f32::max(
            c00 * (c11 * c22 - c12 * c12) - c01 * (c01 * c22 - c12 * c02)
                + c02 * (c01 * c12 - c11 * c02),
            1e-24f32,
        );
        c00 += mip_filter_3d_var;
        c11 += mip_filter_3d_var;
        c22 += mip_filter_3d_var;
        let det_reg = f32::max(
            c00 * (c11 * c22 - c12 * c12) - c01 * (c01 * c22 - c12 * c02)
                + c02 * (c01 * c12 - c11 * c02),
            det_orig,
        );
        let comp = f32::sqrt(f32::clamp(det_orig / det_reg, 0.0f32, 1.0f32));
        child_alpha = f32::clamp(child_alpha * comp, 0.0f32, 1.0f32);
    }

    let pos_base0 = idx * 3usize;
    out_positions[pos_base0] = px - dx;
    out_positions[pos_base0 + 1usize] = py - dy;
    out_positions[pos_base0 + 2usize] = pz - dz;

    out_covariances[cov_base] = c00;
    out_covariances[cov_base + 1usize] = c01;
    out_covariances[cov_base + 2usize] = c02;
    out_covariances[cov_base + 3usize] = c11;
    out_covariances[cov_base + 4usize] = c12;
    out_covariances[cov_base + 5usize] = c22;
    out_opacities[idx] = child_alpha;

    let child_offset = split_counter[0usize].fetch_add(1u32) as usize;
    let child_idx = num_gaussians as usize + child_offset;

    let pos_base1 = child_idx * 3usize;
    out_positions[pos_base1] = px + dx;
    out_positions[pos_base1 + 1usize] = py + dy;
    out_positions[pos_base1 + 2usize] = pz + dz;

    let cov_base1 = child_idx * 6usize;
    out_covariances[cov_base1] = c00;
    out_covariances[cov_base1 + 1usize] = c01;
    out_covariances[cov_base1 + 2usize] = c02;
    out_covariances[cov_base1 + 3usize] = c11;
    out_covariances[cov_base1 + 4usize] = c12;
    out_covariances[cov_base1 + 5usize] = c22;
    out_opacities[child_idx] = child_alpha;

    let meta_base = child_offset * 2usize;
    out_split_meta[meta_base] = idx as u32;
    out_split_meta[meta_base + 1usize] = tet_raw;
}
