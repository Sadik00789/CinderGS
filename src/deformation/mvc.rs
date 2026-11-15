use crate::deformation::cpu_reference::DeformedSceneOutput;
use crate::deformation::{
    TetElement, TetMesh, higham_polar_decomposition, jacobian_singular_values,
};
use crate::math::covariance::{apply_mip_filter_3d, split_gaussian_along_major_axis};
use crate::math::covariance_matrix_from_slice;
use crate::parallel;
use crate::render::rasterizer::sh_rotation::ShRotator;
use crate::scene::GaussianSceneSoa;
use glam::{Mat3, Vec3};
use std::collections::HashMap;
use std::f32::consts::PI;

/// Arbitrary closed triangular surface cage supporting 3D Mean Value Coordinates (MVC).
#[derive(Clone, Debug, PartialEq)]
pub struct PolyCageMesh {
    /// Rest vertex positions of the surface cage.
    pub rest_vertices: Vec<[f32; 3]>,
    /// Outward-oriented triangle connectivity referencing `rest_vertices`.
    pub triangles: Vec<[u32; 3]>,
}

/// Precomputed Mean Value Coordinate bindings for a batch of Gaussians.
#[derive(Clone, Debug, PartialEq)]
pub struct MvcBindings {
    /// Flattened dense weight table `[num_gaussians * num_cage_vertices]`.
    pub weights: Vec<f32>,
    /// Number of cage vertices per Gaussian row.
    pub num_cage_vertices: usize,
    /// Whether each Gaussian lies inside the closed cage surface.
    pub bound_mask: Vec<bool>,
}

impl MvcBindings {
    /// Number of Gaussians represented in this binding table.
    #[inline]
    pub fn len(&self) -> usize {
        self.bound_mask.len()
    }

    /// Returns true if zero Gaussians are stored.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bound_mask.is_empty()
    }

    /// Slice of MVC weights for Gaussian `idx`.
    #[inline]
    pub fn weights_for(&self, idx: usize) -> &[f32] {
        let start = idx * self.num_cage_vertices;
        &self.weights[start..start + self.num_cage_vertices]
    }
}

impl PolyCageMesh {
    /// Constructs a closed polyhedral cage after validating vertex bounds.
    pub fn new(rest_vertices: Vec<[f32; 3]>, triangles: Vec<[u32; 3]>) -> Result<Self, String> {
        if rest_vertices.len() < 4 {
            return Err("polyhedral cage requires at least 4 vertices".to_string());
        }
        if triangles.len() < 4 {
            return Err("polyhedral cage requires at least 4 triangles".to_string());
        }
        let nv = rest_vertices.len();
        for (ti, tri) in triangles.iter().enumerate() {
            if tri[0] as usize >= nv || tri[1] as usize >= nv || tri[2] as usize >= nv {
                return Err(format!(
                    "triangle {ti} references out-of-bounds vertex index {:?} (vertex count {nv})",
                    tri
                ));
            }
            if tri[0] == tri[1] || tri[1] == tri[2] || tri[2] == tri[0] {
                return Err(format!(
                    "triangle {ti} is degenerate with repeated vertex {:?}",
                    tri
                ));
            }
        }
        Ok(Self {
            rest_vertices,
            triangles,
        })
    }

    /// Creates a closed 8-vertex, 12-triangle axis-aligned box surface cage.
    pub fn create_box_surface_cage(min: [f32; 3], max: [f32; 3]) -> Self {
        let vertices = vec![
            [min[0], min[1], min[2]], // 0: ---
            [max[0], min[1], min[2]], // 1: +--
            [max[0], max[1], min[2]], // 2: ++-
            [min[0], max[1], min[2]], // 3: -+-
            [min[0], min[1], max[2]], // 4: --+
            [max[0], min[1], max[2]], // 5: +-+
            [max[0], max[1], max[2]], // 6: +++
            [min[0], max[1], max[2]], // 7: -++
        ];
        // Outward CCW triangles on every face.
        let triangles = vec![
            [0, 2, 1],
            [0, 3, 2], // -Z
            [4, 5, 6],
            [4, 6, 7], // +Z
            [0, 1, 5],
            [0, 5, 4], // -Y
            [3, 7, 6],
            [3, 6, 2], // +Y
            [0, 4, 7],
            [0, 7, 3], // -X
            [1, 2, 6],
            [1, 6, 5], // +X
        ];
        Self {
            rest_vertices: vertices,
            triangles,
        }
    }

    /// Creates a closed 6-vertex, 8-triangle octahedron cage around `center`.
    pub fn create_octahedron_cage(center: [f32; 3], radius: f32) -> Self {
        let c = Vec3::from(center);
        let r = radius.abs().max(1e-4);
        let vertices = vec![
            (c + Vec3::new(r, 0.0, 0.0)).to_array(),  // 0: +X
            (c + Vec3::new(-r, 0.0, 0.0)).to_array(), // 1: -X
            (c + Vec3::new(0.0, r, 0.0)).to_array(),  // 2: +Y
            (c + Vec3::new(0.0, -r, 0.0)).to_array(), // 3: -Y
            (c + Vec3::new(0.0, 0.0, r)).to_array(),  // 4: +Z
            (c + Vec3::new(0.0, 0.0, -r)).to_array(), // 5: -Z
        ];
        let triangles = vec![
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ];
        Self {
            rest_vertices: vertices,
            triangles,
        }
    }

    /// Creates a closed 12-vertex, 20-triangle regular icosahedron cage around `center`.
    pub fn create_icosahedron_cage(center: [f32; 3], radius: f32) -> Self {
        let c = Vec3::from(center);
        let r = radius.abs().max(1e-4);
        let phi = (1.0 + 5.0_f32.sqrt()) * 0.5;
        let raw = [
            [-1.0, phi, 0.0],
            [1.0, phi, 0.0],
            [-1.0, -phi, 0.0],
            [1.0, -phi, 0.0],
            [0.0, -1.0, phi],
            [0.0, 1.0, phi],
            [0.0, -1.0, -phi],
            [0.0, 1.0, -phi],
            [phi, 0.0, -1.0],
            [phi, 0.0, 1.0],
            [-phi, 0.0, -1.0],
            [-phi, 0.0, 1.0],
        ];
        let vertices = raw
            .iter()
            .map(|p| (c + Vec3::from(*p).normalize() * r).to_array())
            .collect();
        let triangles = vec![
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ];
        Self {
            rest_vertices: vertices,
            triangles,
        }
    }

    /// Parses a Wavefront `.obj` cage mesh (`v` and `f` records, fanning polygon faces).
    pub fn from_obj_str(obj: &str) -> Result<Self, String> {
        let mut vertices = Vec::new();
        let mut triangles = Vec::new();

        for (line_no, raw_line) in obj.lines().enumerate() {
            let line = raw_line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let mut parts = line.split_whitespace();
            let Some(kind) = parts.next() else {
                continue;
            };
            match kind {
                "v" => {
                    let coords: Vec<f32> = parts
                        .take(3)
                        .map(|tok| {
                            tok.parse::<f32>().map_err(|e| {
                                format!(
                                    "line {}: invalid vertex coordinate '{tok}': {e}",
                                    line_no + 1
                                )
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    if coords.len() != 3 {
                        return Err(format!(
                            "line {}: expected 3 coordinates for 'v'",
                            line_no + 1
                        ));
                    }
                    vertices.push([coords[0], coords[1], coords[2]]);
                }
                "f" => {
                    let mut face_indices = Vec::new();
                    for tok in parts {
                        let v_part = tok.split('/').next().unwrap_or("");
                        let signed: isize = v_part.parse().map_err(|e| {
                            format!("line {}: invalid face vertex '{tok}': {e}", line_no + 1)
                        })?;
                        let idx = if signed > 0 {
                            (signed - 1) as usize
                        } else if signed < 0 {
                            let resolved = vertices.len() as isize + signed;
                            if resolved < 0 {
                                return Err(format!(
                                    "line {}: negative face index {signed} out of bounds",
                                    line_no + 1
                                ));
                            }
                            resolved as usize
                        } else {
                            return Err(format!(
                                "line {}: 0 is not a valid OBJ index",
                                line_no + 1
                            ));
                        };
                        face_indices.push(idx as u32);
                    }
                    if face_indices.len() < 3 {
                        return Err(format!(
                            "line {}: face must have at least 3 vertices",
                            line_no + 1
                        ));
                    }
                    for i in 1..face_indices.len() - 1 {
                        triangles.push([face_indices[0], face_indices[i], face_indices[i + 1]]);
                    }
                }
                _ => {}
            }
        }

        Self::new(vertices, triangles)
    }

    /// Number of cage vertices.
    #[inline]
    pub fn num_vertices(&self) -> usize {
        self.rest_vertices.len()
    }

    /// Verifies that every half-edge `(a, b)` is paired with one opposite `(b, a)`.
    pub fn is_closed_manifold(&self) -> bool {
        if self.triangles.is_empty() {
            return false;
        }
        let mut half_edges: HashMap<(u32, u32), u32> = HashMap::new();
        for &[a, b, c] in &self.triangles {
            for edge in [(a, b), (b, c), (c, a)] {
                let count = half_edges.entry(edge).or_insert(0);
                *count += 1;
                if *count > 1 {
                    return false;
                }
            }
        }
        half_edges
            .keys()
            .all(|&(u, v)| half_edges.contains_key(&(v, u)))
    }

    /// Signed enclosed volume via the divergence theorem (positive for outward winding).
    pub fn signed_volume(&self) -> f32 {
        let mut vol6 = 0.0f32;
        for &[a, b, c] in &self.triangles {
            let va = Vec3::from(self.rest_vertices[a as usize]);
            let vb = Vec3::from(self.rest_vertices[b as usize]);
            let vc = Vec3::from(self.rest_vertices[c as usize]);
            vol6 += va.dot(vb.cross(vc));
        }
        vol6 / 6.0
    }

    /// Generalized winding number at `point` (`~1.0` inside outward mesh, `~0.0` outside).
    pub fn winding_number(&self, point: Vec3) -> f32 {
        let mut solid_angle_sum = 0.0f32;
        for &[i0, i1, i2] in &self.triangles {
            let a = Vec3::from(self.rest_vertices[i0 as usize]) - point;
            let b = Vec3::from(self.rest_vertices[i1 as usize]) - point;
            let c = Vec3::from(self.rest_vertices[i2 as usize]) - point;
            let la = a.length();
            let lb = b.length();
            let lc = c.length();
            if la < 1e-7 || lb < 1e-7 || lc < 1e-7 {
                return 1.0;
            }
            let det = a.dot(b.cross(c));
            let denom = la * lb * lc + a.dot(b) * lc + b.dot(c) * la + c.dot(a) * lb;
            solid_angle_sum += 2.0 * det.atan2(denom);
        }
        solid_angle_sum / (4.0 * PI)
    }

    /// Returns true if `point` is enclosed by the cage (`|winding_number| >= 0.5`).
    #[inline]
    pub fn contains_point(&self, point: Vec3) -> bool {
        self.winding_number(point).abs() >= 0.5
    }

    /// Computes normalized 3D Mean Value Coordinates (Ju, Schaefer & Warren 2005) for `point`.
    pub fn compute_mvc_weights(&self, point: [f32; 3]) -> Vec<f32> {
        let nv = self.rest_vertices.len();
        let x = Vec3::from(point);
        let mut dists = Vec::with_capacity(nv);
        let mut dirs = Vec::with_capacity(nv);

        for (idx, v) in self.rest_vertices.iter().enumerate() {
            let diff = Vec3::from(*v) - x;
            let d = diff.length();
            if d < 1e-6 {
                let mut exact = vec![0.0f32; nv];
                exact[idx] = 1.0;
                return exact;
            }
            dists.push(d);
            dirs.push(diff / d);
        }

        let mut weights = vec![0.0f32; nv];
        let mut total_weight = 0.0f32;

        for &[i0, i1, i2] in &self.triangles {
            let i0 = i0 as usize;
            let i1 = i1 as usize;
            let i2 = i2 as usize;

            let u0 = dirs[i0];
            let u1 = dirs[i1];
            let u2 = dirs[i2];

            let l0 = (u1 - u2).length();
            let l1 = (u2 - u0).length();
            let l2 = (u0 - u1).length();

            let theta0 = 2.0 * (l0 * 0.5).clamp(0.0, 1.0).asin();
            let theta1 = 2.0 * (l1 * 0.5).clamp(0.0, 1.0).asin();
            let theta2 = 2.0 * (l2 * 0.5).clamp(0.0, 1.0).asin();
            let h = 0.5 * (theta0 + theta1 + theta2);

            // Point lies on or inside this triangle on the cage boundary.
            if (PI - h).abs() < 1e-5 {
                let w0 = theta0.sin() * dists[i1] * dists[i2];
                let w1 = theta1.sin() * dists[i2] * dists[i0];
                let w2 = theta2.sin() * dists[i0] * dists[i1];
                let sum = w0 + w1 + w2;
                let mut exact = vec![0.0f32; nv];
                if sum.abs() > 1e-12 {
                    exact[i0] = w0 / sum;
                    exact[i1] = w1 / sum;
                    exact[i2] = w2 / sum;
                }
                return exact;
            }

            let sin_t0 = theta0.sin();
            let sin_t1 = theta1.sin();
            let sin_t2 = theta2.sin();
            if sin_t0.abs() < 1e-6 || sin_t1.abs() < 1e-6 || sin_t2.abs() < 1e-6 {
                continue;
            }

            let sin_h = h.sin();
            let c0 =
                ((2.0 * sin_h * (h - theta0).sin()) / (sin_t1 * sin_t2) - 1.0).clamp(-1.0, 1.0);
            let c1 =
                ((2.0 * sin_h * (h - theta1).sin()) / (sin_t2 * sin_t0) - 1.0).clamp(-1.0, 1.0);
            let c2 =
                ((2.0 * sin_h * (h - theta2).sin()) / (sin_t0 * sin_t1) - 1.0).clamp(-1.0, 1.0);

            let det = Mat3::from_cols(u0, u1, u2).determinant();
            let det_sign = if det >= 0.0 { 1.0 } else { -1.0 };

            let s0 = det_sign * (1.0 - c0 * c0).max(0.0).sqrt();
            let s1 = det_sign * (1.0 - c1 * c1).max(0.0).sqrt();
            let s2 = det_sign * (1.0 - c2 * c2).max(0.0).sqrt();
            if s0.abs() <= 1e-6 || s1.abs() <= 1e-6 || s2.abs() <= 1e-6 {
                continue;
            }

            let mu0 = (theta0 - c1 * theta2 - c2 * theta1) / (2.0 * dists[i0] * sin_t1 * s2);
            let mu1 = (theta1 - c2 * theta0 - c0 * theta2) / (2.0 * dists[i1] * sin_t2 * s0);
            let mu2 = (theta2 - c0 * theta1 - c1 * theta0) / (2.0 * dists[i2] * sin_t0 * s1);

            weights[i0] += mu0;
            weights[i1] += mu1;
            weights[i2] += mu2;
            total_weight += mu0 + mu1 + mu2;
        }

        if total_weight.abs() > 1e-12 {
            let inv = 1.0 / total_weight;
            for w in &mut weights {
                *w *= inv;
            }
        } else {
            let nearest = dists
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap_or(0);
            weights[nearest] = 1.0;
        }

        weights
    }

    /// Evaluates the deformed world position at `weights` under `deformed_vertices`.
    #[inline]
    pub fn evaluate_position(weights: &[f32], deformed_vertices: &[[f32; 3]]) -> Vec3 {
        let mut p = Vec3::ZERO;
        for (w, v) in weights.iter().zip(deformed_vertices.iter()) {
            p += Vec3::from(*v) * *w;
        }
        p
    }

    /// Binds an array of Gaussian rest positions to this closed MVC cage in parallel.
    pub fn bind_gaussians(&self, positions: &[[f32; 3]]) -> MvcBindings {
        let nv = self.rest_vertices.len();
        let n = positions.len();
        let rows: Vec<(bool, Vec<f32>)> = parallel::map_slice(positions, |&p| {
            let inside = self.contains_point(Vec3::from(p));
            let row = if inside {
                self.compute_mvc_weights(p)
            } else {
                vec![0.0; nv]
            };
            (inside, row)
        });

        let mut weights = Vec::with_capacity(n * nv);
        let mut bound_mask = Vec::with_capacity(n);
        for (inside, row) in rows {
            bound_mask.push(inside);
            weights.extend_from_slice(&row);
        }

        MvcBindings {
            weights,
            num_cage_vertices: nv,
            bound_mask,
        }
    }

    /// Approximates the smooth MVC deformation gradient $F = \partial \phi / \partial X$ at `point`.
    pub fn local_jacobian(&self, point: [f32; 3], deformed_vertices: &[[f32; 3]]) -> Mat3 {
        let mut bmin = Vec3::splat(f32::INFINITY);
        let mut bmax = Vec3::splat(f32::NEG_INFINITY);
        for &v in &self.rest_vertices {
            let pv = Vec3::from(v);
            bmin = bmin.min(pv);
            bmax = bmax.max(pv);
        }
        let diag = bmax.distance(bmin).max(1e-3);
        let h = (diag * 1e-3).clamp(1e-5, 1e-2);
        let inv_2h = 0.5 / h;

        let sample_col = |axis: usize| -> Vec3 {
            let mut p_plus = point;
            let mut p_minus = point;
            p_plus[axis] += h;
            p_minus[axis] -= h;
            let f_plus =
                Self::evaluate_position(&self.compute_mvc_weights(p_plus), deformed_vertices);
            let f_minus =
                Self::evaluate_position(&self.compute_mvc_weights(p_minus), deformed_vertices);
            (f_plus - f_minus) * inv_2h
        };

        Mat3::from_cols(sample_col(0), sample_col(1), sample_col(2))
    }

    /// Deforms a rest scene using smooth MVC bindings, SH rotation, 3D mip filter, and transient strain splitting.
    pub fn deform_scene_cpu(
        &self,
        scene: &GaussianSceneSoa,
        bindings: &MvcBindings,
        deformed_vertices: &[[f32; 3]],
        sh_degree: u32,
        split_stretch_threshold: f32,
        mip_filter_3d_var: f32,
    ) -> DeformedSceneOutput {
        let count = scene.count;
        let degree = sh_degree.min(3);
        let nv = self.rest_vertices.len();

        let mut positions = vec![0.0f32; count * 3];
        let mut covariances_3d = vec![0.0f32; count * 6];
        let mut opacities = vec![0.0f32; count];
        let mut sh_coeffs = vec![0.0f32; count * 48];
        let mut needs_split = if split_stretch_threshold > 0.0 {
            vec![0u8; count]
        } else {
            Vec::new()
        };

        for i in 0..count {
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
            let rest_sh = &scene.sh_coeffs[i * 48..(i + 1) * 48];

            if !bindings.bound_mask[i] {
                positions[i * 3..i * 3 + 3].copy_from_slice(&rest_pos);
                covariances_3d[i * 6..i * 6 + 6].copy_from_slice(&rest_cov);
                opacities[i] = rest_alpha;
                sh_coeffs[i * 48..(i + 1) * 48].copy_from_slice(rest_sh);
                continue;
            }

            let w = &bindings.weights[i * nv..(i + 1) * nv];
            let deformed_pos = Self::evaluate_position(w, deformed_vertices).to_array();
            let f = self.local_jacobian(rest_pos, deformed_vertices);
            let (rotation, _) = higham_polar_decomposition(f);
            let f_cov = f * covariance_matrix_from_slice(&rest_cov) * f.transpose();
            let raw_cov = [
                f_cov.x_axis.x,
                f_cov.x_axis.y,
                f_cov.x_axis.z,
                f_cov.y_axis.y,
                f_cov.y_axis.z,
                f_cov.z_axis.z,
            ];

            positions[i * 3..i * 3 + 3].copy_from_slice(&deformed_pos);
            covariances_3d[i * 6..i * 6 + 6].copy_from_slice(&raw_cov);
            opacities[i] = rest_alpha;
            sh_coeffs[i * 48..(i + 1) * 48].copy_from_slice(rest_sh);
            if degree > 0 {
                let rotator = ShRotator::new(rotation, degree);
                if let Ok(dst) = <&mut [f32; 48]>::try_from(&mut sh_coeffs[i * 48..(i + 1) * 48]) {
                    rotator.apply(dst);
                }
            }

            if split_stretch_threshold > 0.0 {
                let strain = jacobian_singular_values(f);
                if strain.max_stretch() > split_stretch_threshold {
                    needs_split[i] = 1;
                }
            }
        }

        let mut total_count = count;
        for i in 0..count {
            let split = split_stretch_threshold > 0.0 && needs_split[i] != 0;
            let cov = [
                covariances_3d[i * 6],
                covariances_3d[i * 6 + 1],
                covariances_3d[i * 6 + 2],
                covariances_3d[i * 6 + 3],
                covariances_3d[i * 6 + 4],
                covariances_3d[i * 6 + 5],
            ];
            let alpha = opacities[i];
            if !split {
                if mip_filter_3d_var > 0.0 {
                    let (reg_cov, reg_alpha) = apply_mip_filter_3d(&cov, alpha, mip_filter_3d_var);
                    covariances_3d[i * 6..i * 6 + 6].copy_from_slice(&reg_cov);
                    opacities[i] = reg_alpha;
                }
                continue;
            }

            let center = Vec3::new(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
            let [child_a, child_b] = split_gaussian_along_major_axis(center, &cov, alpha);
            let (cov_a, alpha_a) = if mip_filter_3d_var > 0.0 {
                apply_mip_filter_3d(&child_a.covariance_3d, child_a.opacity, mip_filter_3d_var)
            } else {
                (child_a.covariance_3d, child_a.opacity)
            };
            let (cov_b, alpha_b) = if mip_filter_3d_var > 0.0 {
                apply_mip_filter_3d(&child_b.covariance_3d, child_b.opacity, mip_filter_3d_var)
            } else {
                (child_b.covariance_3d, child_b.opacity)
            };

            let pos_a = center + child_a.offset;
            let pos_b = center + child_b.offset;

            positions[i * 3..i * 3 + 3].copy_from_slice(&pos_a.to_array());
            covariances_3d[i * 6..i * 6 + 6].copy_from_slice(&cov_a);
            opacities[i] = alpha_a;

            positions.extend_from_slice(&pos_b.to_array());
            covariances_3d.extend_from_slice(&cov_b);
            opacities.push(alpha_b);
            let sh_copy = sh_coeffs[i * 48..(i + 1) * 48].to_vec();
            sh_coeffs.extend_from_slice(&sh_copy);
            total_count += 1;
        }

        DeformedSceneOutput {
            scene: GaussianSceneSoa {
                positions,
                covariances_3d,
                opacities,
                sh_coeffs,
                count: total_count,
            },
        }
    }

    /// Converts a star-shaped closed surface cage into a centroid-fanned `TetMesh` for fast GPU deformation.
    pub fn to_tet_mesh(&self) -> TetMesh {
        let mut vertices = self.rest_vertices.clone();
        let inv_n = 1.0 / (vertices.len() as f32);
        let mut centroid = Vec3::ZERO;
        for v in &vertices {
            centroid += Vec3::from(*v);
        }
        centroid *= inv_n;

        let center_idx = vertices.len() as u32;
        vertices.push(centroid.to_array());

        let mut elements = Vec::with_capacity(self.triangles.len());
        for &[i0, i1, i2] in &self.triangles {
            let v0 = Vec3::from(vertices[i0 as usize]);
            let v1 = Vec3::from(vertices[i1 as usize]);
            let v2 = Vec3::from(vertices[i2 as usize]);
            // Ensure positive signed determinant for each tetrahedron.
            let det = Mat3::from_cols(v1 - v0, v2 - v0, centroid - v0).determinant();
            if det >= 0.0 {
                elements.push(TetElement::new(i0, i1, i2, center_idx));
            } else {
                elements.push(TetElement::new(i0, i2, i1, center_idx));
            }
        }

        TetMesh::new(vertices, elements)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mvc_presets_are_closed_manifolds() {
        let box_cage = PolyCageMesh::create_box_surface_cage([-1.0; 3], [1.0; 3]);
        assert!(box_cage.is_closed_manifold());
        assert!((box_cage.signed_volume() - 8.0).abs() < 1e-4);

        let oct = PolyCageMesh::create_octahedron_cage([0.0; 3], 1.5);
        assert!(oct.is_closed_manifold());
        assert!(oct.signed_volume() > 0.0);

        let ico = PolyCageMesh::create_icosahedron_cage([0.0; 3], 2.0);
        assert!(ico.is_closed_manifold());
        assert!(ico.signed_volume() > 0.0);
    }

    #[test]
    fn test_mvc_linear_reproduction_inside_polyhedra() {
        let cages = [
            PolyCageMesh::create_box_surface_cage([-1.0, -2.0, -0.5], [2.0, 1.5, 2.5]),
            PolyCageMesh::create_octahedron_cage([0.3, -0.2, 0.4], 2.0),
            PolyCageMesh::create_icosahedron_cage([-0.5, 0.5, 0.2], 2.5),
        ];

        for cage in &cages {
            let point = [0.45, -0.1, 0.35];
            assert!(cage.contains_point(Vec3::from(point)));
            let w = cage.compute_mvc_weights(point);
            let sum: f32 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "weights sum {sum} != 1");

            let reconstructed = PolyCageMesh::evaluate_position(&w, &cage.rest_vertices);
            let err = reconstructed.distance(Vec3::from(point));
            assert!(err < 1e-4, "MVC reproduction error {err} too large");
        }
    }

    #[test]
    fn test_mvc_vertex_and_face_boundary_reproduction() {
        let oct = PolyCageMesh::create_octahedron_cage([0.0; 3], 1.0);
        let w_v = oct.compute_mvc_weights([1.0, 0.0, 0.0]);
        assert!((w_v[0] - 1.0).abs() < 1e-6);

        // Centroid of first triangle [0, 2, 4] -> (1/3, 1/3, 1/3)
        let on_face = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        let w_f = oct.compute_mvc_weights(on_face);
        let reconstructed = PolyCageMesh::evaluate_position(&w_f, &oct.rest_vertices);
        assert!(reconstructed.distance(Vec3::from(on_face)) < 1e-4);
    }

    #[test]
    fn test_obj_parsing_and_triangulation() {
        let obj = "\
# Simple tetrahedron in OBJ
v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1/1/1 3/2/1 2/3/1
f 1 2 4
f 2 3 4
f 3 1 4
";
        let cage = PolyCageMesh::from_obj_str(obj).unwrap();
        assert_eq!(cage.num_vertices(), 4);
        assert_eq!(cage.triangles.len(), 4);
        assert!(cage.is_closed_manifold());
    }

    #[test]
    fn test_mvc_deform_scene_and_to_tet_mesh() {
        let oct = PolyCageMesh::create_octahedron_cage([0.0; 3], 2.0);
        let tet = oct.to_tet_mesh();
        assert_eq!(tet.num_tets(), oct.triangles.len());
        assert_eq!(tet.num_vertices(), oct.num_vertices() + 1);

        let mut scene = GaussianSceneSoa::with_capacity(2);
        scene.push(
            [0.1, 0.2, 0.0],
            [0.02, 0.0, 0.0, 0.02, 0.0, 0.02],
            0.8,
            &[0.4; 48],
        );
        scene.push(
            [5.0, 0.0, 0.0],
            [0.02, 0.0, 0.0, 0.02, 0.0, 0.02],
            0.8,
            &[0.2; 48],
        );

        let bindings = oct.bind_gaussians(&[[0.1, 0.2, 0.0], [5.0, 0.0, 0.0]]);
        assert_eq!(bindings.bound_mask, vec![true, false]);

        let mut deformed = oct.rest_vertices.clone();
        deformed[0][0] += 3.0; // Heavy X stretch on +X vertex
        deformed[1][0] -= 3.0; // Heavy X stretch on -X vertex

        let out = oct.deform_scene_cpu(&scene, &bindings, &deformed, 0, 1.3, 0.001);
        assert_eq!(out.scene.len(), 3);
        // Outside splat keeps rest position.
        assert!((out.scene.positions[3] - 5.0).abs() < 1e-6);
    }
}
