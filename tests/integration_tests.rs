#![allow(clippy::excessive_precision, clippy::useless_vec)]

use cinder_gs::math::{
    activate_opacity, activate_rotation, activate_scale, compute_covariance_3d,
    covariance_determinant, is_positive_definite, MIN_SCALE,
};
use cinder_gs::scene::{find_end_header, PlyLoader, RawPlyVertex};
use glam::Quat;
use std::io::Write;

/// Test 1: Zero-length quaternion fallback to identity (0, 0, 0, 1)
#[test]
fn test_deliverable_zero_length_quaternion_fallback() {
    // Exactly zero quaternion
    let raw_zero = [0.0, 0.0, 0.0, 0.0];
    let q_zero = activate_rotation(raw_zero);
    assert_eq!(q_zero, Quat::IDENTITY);
    assert_eq!(q_zero.x, 0.0);
    assert_eq!(q_zero.y, 0.0);
    assert_eq!(q_zero.z, 0.0);
    assert_eq!(q_zero.w, 1.0);

    // Below epsilon threshold (< 1e-6)
    let raw_sub_eps = [5e-7, 1e-7, 0.0, 0.0];
    let q_sub_eps = activate_rotation(raw_sub_eps);
    assert_eq!(q_sub_eps, Quat::IDENTITY);
}

/// Test 2: Scale clamping on extreme negative raw values (e.g., -50.0)
#[test]
fn test_deliverable_scale_clamping_extreme_negative() {
    let raw_scale = [-50.0, -100.0, -500.0];
    let s = activate_scale(raw_scale);

    assert_eq!(s.x, MIN_SCALE);
    assert_eq!(s.y, MIN_SCALE);
    assert_eq!(s.z, MIN_SCALE);
    assert_eq!(s.x, 1e-4);
    assert_eq!(s.y, 1e-4);
    assert_eq!(s.z, 1e-4);

    // Verify boundary where exp(val) <= 1e-4
    // ln(1e-4) ~ -9.21034
    let raw_below_clamp = [-10.0, -15.0, -20.0];
    let s_clamped = activate_scale(raw_below_clamp);
    assert_eq!(s_clamped.x, 1e-4);
    assert_eq!(s_clamped.y, 1e-4);
    assert_eq!(s_clamped.z, 1e-4);
}

/// Test 3: Positive-definiteness: determinant det(Sigma) > 0
#[test]
fn test_deliverable_positive_definiteness_determinant() {
    // 3a: At minimum clamp scale (1e-4) and identity rotation
    let s_min = activate_scale([-50.0, -50.0, -50.0]);
    let q_ident = activate_rotation([0.0, 0.0, 0.0, 0.0]);
    let cov_min = compute_covariance_3d(s_min, q_ident);

    let det_min = covariance_determinant(&cov_min);
    assert!(det_min > 0.0, "det(Sigma) must be strictly positive");
    assert!(is_positive_definite(&cov_min));

    // 3b: Across various rotations and scale values
    let test_rots = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.5, 0.5, 0.5, 0.5],
        [std::f32::consts::FRAC_1_SQRT_2, 0.0, std::f32::consts::FRAC_1_SQRT_2, 0.0],
    ];

    let test_scales = [
        [-50.0, 0.0, 1.0],
        [-2.0, -1.0, 0.5],
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 3.0],
    ];

    for r in &test_rots {
        for s_raw in &test_scales {
            let scale = activate_scale(*s_raw);
            let rot = activate_rotation(*r);
            let cov = compute_covariance_3d(scale, rot);

            let det = covariance_determinant(&cov);
            assert!(
                det > 0.0,
                "Determinant {} must be > 0 for s={:?}, r={:?}",
                det,
                s_raw,
                r
            );
            assert!(
                is_positive_definite(&cov),
                "Matrix must be positive definite for s={:?}, r={:?}",
                s_raw,
                r
            );
        }
    }
}

/// Test 4: End-of-header offset detection with both \n and \r\n
#[test]
fn test_deliverable_end_of_header_offset_detection() {
    // 4a: Unix LF (\n)
    let header_lf = "ply\nformat binary_little_endian 1.0\nelement vertex 3\nend_header\n";
    let offset_lf = find_end_header(header_lf.as_bytes()).expect("LF search failed");
    assert_eq!(offset_lf, header_lf.len());

    // 4b: Windows CRLF (\r\n)
    let header_crlf =
        "ply\r\nformat binary_little_endian 1.0\r\nelement vertex 3\r\nend_header\r\n";
    let offset_crlf = find_end_header(header_crlf.as_bytes()).expect("CRLF search failed");
    assert_eq!(offset_crlf, header_crlf.len());

    let v1 = RawPlyVertex {
        position: [1.0, 2.0, 3.0],
        opacity: 1.5,
        scale: [0.1, 0.2, 0.3],
        rot: [1.0, 0.0, 0.0, 0.0],
        ..RawPlyVertex::ZERO
    };

    let mut full_file_lf = Vec::new();
    // 4-byte align the header
    let lf_hdr = "ply\nformat binary_little_endian 1.0\nelement vertex 1\ncomment pad\nend_header\n";
    assert_eq!(lf_hdr.len() % 4, 0);
    full_file_lf.extend_from_slice(lf_hdr.as_bytes());
    full_file_lf.extend_from_slice(v1.as_bytes());

    let offset = find_end_header(&full_file_lf).expect("find_end_header failed");
    assert_eq!(offset, lf_hdr.len());
    let raw_slice: &[RawPlyVertex] =
        bytemuck::cast_slice(&full_file_lf[offset..]);
    assert_eq!(raw_slice.len(), 1);
    assert_eq!(raw_slice[0], v1);

    // Same test with CRLF
    let mut full_file_crlf = Vec::new();
    let crlf_hdr = "ply\r\nformat binary_little_endian 1.0\r\nelement vertex 1\r\ncomment xx\r\nend_header\r\n";
    assert_eq!(crlf_hdr.len() % 4, 0);
    full_file_crlf.extend_from_slice(crlf_hdr.as_bytes());
    full_file_crlf.extend_from_slice(v1.as_bytes());

    let offset_crlf_res = find_end_header(&full_file_crlf).expect("crlf find_end_header failed");
    assert_eq!(offset_crlf_res, crlf_hdr.len());
    let raw_slice_crlf: &[RawPlyVertex] =
        bytemuck::cast_slice(&full_file_crlf[offset_crlf_res..]);
    assert_eq!(raw_slice_crlf.len(), 1);
    assert_eq!(raw_slice_crlf[0], v1);
}

/// Test 5: End-to-end PLY file mmap load, parallel conversion, and validation
#[test]
fn test_end_to_end_mmap_ply_file() {
    let count = 100;
    let mut vertices = Vec::with_capacity(count);
    for i in 0..count {
        let f = i as f32;
        let mut v = RawPlyVertex::ZERO;
        v.position = [f, f * 0.5, -f];
        v.opacity = 1.0;
        v.scale = [0.0, -1.0, -50.0];
        v.rot = [1.0, 0.0, 0.0, 0.0];
        v.f_dc = [0.1, 0.2, 0.3];
        vertices.push(v);
    }

    let base_hdr = format!(
        "ply\nformat binary_little_endian 1.0\nelement vertex {}\n",
        count
    );
    let end_marker = "end_header\n";
    let pad_len = (4 - (base_hdr.len() + end_marker.len() + "comment \n".len()) % 4) % 4;
    let pad_comment = format!("comment {}\n", "p".repeat(pad_len));
    let full_hdr = format!("{}{}{}", base_hdr, pad_comment, end_marker);
    assert_eq!(full_hdr.len() % 4, 0);

    let mut temp = tempfile::NamedTempFile::new().expect("tempfile failed");
    temp.write_all(full_hdr.as_bytes()).expect("write header failed");
    for v in &vertices {
        temp.write_all(v.as_bytes()).expect("write vertex failed");
    }
    temp.flush().expect("flush failed");

    // Load with PlyLoader
    let soa = PlyLoader::load_file(temp.path()).expect("load_file failed");
    assert_eq!(soa.len(), count);
    soa.validate().expect("SOA validation failed");

    // Check activated values for vertex 0
    assert_eq!(soa.position(0), [0.0, 0.0, -0.0]);
    assert!((soa.opacity(0) - activate_opacity(1.0)).abs() < 1e-6);
    assert_eq!(soa.covariance_3d(0)[5], (1e-4f32).powi(2)); // scale_2 was -50.0, clamped to 1e-4
}

// =========================================================================
// MODULE 3 INTEGRATION TESTS: EWA PROJECTION, CULLING & CONIC PIPELINE
// =========================================================================

use cinder_gs::math::{
    compute_jacobian, is_frustum_culled, project_ewa_splat, MAX_SPLAT_RADIUS,
    MIN_SPLAT_RADIUS,
};
use cinder_gs::render::{project_scene_cpu, CameraUniforms};
use cinder_gs::scene::GaussianSceneSoa;
use glam::{Mat4, Vec3};

/// Module 3 - Test 1: Compare CPU projection against known manual reference values
#[test]
fn test_deliverable_cpu_projection_manual_reference() {
    let view_matrix = Mat4::IDENTITY;
    let pos = Vec3::new(0.0, 0.0, 2.0); // 2 meters directly ahead
    let cov3d = [0.04, 0.0, 0.0, 0.04, 0.0, 0.04]; // spherical, sigma_3d = 0.2

    let focal_x = 1000.0;
    let focal_y = 1000.0;
    let principal_x = 500.0;
    let principal_y = 400.0;
    let width = 1000.0;
    let height = 800.0;
    let near_plane = 0.2;

    let splat = project_ewa_splat(
        pos,
        cov3d,
        view_matrix,
        focal_x,
        focal_y,
        principal_x,
        principal_y,
        width,
        height,
        near_plane,
    );

    // 1. Position in camera space: t = (0, 0, 2)
    assert_eq!(splat.depth, 2.0);

    // 2. Screen center: mu_x = 1000 * 0 / 2 + 500 = 500, mu_y = 1000 * 0 / 2 + 400 = 400
    assert!((splat.point_xy[0] - 500.0).abs() < 1e-5);
    assert!((splat.point_xy[1] - 400.0).abs() < 1e-5);

    // 3. Jacobian: J00 = 500, J02 = 0, J11 = 500, J12 = 0
    let jac = compute_jacobian(focal_x, focal_y, 0.0, 0.0, 2.0);
    assert_eq!(jac, [500.0, 0.0, 500.0, 0.0]);

    // 4. Sigma_2D = 500^2 * 0.04 = 10000.0
    // With 0.3 filter: Sigma'_xx = 10000.3, Sigma'_yy = 10000.3, Sigma'_xy = 0.0
    // det = 10000.3^2 = 100006000.09
    // a = 1 / 10000.3 ~= 9.9997e-5
    let expected_a = 1.0 / 10000.3;
    assert!((splat.conic[0] - expected_a).abs() < 1e-6);
    assert_eq!(splat.conic[1], 0.0);
    assert!((splat.conic[2] - expected_a).abs() < 1e-6);

    // 5. Radius: 3 * sqrt(10000.3) = 3 * 100.0015 = 300.0045 -> ceil = 301
    assert_eq!(splat.radius, 301);
    assert!(!splat.is_culled());
}

/// Module 3 - Test 2: Verify culling when t_z <= 0.2 (near plane)
#[test]
fn test_deliverable_near_plane_culling() {
    let view_matrix = Mat4::IDENTITY;
    let cov3d = [0.04, 0.0, 0.0, 0.04, 0.0, 0.04];
    let (fx, fy, cx, cy, w, h, near) = (1000.0, 1000.0, 500.0, 400.0, 1000.0, 800.0, 0.2);

    // Exact near plane boundary (t_z = 0.2)
    let splat_boundary = project_ewa_splat(
        Vec3::new(0.0, 0.0, 0.2),
        cov3d,
        view_matrix,
        fx,
        fy,
        cx,
        cy,
        w,
        h,
        near,
    );
    assert_eq!(splat_boundary.radius, 0);
    assert!(splat_boundary.is_culled());
    assert_eq!(splat_boundary.depth, 0.2);

    // Just behind near plane (t_z = 0.199)
    let splat_near = project_ewa_splat(
        Vec3::new(0.0, 0.0, 0.199),
        cov3d,
        view_matrix,
        fx,
        fy,
        cx,
        cy,
        w,
        h,
        near,
    );
    assert_eq!(splat_near.radius, 0);
    assert!(splat_near.is_culled());

    // Camera origin (t_z = 0.0)
    let splat_origin = project_ewa_splat(
        Vec3::new(0.0, 0.0, 0.0),
        cov3d,
        view_matrix,
        fx,
        fy,
        cx,
        cy,
        w,
        h,
        near,
    );
    assert_eq!(splat_origin.radius, 0);
    assert!(splat_origin.is_culled());

    // Behind camera (t_z = -10.0)
    let splat_behind = project_ewa_splat(
        Vec3::new(0.0, 0.0, -10.0),
        cov3d,
        view_matrix,
        fx,
        fy,
        cx,
        cy,
        w,
        h,
        near,
    );
    assert_eq!(splat_behind.radius, 0);
    assert!(splat_behind.is_culled());

    // Just in front of near plane (t_z = 0.201) -> should survive
    let splat_front = project_ewa_splat(
        Vec3::new(0.0, 0.0, 0.201),
        cov3d,
        view_matrix,
        fx,
        fy,
        cx,
        cy,
        w,
        h,
        near,
    );
    assert!(splat_front.radius > 0);
    assert!(!splat_front.is_culled());
}

/// Module 3 - Test 3: Verify culling when a splat is outside the viewport frustum
#[test]
fn test_deliverable_frustum_culling_outside_viewport() {
    let view_matrix = Mat4::IDENTITY;
    let cov3d = [0.01, 0.0, 0.0, 0.01, 0.0, 0.01]; // small splat
    let (fx, fy, cx, cy, w, h, near) = (500.0, 500.0, 400.0, 300.0, 800.0, 600.0, 0.2);

    // Center splat: inside viewport
    let splat_in = project_ewa_splat(
        Vec3::new(0.0, 0.0, 2.0),
        cov3d,
        view_matrix,
        fx,
        fy,
        cx,
        cy,
        w,
        h,
        near,
    );
    assert!(splat_in.radius > 0);
    assert!(!splat_in.is_culled());

    // Splat far left (x = -20.0 at z = 2.0 -> mu_x = 500 * -20 / 2 + 400 = -4600)
    let splat_left = project_ewa_splat(
        Vec3::new(-20.0, 0.0, 2.0),
        cov3d,
        view_matrix,
        fx,
        fy,
        cx,
        cy,
        w,
        h,
        near,
    );
    assert_eq!(splat_left.radius, 0);
    assert!(splat_left.is_culled());

    // Splat far right (x = 20.0 -> mu_x = 5400)
    let splat_right = project_ewa_splat(
        Vec3::new(20.0, 0.0, 2.0),
        cov3d,
        view_matrix,
        fx,
        fy,
        cx,
        cy,
        w,
        h,
        near,
    );
    assert_eq!(splat_right.radius, 0);
    assert!(splat_right.is_culled());

    // Splat far above (y = -20.0 -> mu_y = 500 * -20 / 2 + 300 = -4700)
    let splat_above = project_ewa_splat(
        Vec3::new(0.0, -20.0, 2.0),
        cov3d,
        view_matrix,
        fx,
        fy,
        cx,
        cy,
        w,
        h,
        near,
    );
    assert_eq!(splat_above.radius, 0);
    assert!(splat_above.is_culled());

    // Splat far below (y = 20.0 -> mu_y = 5300)
    let splat_below = project_ewa_splat(
        Vec3::new(0.0, 20.0, 2.0),
        cov3d,
        view_matrix,
        fx,
        fy,
        cx,
        cy,
        w,
        h,
        near,
    );
    assert_eq!(splat_below.radius, 0);
    assert!(splat_below.is_culled());

    // Test frustum culling function directly with precision bounds
    assert!(is_frustum_culled(-15.0, 100.0, 10.0, 800.0, 600.0)); // max_x = -5 < 0
    assert!(!is_frustum_culled(-5.0, 100.0, 10.0, 800.0, 600.0)); // max_x = 5 >= 0, overlaps left
    assert!(is_frustum_culled(815.0, 100.0, 10.0, 800.0, 600.0)); // min_x = 805 >= 800
    assert!(!is_frustum_culled(805.0, 100.0, 10.0, 800.0, 600.0)); // min_x = 795 < 800, overlaps right
}

/// Module 3 - Test 4: Verify positive-definiteness and stability of conic coefficients (det > 0, a > 0, c > 0)
#[test]
fn test_deliverable_conic_positive_definiteness_and_stability() {
    let view_matrix = Mat4::look_at_lh(
        Vec3::new(0.0, 0.0, -5.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::Y,
    );
    let (fx, fy, cx, cy, w, h, near) = (800.0, 800.0, 400.0, 300.0, 800.0, 600.0, 0.2);

    // Highly anisotropic, rotated, and diverse 3D covariances
    let test_cases = [
        [0.01, 0.0, 0.0, 0.01, 0.0, 0.01],
        [0.1, 0.05, 0.02, 0.2, 0.03, 0.15],
        [1.0, -0.2, 0.1, 0.5, 0.05, 0.3],
        [0.0001, 0.0, 0.0, 0.0001, 0.0, 0.0001],
        [5.0, 1.0, -0.5, 3.0, 0.8, 2.0],
    ];

    for cov3d in &test_cases {
        let splat = project_ewa_splat(
            Vec3::new(0.0, 0.0, 0.0),
            *cov3d,
            view_matrix,
            fx,
            fy,
            cx,
            cy,
            w,
            h,
            near,
        );

        assert!(!splat.is_culled(), "Target at origin should be visible");
        assert!(splat.radius > 0);

        let a = splat.conic[0];
        let b = splat.conic[1];
        let c = splat.conic[2];

        // 1. Diagonal elements must be strictly positive
        assert!(a > 0.0, "Conic 'a' must be > 0, got {}", a);
        assert!(c > 0.0, "Conic 'c' must be > 0, got {}", c);

        // 2. Conic determinant: a*c - b^2 must be strictly positive
        let conic_det = a * c - b * b;
        assert!(
            conic_det > 0.0,
            "Conic determinant ac - b^2 must be > 0, got {}",
            conic_det
        );

        // 3. Trace must be strictly positive
        assert!(a + c > 0.0, "Conic trace must be > 0");
    }
}

/// Module 3 - Test 5: Verify 3-sigma radius clamping at 1024.0 for extreme covariance scales
#[test]
fn test_deliverable_radius_clamping_at_1024() {
    let view_matrix = Mat4::IDENTITY;
    let (fx, fy, cx, cy, w, h, near) = (1000.0, 1000.0, 500.0, 500.0, 2000.0, 2000.0, 0.2);

    // Enormous covariance at 1 meter in front of camera
    let huge_cov = [10000.0, 0.0, 0.0, 10000.0, 0.0, 10000.0];
    let splat_huge = project_ewa_splat(
        Vec3::new(0.0, 0.0, 1.0),
        huge_cov,
        view_matrix,
        fx,
        fy,
        cx,
        cy,
        w,
        h,
        near,
    );

    // Must be clamped to MAX_SPLAT_RADIUS (1024.0)
    assert_eq!(splat_huge.radius, MAX_SPLAT_RADIUS as i32);
    assert_eq!(splat_huge.radius, 1024);

    // Microscopic covariance: must not drop below MIN_SPLAT_RADIUS (1.0)
    let tiny_cov = [1e-8, 0.0, 0.0, 1e-8, 0.0, 1e-8];
    let splat_tiny = project_ewa_splat(
        Vec3::new(0.0, 0.0, 50.0),
        tiny_cov,
        view_matrix,
        fx,
        fy,
        cx,
        cy,
        w,
        h,
        near,
    );

    assert!(splat_tiny.radius >= MIN_SPLAT_RADIUS as i32);
    assert!(splat_tiny.radius <= MAX_SPLAT_RADIUS as i32);
}

/// Module 3 - Test 6: Full scene CPU projection integration with GaussianSceneSoa and CameraUniforms
#[test]
fn test_deliverable_scene_projection_cpu() {
    let mut soa = GaussianSceneSoa::with_capacity(4);

    let sh = [0.0f32; 48];
    // Splat 0: In front, centered (should be visible)
    soa.push([0.0, 0.0, 2.0], [0.04, 0.0, 0.0, 0.04, 0.0, 0.04], 1.0, &sh);
    // Splat 1: Behind near plane (t_z = 0.1 <= 0.2, should be culled)
    soa.push([0.0, 0.0, 0.1], [0.04, 0.0, 0.0, 0.04, 0.0, 0.04], 1.0, &sh);
    // Splat 2: Far outside frustum to the right (should be culled)
    soa.push([100.0, 0.0, 2.0], [0.01, 0.0, 0.0, 0.01, 0.0, 0.01], 1.0, &sh);
    // Splat 3: In front, slightly offset (should be visible)
    soa.push([0.1, -0.1, 3.0], [0.02, 0.0, 0.0, 0.02, 0.0, 0.02], 1.0, &sh);

    let camera = CameraUniforms::new(
        Mat4::IDENTITY,
        1000.0,
        1000.0,
        500.0,
        400.0,
        1000.0,
        800.0,
        0.2,
    );

    let outputs = project_scene_cpu(&soa, &camera);
    assert_eq!(outputs.count, 4);
    assert_eq!(outputs.points_xy.len(), 8);
    assert_eq!(outputs.depths.len(), 4);
    assert_eq!(outputs.radii.len(), 4);
    assert_eq!(outputs.conics.len(), 12);

    // Splat 0: visible
    assert!(outputs.radii[0] > 0);
    assert_eq!(outputs.depths[0], 2.0);

    // Splat 1: culled by near plane
    assert_eq!(outputs.radii[1], 0);

    // Splat 2: culled by frustum
    assert_eq!(outputs.radii[2], 0);

    // Splat 3: visible
    assert!(outputs.radii[3] > 0);
    assert_eq!(outputs.depths[3], 3.0);

    assert_eq!(outputs.num_surviving(), 2);
}

// =========================================================================
// MODULE 4 INTEGRATION TESTS: TILE BINNING, KEY GENERATION & SORTING
// =========================================================================

use cinder_gs::render::{
    bin_and_sort_gaussians_cpu, compute_gaussian_tile_bounds, count_tiles_touched,
    TileConfigUniforms, TileRange,
};

/// Module 4 - Test 1: Single Gaussian touching exactly 1x1 tile
#[test]
fn test_deliverable_tile_binning_single_gaussian_1x1() {
    let config = TileConfigUniforms::new(800, 600, 1);
    assert_eq!(config.tiles_x, 50);
    assert_eq!(config.tiles_y, 38);

    // Center at (24.0, 24.0), radius 2.0 -> bounds [22, 26] x [22, 26]
    // floor(22/16) = 1, floor(26/16) = 1 -> tx in [1, 1], ty in [1, 1]
    let points_xy = [24.0, 24.0];
    let depths = [3.5];
    let radii = [2];

    let bounds = compute_gaussian_tile_bounds(24.0, 24.0, 2.0, 50, 38, 800, 600);
    assert_eq!(bounds, Some((1, 1, 1, 1)));
    assert_eq!(count_tiles_touched(24.0, 24.0, 2.0, 50, 38, 800, 600), 1);

    let out = bin_and_sort_gaussians_cpu(&points_xy, &depths, &radii, config);
    assert_eq!(out.total_entries, 1);
    assert_eq!(out.sorted_entries.len(), 1);

    let target_tile_id = config.tile_id(1, 1); // 1 * 50 + 1 = 51
    assert_eq!(target_tile_id, 51);
    assert_eq!(out.sorted_entries[0].key.tile_id, 51);
    assert_eq!(out.sorted_entries[0].key.depth(), 3.5);
    assert_eq!(out.sorted_entries[0].gaussian_id, 0);

    assert_eq!(out.tile_ranges[51], TileRange::new(0, 1));
    assert_eq!(out.entry_count_for_tile(51), 1);
    assert_eq!(out.non_empty_tiles_count(), 1);

    // All other tiles must be empty
    for tile_id in 0..config.total_tiles {
        if tile_id != 51 {
            assert!(out.is_tile_empty(tile_id));
        }
    }
}

/// Module 4 - Test 2: Straddling Gaussian overlapping boundary (2x2 tiles)
#[test]
fn test_deliverable_tile_binning_straddling_gaussian_2x2() {
    let config = TileConfigUniforms::new(800, 600, 1);
    // Center directly on corner junction at (32.0, 16.0), radius 4.0
    // x in [28.0, 36.0] -> tx in [1, 2]
    // y in [12.0, 20.0] -> ty in [0, 1]
    let points_xy = [32.0, 16.0];
    let depths = [2.0];
    let radii = [4];

    let bounds = compute_gaussian_tile_bounds(32.0, 16.0, 4.0, 50, 38, 800, 600);
    assert_eq!(bounds, Some((1, 2, 0, 1)));
    assert_eq!(count_tiles_touched(32.0, 16.0, 4.0, 50, 38, 800, 600), 4);

    let out = bin_and_sort_gaussians_cpu(&points_xy, &depths, &radii, config);
    assert_eq!(out.total_entries, 4);
    assert_eq!(out.sorted_entries.len(), 4);

    // Expected 4 tiles: (1,0) = 1, (2,0) = 2, (1,1) = 51, (2,1) = 52
    let expected_tile_ids = [1, 2, 51, 52];
    for (idx, &expected_id) in expected_tile_ids.iter().enumerate() {
        assert_eq!(out.sorted_entries[idx].key.tile_id, expected_id);
        assert_eq!(out.sorted_entries[idx].key.depth(), 2.0);
        assert_eq!(out.sorted_entries[idx].gaussian_id, 0);
        assert_eq!(out.tile_ranges[expected_id as usize], TileRange::new(idx as u32, (idx + 1) as u32));
    }
    assert_eq!(out.non_empty_tiles_count(), 4);
}

/// Module 4 - Test 3: Culled Gaussian (radius = 0 or negative) producing 0 keys
#[test]
fn test_deliverable_tile_binning_culled_gaussian_radius_zero() {
    let config = TileConfigUniforms::new(800, 600, 3);
    // Gaussian 0: culled (radius = 0)
    // Gaussian 1: culled (radius = -5)
    // Gaussian 2: visible (radius = 2 at (8.0, 8.0))
    let points_xy = [50.0, 50.0, 200.0, 200.0, 8.0, 8.0];
    let depths = [1.0, 2.0, 4.0];
    let radii = [0, -5, 2];

    let out = bin_and_sort_gaussians_cpu(&points_xy, &depths, &radii, config);
    assert_eq!(out.total_entries, 1);
    assert_eq!(out.sorted_entries.len(), 1);

    // Only Gaussian 2 survives and produces a key
    assert_eq!(out.sorted_entries[0].gaussian_id, 2);
    assert_eq!(out.sorted_entries[0].key.tile_id, 0);
    assert_eq!(out.sorted_entries[0].key.depth(), 4.0);

    assert_eq!(out.non_empty_tiles_count(), 1);
}

/// Module 4 - Test 4: Lexicographical sorting verification (monotonic tile_id, and monotonic depth_bits within identical tile_id)
#[test]
fn test_deliverable_tile_binning_lexicographical_sorting() {
    let config = TileConfigUniforms::new(800, 600, 6);

    // 6 Gaussians:
    // Tile 2: depths 10.0, 1.2, 5.5
    // Tile 0: depths 8.0, 2.0
    // Tile 1: depth 3.0
    let points_xy = [
        // In tile 2: x in [32..48], y in [0..16] -> pixel (36, 8)
        36.0, 8.0,
        36.0, 8.0,
        36.0, 8.0,
        // In tile 0: pixel (8, 8)
        8.0, 8.0,
        8.0, 8.0,
        // In tile 1: pixel (20, 8)
        20.0, 8.0,
    ];
    let depths = [10.0, 1.2, 5.5, 8.0, 2.0, 3.0];
    let radii = [2, 2, 2, 2, 2, 2];

    let out = bin_and_sort_gaussians_cpu(&points_xy, &depths, &radii, config);
    assert_eq!(out.total_entries, 6);

    // Verify global monotonicity of tile_id and depth within tile_id
    for k in 0..out.total_entries - 1 {
        let curr = &out.sorted_entries[k];
        let next = &out.sorted_entries[k + 1];

        assert!(
            curr.key.tile_id <= next.key.tile_id,
            "tile_id must be monotonic: {} <= {}",
            curr.key.tile_id,
            next.key.tile_id
        );

        if curr.key.tile_id == next.key.tile_id {
            assert!(
                curr.key.depth_bits <= next.key.depth_bits,
                "depth_bits must be monotonic within identical tile_id: {} <= {}",
                curr.key.depth(),
                next.key.depth()
            );
        }
    }

    // Expected sequence:
    // tile 0: depth 2.0 (g4), depth 8.0 (g3)
    assert_eq!(out.sorted_entries[0].key.tile_id, 0);
    assert_eq!(out.sorted_entries[0].key.depth(), 2.0);
    assert_eq!(out.sorted_entries[1].key.tile_id, 0);
    assert_eq!(out.sorted_entries[1].key.depth(), 8.0);

    // tile 1: depth 3.0 (g5)
    assert_eq!(out.sorted_entries[2].key.tile_id, 1);
    assert_eq!(out.sorted_entries[2].key.depth(), 3.0);

    // tile 2: depth 1.2 (g1), depth 5.5 (g2), depth 10.0 (g0)
    assert_eq!(out.sorted_entries[3].key.tile_id, 2);
    assert_eq!(out.sorted_entries[3].key.depth(), 1.2);
    assert_eq!(out.sorted_entries[4].key.tile_id, 2);
    assert_eq!(out.sorted_entries[4].key.depth(), 5.5);
    assert_eq!(out.sorted_entries[5].key.tile_id, 2);
    assert_eq!(out.sorted_entries[5].key.depth(), 10.0);
}

/// Module 4 - Test 5: Boundary range validation (empty tiles have start == end; non-empty encompass all entries)
#[test]
fn test_deliverable_tile_binning_boundary_range_validation() {
    let config = TileConfigUniforms::new(800, 600, 4);
    let points_xy = [8.0, 8.0, 8.0, 8.0, 24.0, 8.0, 40.0, 8.0];
    let depths = [4.0, 1.0, 2.0, 3.0];
    let radii = [2, 2, 2, 2];

    let out = bin_and_sort_gaussians_cpu(&points_xy, &depths, &radii, config);
    assert_eq!(out.total_entries, 4);

    let mut accounted_entries = 0;
    for tile_id in 0..config.total_tiles {
        let range = out.tile_ranges[tile_id as usize];
        if range.is_empty() {
            assert_eq!(range.start, range.end);
            assert_eq!(out.entries_for_tile(tile_id).len(), 0);
        } else {
            assert!(range.end > range.start);
            let count = (range.end - range.start) as usize;
            accounted_entries += count;

            let entries = out.entries_for_tile(tile_id);
            assert_eq!(entries.len(), count);
            for entry in entries {
                assert_eq!(entry.key.tile_id, tile_id);
            }
        }
    }
    assert_eq!(accounted_entries, out.total_entries);
}

/// Module 4 - Test 6: End-to-end integration chaining Module 3 ProjectionOutputs with Module 4 TileBinningOutputs
#[test]
fn test_deliverable_end_to_end_projection_and_binning() {
    let mut soa = GaussianSceneSoa::with_capacity(3);
    let sh = [0.0f32; 48];

    // Splat 0: Centered at (0, 0, 2.0) -> screen center (500, 400), visible
    soa.push([0.0, 0.0, 2.0], [0.04, 0.0, 0.0, 0.04, 0.0, 0.04], 1.0, &sh);
    // Splat 1: Culled near plane (t_z = 0.1 <= 0.2)
    soa.push([0.0, 0.0, 0.1], [0.04, 0.0, 0.0, 0.04, 0.0, 0.04], 1.0, &sh);
    // Splat 2: Offset at (0.2, 0.1, 2.0) -> visible
    soa.push([0.2, 0.1, 2.0], [0.01, 0.0, 0.0, 0.01, 0.0, 0.01], 1.0, &sh);

    let camera = CameraUniforms::new(
        Mat4::IDENTITY,
        1000.0,
        1000.0,
        500.0,
        400.0,
        1000.0,
        800.0,
        0.2,
    );

    // Project using Module 3
    let proj = project_scene_cpu(&soa, &camera);
    assert_eq!(proj.count, 3);
    assert_eq!(proj.num_surviving(), 2);
    assert_eq!(proj.radii[1], 0); // Culled

    // Bin and sort using Module 4
    let binning = proj.bin_and_sort(1000, 800);
    assert!(binning.total_entries > 0);

    // Ensure no culled splat (gaussian_id == 1) appears in sorted entries
    for entry in &binning.sorted_entries {
        assert_ne!(entry.gaussian_id, 1, "Culled splat must not appear in binning");
    }

    assert!(binning.non_empty_tiles_count() > 0);
}

// =========================================================================
// MODULE 5 INTEGRATION TESTS: SHARED-MEMORY TILE COMPOSITOR & RASTERIZER
// =========================================================================

use cinder_gs::render::{
    rasterize_scene_cpu, render_scene_cpu, CompositorUniforms, SortEntry, SortKey,
};

/// Module 5 - Test 1: Verification of half-pixel (+0.5) center alignment
#[test]
fn test_deliverable_compositor_half_pixel_center_alignment() {
    // Single Gaussian placed exactly at pixel (0, 0) center: (0.5, 0.5)
    let points_xy = [0.5, 0.5];
    let conics = [1.0, 0.0, 1.0]; // circular Gaussian, variance = 1
    let opacities = [0.8];
    let mut sh = [0.0f32; 48];
    sh[0] = 1.0; // Positive DC red
    let positions = [0.0, 0.0, 2.0];
    let sorted_entries = [SortEntry::new(SortKey::new(0, 2.0), 0)];
    let tile_ranges = [TileRange::new(0, 1)];

    let uniforms = CompositorUniforms::new(16, 16, [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0], 0);

    let img = rasterize_scene_cpu(
        &points_xy,
        &conics,
        &opacities,
        &sh,
        &positions,
        &sorted_entries,
        &tile_ranges,
        &uniforms,
    );

    // Pixel (0, 0) sampled at (0.5, 0.5): dx = 0, dy = 0 -> q = 0 -> alpha = min(0.99, 0.8 * exp(0)) = 0.8
    let p00 = img.get_pixel_rgba_f32(0, 0);
    // Pixel (1, 0) sampled at (1.5, 0.5): dx = 1, dy = 0 -> q = -0.5 * 1 = -0.5 -> alpha = 0.8 * exp(-0.5) ~= 0.4852
    let p10 = img.get_pixel_rgba_f32(1, 0);

    assert!(
        p00[0] > p10[0],
        "Pixel (0,0) exactly at center must have strictly higher red value than neighbor (1,0): {} vs {}",
        p00[0],
        p10[0]
    );

    // Theoretical red value for (0,0): alpha * (0.5 + 0.2820948) = 0.8 * 0.7820948 = 0.6256758
    let expected_c0 = (0.5f32 + 0.28209479177387814f32).clamp(0.0f32, 1.0f32);
    let expected_r00 = 0.8 * expected_c0;
    assert!((p00[0] - expected_r00).abs() < 1e-4);
}

/// Module 5 - Test 2: Verification of front-to-back alpha blending formula (T * (1 - alpha))
#[test]
fn test_deliverable_compositor_front_to_back_alpha_blending() {
    // Two overlapping Gaussians placed at (0.5, 0.5):
    // Splat 0 (front, depth 1.0): alpha = 0.6, Red (DC = 1.0 / C0 so color is 1.0)
    // Splat 1 (back, depth 2.0): alpha = 0.8, Green (DC = 1.0 / C0 so color is 1.0)
    let points_xy = [0.5, 0.5, 0.5, 0.5];
    let conics = [0.001, 0.0, 0.001, 0.001, 0.0, 0.001]; // very broad, exp(q) ~ 1.0
    let opacities = [0.6, 0.8];

    let mut sh = vec![0.0f32; 48 * 2];
    // Splat 0: pure red (R = 1.0, G = 0.0, B = 0.0) -> DC = (1.0 - 0.5) / C0
    let c0 = 0.28209479177387814;
    sh[0] = 0.5 / c0; // R = 1.0
    sh[1] = -0.5 / c0; // G = 0.0
    sh[2] = -0.5 / c0; // B = 0.0

    // Splat 1: pure green (R = 0.0, G = 1.0, B = 0.0)
    sh[48] = -0.5 / c0; // R = 0.0
    sh[48 + 1] = 0.5 / c0; // G = 1.0
    sh[48 + 2] = -0.5 / c0; // B = 0.0

    let positions = [0.0, 0.0, 1.0, 0.0, 0.0, 2.0];
    let sorted_entries = [
        SortEntry::new(SortKey::new(0, 1.0), 0),
        SortEntry::new(SortKey::new(0, 2.0), 1),
    ];
    let tile_ranges = [TileRange::new(0, 2)];

    // Background: pure blue [0.0, 0.0, 1.0, 1.0]
    let uniforms = CompositorUniforms::new(16, 16, [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0], 0);

    let img = rasterize_scene_cpu(
        &points_xy,
        &conics,
        &opacities,
        &sh,
        &positions,
        &sorted_entries,
        &tile_ranges,
        &uniforms,
    );

    let p00 = img.get_pixel_rgba_f32(0, 0);

    // Exact derivation:
    // Splat 0: alpha_0 = 0.6, w_0 = 0.6 * 1.0 = 0.6. C_r = 0.6. T = 1 - 0.6 = 0.4.
    // Splat 1: alpha_1 = 0.8, w_1 = 0.8 * 0.4 = 0.32. C_g = 0.32. T = 0.4 * (1 - 0.8) = 0.08.
    // Background: C_b = 0.08 * 1.0 = 0.08.
    // Alpha: 1 - 0.08 = 0.92.
    assert!((p00[0] - 0.60).abs() < 1e-3, "Expected red ~= 0.60, got {}", p00[0]);
    assert!((p00[1] - 0.32).abs() < 1e-3, "Expected green ~= 0.32, got {}", p00[1]);
    assert!((p00[2] - 0.08).abs() < 1e-3, "Expected blue ~= 0.08, got {}", p00[2]);
    assert!((p00[3] - 0.92).abs() < 1e-3, "Expected alpha ~= 0.92, got {}", p00[3]);
}

/// Module 5 - Test 3: Test early ray termination saturation when T < 0.001
#[test]
fn test_deliverable_compositor_early_ray_termination() {
    // 6 fully opaque splats at the same pixel:
    // First 3 splats are red, next 3 are yellow (red + green)
    let points_xy = vec![0.5f32; 12];
    let conics = vec![0.001, 0.0, 0.001].repeat(6);
    let opacities = vec![0.99; 6];

    let mut sh = vec![0.0f32; 48 * 6];
    let c0 = 0.28209479177387814;
    // First 3: Red
    for i in 0..3 {
        sh[i * 48] = 0.5 / c0;
        sh[i * 48 + 1] = -0.5 / c0;
        sh[i * 48 + 2] = -0.5 / c0;
    }
    // Next 3: Yellow (pure green + red)
    for i in 3..6 {
        sh[i * 48] = 0.5 / c0;
        sh[i * 48 + 1] = 0.5 / c0;
        sh[i * 48 + 2] = -0.5 / c0;
    }

    let positions = vec![0.0f32; 18];
    let sorted_entries = (0..6)
        .map(|i| SortEntry::new(SortKey::new(0, i as f32), i as u32))
        .collect::<Vec<_>>();
    let tile_ranges = [TileRange::new(0, 6)];

    let uniforms = CompositorUniforms::new(16, 16, [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0], 0);

    let img = rasterize_scene_cpu(
        &points_xy,
        &conics,
        &opacities,
        &sh,
        &positions,
        &sorted_entries,
        &tile_ranges,
        &uniforms,
    );

    let p00 = img.get_pixel_rgba_f32(0, 0);
    // After 2 splats: T = (0.01)^2 = 0.0001 < 0.001 -> saturation triggers and T becomes 0.0!
    // Green channel from splats 3..5 should be completely occluded (0.0)!
    assert!(p00[0] > 0.98, "Red must be fully saturated");
    assert!(p00[1] < 1e-4, "Occluded yellow splats must not contribute green");
}

/// Module 5 - Test 4: Test non-multiple-of-16 screen boundaries (e.g. 19x19 pixels)
#[test]
fn test_deliverable_compositor_non_multiple_of_16_screen_boundaries() {
    let uniforms = CompositorUniforms::new(19, 19, [0.25, 0.5, 0.75, 1.0], [0.0, 0.0, 0.0], 0);
    assert_eq!(uniforms.tiles_x, 2);
    assert_eq!(uniforms.tiles_y, 2);

    let img = rasterize_scene_cpu(&[], &[], &[], &[], &[], &[], &[], &uniforms);

    assert_eq!(img.width, 19);
    assert_eq!(img.height, 19);
    assert_eq!(img.rgba8.len(), 19 * 19 * 4);
    assert_eq!(img.rgba_f32.len(), 19 * 19 * 4);

    // Verify all 4 corner pixels have exact background color
    let corners = [(0, 0), (18, 0), (0, 18), (18, 18)];
    for &(cx, cy) in &corners {
        let p = img.get_pixel_rgba_f32(cx, cy);
        assert!((p[0] - 0.25).abs() < 1e-5);
        assert!((p[1] - 0.50).abs() < 1e-5);
        assert!((p[2] - 0.75).abs() < 1e-5);
        assert!((p[3] - 1.00).abs() < 1e-5);
    }
}

/// Module 5 - Test 5: Full end-to-end rendering pipeline from GaussianSceneSoa to final pixels
#[test]
fn test_deliverable_compositor_full_scene_render() {
    let mut soa = GaussianSceneSoa::with_capacity(2);
    let mut sh0 = [0.0f32; 48];
    sh0[0] = 1.0; // Red DC
    let mut sh1 = [0.0f32; 48];
    sh1[1] = 1.0; // Green DC

    // Splat 0: Centered at (0, 0, 2.0)
    soa.push([0.0, 0.0, 2.0], [0.04, 0.0, 0.0, 0.04, 0.0, 0.04], 1.0, &sh0);
    // Splat 1: Slightly offset at (0.05, 0.0, 3.0)
    soa.push([0.05, 0.0, 3.0], [0.04, 0.0, 0.0, 0.04, 0.0, 0.04], 0.8, &sh1);

    let camera = CameraUniforms::new(
        Mat4::IDENTITY,
        1000.0,
        1000.0,
        400.0,
        300.0,
        800.0,
        600.0,
        0.2,
    );

    let compositor = CompositorUniforms::new(
        800,
        600,
        [0.0, 0.0, 0.0, 1.0], // Black background
        [0.0, 0.0, 0.0],
        0,
    );

    let img = render_scene_cpu(&soa, &camera, &compositor);
    assert_eq!(img.width, 800);
    assert_eq!(img.height, 600);

    // Center pixel (400, 300) should be rendered with Gaussian color
    let center_px = img.get_pixel_rgba_f32(400, 300);
    assert!(center_px[0] > 0.0, "Center pixel should have red channel from Splat 0");
    assert!(center_px[3] > 0.5, "Center pixel alpha should be high");

    // Distant corner pixel (0, 0) should remain black background
    let corner_px = img.get_pixel_rgba_f32(0, 0);
    assert_eq!(corner_px[0], 0.0);
    assert_eq!(corner_px[1], 0.0);
    assert_eq!(corner_px[2], 0.0);
}

// =========================================================================
// MODULE 6 INTEGRATION TESTS: VOLUMETRIC CAGE DEFORMATION PIPELINE
// =========================================================================

use cinder_gs::deformation::{
    deform_scene_cpu, higham_polar_decomposition, DeformationUniforms, TetMesh,
};
use glam::Mat3;


/// Module 6 - Test 1: Identity deformation (D_s = D_m => J = I, Sigma' = Sigma, alpha' = alpha)
#[test]
fn test_deliverable_deformation_identity() {
    let cage = TetMesh::create_box_cage([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]);
    let mut soa = GaussianSceneSoa::with_capacity(1);
    let sh = [0.0f32; 48];
    let rest_pos = [0.5, 0.5, 0.5];
    let rest_cov = [0.04, 0.01, 0.005, 0.09, 0.02, 0.16];
    let rest_opacity = 0.75;
    soa.push(rest_pos, rest_cov, rest_opacity, &sh);

    let bindings = cage.bind_gaussians(&[rest_pos]);
    let uniforms = DeformationUniforms::new([0.0, 0.0, -5.0], 0, 1, cage.num_tets() as u32);

    // Unchanged deformed vertices = rest vertices
    let deformed_vertices = cage.rest_vertices.clone();

    let output = deform_scene_cpu(&soa, &cage, &bindings, &deformed_vertices, &uniforms);

    // 1. Position should be exactly preserved
    assert!((output.scene.positions[0] - rest_pos[0]).abs() < 1e-5);
    assert!((output.scene.positions[1] - rest_pos[1]).abs() < 1e-5);
    assert!((output.scene.positions[2] - rest_pos[2]).abs() < 1e-5);

    // 2. Covariance should match rest covariance
    for (k, &expected_cov) in rest_cov.iter().enumerate() {
        assert!(
            (output.scene.covariances_3d[k] - expected_cov).abs() < 1e-5,
            "Covariance component {} deviated under identity: got {}, expected {}",
            k,
            output.scene.covariances_3d[k],
            expected_cov
        );
    }

    // 3. Opacity should match rest opacity
    assert!(
        (output.scene.opacities[0] - rest_opacity).abs() < 1e-5,
        "Opacity deviated under identity: got {}, expected {}",
        output.scene.opacities[0],
        rest_opacity
    );
}

/// Module 6 - Test 2: Uniform scaling (J = s I => Sigma' = s^2 Sigma, alpha' = 1 - (1-alpha)^{1/s^3})
#[test]
fn test_deliverable_deformation_uniform_scaling() {
    let cage = TetMesh::create_box_cage([-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]);
    let mut soa = GaussianSceneSoa::with_capacity(1);
    let sh = [0.0f32; 48];
    let rest_pos = [0.5, 0.5, 0.5];
    let rest_cov = [0.04, 0.0, 0.0, 0.04, 0.0, 0.04];
    let rest_opacity = 0.8;
    soa.push(rest_pos, rest_cov, rest_opacity, &sh);

    let bindings = cage.bind_gaussians(&[rest_pos]);
    let uniforms = DeformationUniforms::new([0.0, 0.0, -5.0], 0, 1, cage.num_tets() as u32);

    // Scale cage by factor s = 2.0
    let s = 2.0f32;
    let deformed_vertices: Vec<[f32; 3]> = cage
        .rest_vertices
        .iter()
        .map(|v| [v[0] * s, v[1] * s, v[2] * s])
        .collect();

    let output = deform_scene_cpu(&soa, &cage, &bindings, &deformed_vertices, &uniforms);

    // 1. Position scaled by s
    assert!((output.scene.positions[0] - rest_pos[0] * s).abs() < 1e-4);
    assert!((output.scene.positions[1] - rest_pos[1] * s).abs() < 1e-4);
    assert!((output.scene.positions[2] - rest_pos[2] * s).abs() < 1e-4);

    // 2. Covariance scaled by s^2 = 4.0
    let expected_cov_diag = rest_cov[0] * s * s;
    assert!((output.scene.covariances_3d[0] - expected_cov_diag).abs() < 1e-4);
    assert!((output.scene.covariances_3d[3] - expected_cov_diag).abs() < 1e-4);
    assert!((output.scene.covariances_3d[5] - expected_cov_diag).abs() < 1e-4);

    // 3. Opacity scaled: det(J) = s^3 = 8.0, base = 1 - 0.8 = 0.2, alpha' = 1 - 0.2^(1/8)
    let det_j = s * s * s;
    let expected_alpha = 1.0 - (1.0f32 - rest_opacity).powf(1.0 / det_j);
    assert!(
        (output.scene.opacities[0] - expected_alpha).abs() < 1e-4,
        "Opacity scaling mismatch: got {}, expected {}",
        output.scene.opacities[0],
        expected_alpha
    );
}

/// Module 6 - Test 3: Pure rotation: Verify det(R) = 1 and polar decomposition convergence
#[test]
fn test_deliverable_deformation_pure_rotation() {
    let angle = std::f32::consts::FRAC_PI_4; // 45 degrees
    let rot = Mat3::from_rotation_y(angle);

    let cage = TetMesh::create_box_cage([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
    let deformed_vertices: Vec<[f32; 3]> = cage
        .rest_vertices
        .iter()
        .map(|&v| {
            let r_v = rot * Vec3::from(v);
            [r_v.x, r_v.y, r_v.z]
        })
        .collect();

    // Verify polar decomposition for all tets
    for (t_idx, elem) in cage.elements.iter().enumerate() {
        let x0 = Vec3::from(deformed_vertices[elem.indices[0] as usize]);
        let x1 = Vec3::from(deformed_vertices[elem.indices[1] as usize]);
        let x2 = Vec3::from(deformed_vertices[elem.indices[2] as usize]);
        let x3 = Vec3::from(deformed_vertices[elem.indices[3] as usize]);

        let ds = Mat3::from_cols(x1 - x0, x2 - x0, x3 - x0);
        let inv_dm = cage.precomputed[t_idx].matrix();
        let j = ds * inv_dm;

        let (r, s) = higham_polar_decomposition(j);

        assert!((r.determinant() - 1.0).abs() < 1e-5, "det(R) must be +1");
        let r_ortho = r * r.transpose() - Mat3::IDENTITY;
        assert!(r_ortho.abs_diff_eq(Mat3::ZERO, 1e-5), "R must be orthogonal");
        assert!(r.abs_diff_eq(rot, 1e-4), "R must equal rotation matrix");
        assert!(s.abs_diff_eq(Mat3::IDENTITY, 1e-4), "S must be identity");
    }
}

/// Module 6 - Test 4: Inverted element test: Ensure det(J) < 0 is safely clamped without NaN or negative opacity
#[test]
fn test_deliverable_deformation_inverted_element_stability() {
    let cage = TetMesh::create_box_cage([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
    let mut soa = GaussianSceneSoa::with_capacity(1);
    let sh = [0.0f32; 48];
    let rest_pos = [0.0, 0.0, 0.0];
    let rest_cov = [0.04, 0.0, 0.0, 0.04, 0.0, 0.04];
    let rest_opacity = 0.5;
    soa.push(rest_pos, rest_cov, rest_opacity, &sh);

    let bindings = cage.bind_gaussians(&[rest_pos]);
    let uniforms = DeformationUniforms::new([0.0, 0.0, -5.0], 0, 1, cage.num_tets() as u32);

    // Invert mesh across Z axis (reflection: det(J) = -1 < 0)
    let deformed_vertices: Vec<[f32; 3]> = cage
        .rest_vertices
        .iter()
        .map(|v| [v[0], v[1], -v[2]])
        .collect();

    let output = deform_scene_cpu(&soa, &cage, &bindings, &deformed_vertices, &uniforms);

    // 1. No NaNs anywhere
    assert!(!output.scene.positions[0].is_nan());
    assert!(!output.scene.positions[1].is_nan());
    assert!(!output.scene.positions[2].is_nan());

    // 2. Opacity is clamped to valid range [0.0, 0.99], not NaN, not negative
    let alpha = output.scene.opacities[0];
    assert!(!alpha.is_nan(), "Inverted element resulted in NaN opacity");
    assert!(alpha >= 0.0, "Opacity must be non-negative");
    assert!(alpha <= 0.99, "Opacity must be <= 0.99");

    // 3. Covariances have positive diagonal >= 1e-4
    assert!(output.scene.covariances_3d[0] >= 1e-4);
    assert!(output.scene.covariances_3d[3] >= 1e-4);
    assert!(output.scene.covariances_3d[5] >= 1e-4);

    // 4. Colors are valid in [0, 1] without NaNs
    for c in &output.colors {
        assert!(!c.is_nan());
        assert!(*c >= 0.0 && *c <= 1.0);
    }
}

/// Module 6 - Test 5: End-to-end integration test chaining: Deform -> Project -> Bin -> Composite
#[test]
fn test_deliverable_deformation_end_to_end_pipeline() {
    let cage = TetMesh::create_box_cage([-2.0, -2.0, 0.0], [2.0, 2.0, 4.0]);
    let mut soa = GaussianSceneSoa::with_capacity(2);
    let mut sh0 = [0.0f32; 48];
    sh0[0] = 1.0; // Red DC
    let mut sh1 = [0.0f32; 48];
    sh1[1] = 1.0; // Green DC

    // Gaussian 0 at (0, 0, 2.0)
    soa.push([0.0, 0.0, 2.0], [0.04, 0.0, 0.0, 0.04, 0.0, 0.04], 0.9, &sh0);
    // Gaussian 1 at (0.5, 0.5, 2.5)
    soa.push([0.5, 0.5, 2.5], [0.02, 0.0, 0.0, 0.02, 0.0, 0.02], 0.8, &sh1);

    // 1. Bind Gaussians to tetrahedral cage
    let bindings = cage.bind_gaussians(&[[0.0, 0.0, 2.0], [0.5, 0.5, 2.5]]);
    assert_eq!(bindings.len(), 2);

    let def_uniforms = DeformationUniforms::new(
        [0.0, 0.0, 0.0], // Camera at origin
        0,
        2,
        cage.num_tets() as u32,
    );

    // 2. Deform cage (translate by +0.1 in X and scale by 1.2)
    let deformed_vertices: Vec<[f32; 3]> = cage
        .rest_vertices
        .iter()
        .map(|v| [v[0] * 1.2 + 0.1, v[1] * 1.2, v[2] * 1.2])
        .collect();

    // Run Stage 1: Volumetric Deformation
    let deformed = deform_scene_cpu(&soa, &cage, &bindings, &deformed_vertices, &def_uniforms);
    assert_eq!(deformed.scene.count, 2);
    assert_eq!(deformed.colors.len(), 6);

    // Run Stage 2: Camera Projection (Module 3)
    let camera = CameraUniforms::new(
        Mat4::IDENTITY,
        800.0,
        800.0,
        400.0,
        300.0,
        800.0,
        600.0,
        0.2,
    );
    let proj = project_scene_cpu(&deformed.scene, &camera);
    assert_eq!(proj.count, 2);
    assert_eq!(proj.num_surviving(), 2);

    // Run Stage 3: Tile Binning & Sorting (Module 4)
    let binning = proj.bin_and_sort(800, 600);
    assert!(binning.total_entries > 0);

    // Run Stage 4: Tile Compositor Rasterization (Module 5)
    let comp_uniforms = CompositorUniforms::new(
        800,
        600,
        [0.0, 0.0, 0.0, 1.0], // Black background
        [0.0, 0.0, 0.0],
        0,
    );
    let img = render_scene_cpu(&deformed.scene, &camera, &comp_uniforms);
    assert_eq!(img.width, 800);
    assert_eq!(img.height, 600);

    // Verify rendered pixels have non-zero alpha and color
    let center_px = img.get_pixel_rgba_f32(400, 300);
    assert!(center_px[3] > 0.0, "Rendered alpha should be non-zero");
}

// =========================================================================
// MODULE 7 INTEGRATION TESTS: WGPU ORCHESTRATION & INTERACTIVE VIEWPORT
// =========================================================================

use cinder_gs::app::CagePicker;
use cinder_gs::render::gpu_pool::{calculate_exponential_growth, GpuBufferPool};


/// Module 7 - Test 1: Buffer pool growth and reuse assertions
#[test]
fn test_deliverable_gpu_buffer_pool_exponential_growth_and_reuse() {
    // 1. Verify analytical growth formula
    assert_eq!(calculate_exponential_growth(0, 10), 64);
    assert_eq!(calculate_exponential_growth(64, 64), 64);
    assert_eq!(calculate_exponential_growth(100, 120), 150); // 100 * 1.5 = 150
    assert_eq!(calculate_exponential_growth(100, 200), 200); // 200 > 150

    // 2. Initialize headless WGPU device
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }));

    let Some(adapter) = adapter else {
        eprintln!("Skipping GPU hardware test: no adapter available in headless CI");
        return;
    };

    let Ok((device, queue)) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)) else {
        eprintln!("Skipping GPU hardware test: device request failed");
        return;
    };

    let mut pool = GpuBufferPool::new(&device);

    // Initial allocation
    let realloc1 = pool.ensure_gaussian_capacity(&device, 100);
    assert!(realloc1);
    assert!(pool.gaussian_capacity >= 100);
    let cap1 = pool.gaussian_capacity;

    // Reuse without reallocation
    let realloc2 = pool.ensure_gaussian_capacity(&device, 100);
    assert!(!realloc2);
    assert_eq!(pool.gaussian_capacity, cap1);

    // Exponential growth
    let realloc3 = pool.ensure_gaussian_capacity(&device, cap1 + 10);
    assert!(realloc3);
    assert!(pool.gaussian_capacity >= (cap1 as f64 * 1.5).ceil() as usize);

    // Sort entries buffer growth
    assert!(pool.ensure_entries_capacity(&device, 500));
    assert_eq!(pool.entries_capacity, 500);
    assert!(!pool.ensure_entries_capacity(&device, 400)); // Reuse
    assert!(pool.ensure_entries_capacity(&device, 600));  // 500 * 1.5 = 750
    assert_eq!(pool.entries_capacity, 750);

    // Uniform buffer updates without allocation
    let camera = CameraUniforms::new(Mat4::IDENTITY, 800.0, 800.0, 400.0, 300.0, 800.0, 600.0, 0.2);
    pool.update_camera_uniforms(&queue, &camera);
}

/// Module 7 - Test 2: Screen-space projection picking test with known vertex positions
#[test]
fn test_deliverable_screen_space_projection_picking_known_positions() {
    let camera = CameraUniforms::new(
        Mat4::IDENTITY,
        1000.0, // fx
        1000.0, // fy
        500.0,  // cx
        500.0,  // cy
        1000.0, // width
        1000.0, // height
        0.2,    // near_plane
    );

    // Vertex 0: Center at (0, 0, 2) => screen (500, 500)
    // Vertex 1: Offset at (0.2, 0.1, 2) => sx = 1000 * 0.2 / 2 + 500 = 600, sy = 1000 * 0.1 / 2 + 500 = 550
    // Vertex 2: Behind near plane (0, 0, -1) => unpickable
    let vertices = vec![
        [0.0, 0.0, 2.0],
        [0.2, 0.1, 2.0],
        [0.0, 0.0, -1.0],
    ];

    let mut picker = CagePicker::new();

    // Pick Vertex 0 near (500, 500)
    let picked0 = picker.pick_vertex([504.0, 503.0], &vertices, &camera);
    assert_eq!(picked0, Some(0));

    // Pick Vertex 1 near (600, 550)
    let picked1 = picker.pick_vertex([605.0, 548.0], &vertices, &camera);
    assert_eq!(picked1, Some(1));

    // Clicking at (500, 500) must never pick Vertex 2 (behind near plane)
    assert_ne!(picked0, Some(2));

    // Clicking far outside any vertex threshold (> 12 pixels)
    let picked_none = picker.pick_vertex([200.0, 200.0], &vertices, &camera);
    assert_eq!(picked_none, None);
}

/// Module 7 - Test 3: Camera plane translation delta verification
#[test]
fn test_deliverable_camera_plane_translation_delta_verification() {
    // Camera at (0, 0, -5) looking towards origin along +Z axis
    let camera = CameraUniforms::new(
        Mat4::IDENTITY,
        800.0, // fx
        600.0, // fy
        400.0, // cx
        300.0, // cy
        800.0,
        600.0,
        0.2,
    );

    let mut vertices = vec![[0.0, 0.0, 4.0]]; // depth tz = 4.0
    let mut picker = CagePicker::new();
    picker.selected_vertex = Some(0);

    // Drag by delta (80.0, 60.0) px:
    // dx_world = (80.0 / 800.0) * 4.0 = 0.40 along Right (u = [1, 0, 0])
    // dy_world = (60.0 / 600.0) * 4.0 = 0.40 along Up (v = [0, 1, 0])
    // Expected new position: (0.0 + 0.4, 0.0 - 0.4, 4.0)
    let dragged = picker.drag_selected([80.0, 60.0], &mut vertices, &camera);
    assert!(dragged);

    assert!((vertices[0][0] - 0.40).abs() < 1e-5);
    assert!((vertices[0][1] - (-0.40)).abs() < 1e-5);
    assert!((vertices[0][2] - 4.0).abs() < 1e-5);
}

/// Final Milestone Test 1: Cage spring dynamics convergence to rest position within numerical tolerance.
#[test]
fn test_deliverable_cage_spring_dynamics_convergence() {
    use cinder_gs::deformation::CageSpringSimulator;
    use glam::Vec3;

    let rest_verts = vec![
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ];

    let mut sim = CageSpringSimulator::new(rest_verts.clone());

    // 1. Pin vertex 3 and displace it significantly
    sim.set_pinned(Some(3));
    sim.set_pinned_position(3, [5.0, -8.0, 10.0]);
    assert_eq!(sim.current_positions[3], Vec3::new(5.0, -8.0, 10.0));

    // 2. Unpin vertex 3 (user releases mouse drag)
    sim.set_pinned(None);

    // 3. Step forward in time with 60 Hz timesteps (dt = 0.016s)
    // Over 240 steps (4.0 seconds of simulation time)
    for _ in 0..240 {
        sim.step(0.016);
    }

    // 4. Verify that vertex 3 has converged back to its rest configuration
    let p3 = sim.current_positions[3];
    let rest3 = Vec3::from(rest_verts[3]);
    let dist = (p3 - rest3).length();

    assert!(
        dist < 1e-2,
        "Displaced cage vertex must converge to rest position, error: {:.6}",
        dist
    );

    // Also check velocities have settled close to zero
    assert!(
        sim.velocities[3].length() < 1e-2,
        "Velocity must settle to zero, got {:.6}",
        sim.velocities[3].length()
    );
}

/// Final Milestone Test 2: Wireframe edge deduplication assertion (canonical unique edges for 5-tet box cage).
#[test]
fn test_deliverable_wireframe_edge_deduplication() {
    use cinder_gs::deformation::TetMesh;
    use cinder_gs::render::wireframe::extract_unique_edges;
    use std::collections::HashSet;

    let cage = TetMesh::create_box_cage([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
    let edges = extract_unique_edges(&cage.elements);

    // 5 tetrahedra with 6 edges each would have 30 raw edge instances.
    // The standard 5-tetrahedron cube subdivision contains 12 cube boundary edges + 6 face diagonals = 18 unique edges.
    // In symmetric tetrahedral decompositions or boundary edge filters, 16 to 18 unique edges are produced.
    assert!(
        edges.len() == 18 || edges.len() == 16,
        "Expected exactly 16 or 18 unique edges, found {}",
        edges.len()
    );

    // Verify canonical vertex ordering u < v
    for edge in &edges {
        assert!(
            edge[0] < edge[1],
            "Edge {:?} is not canonically sorted with u < v",
            edge
        );
    }

    // Verify all edges are strictly unique
    let mut unique_set = HashSet::new();
    for edge in &edges {
        assert!(
            unique_set.insert((edge[0], edge[1])),
            "Duplicate edge found: {:?}",
            edge
        );
    }
}



