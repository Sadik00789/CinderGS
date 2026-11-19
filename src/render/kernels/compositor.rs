#![allow(clippy::excessive_precision)]

use cubecl::prelude::*;

/// Cooperative shared-memory tile compositor compute kernel.
///
/// Dispatched with `CubeDim::new_1d(256)` where each workgroup (cube) corresponds
/// to one 16x16 pixel tile `(CUBE_POS_X, CUBE_POS_Y)`. The SH evaluation mirrors
/// [`crate::render::rasterizer::sh::evaluate_sh`] for degrees 0 through 3.
#[allow(clippy::unnecessary_cast, clippy::manual_div_ceil)]
#[cube(launch)]
pub fn tile_compositor_kernel(
    points_xy: &[f32],
    conics: &[f32],
    opacities: &[f32],
    sh_coeffs: &[f32],
    positions_3d: &[f32],
    sorted_entries: &[u32],
    tile_ranges: &[u32],
    output_pixels: &mut [f32],
    screen_width: u32,
    screen_height: u32,
    tiles_x: u32,
    bg_r: f32,
    bg_g: f32,
    bg_b: f32,
    bg_a: f32,
    cam_pos_x: f32,
    cam_pos_y: f32,
    cam_pos_z: f32,
    sh_degree: u32,
) {
    let tx = CUBE_POS_X;
    let ty = CUBE_POS_Y;
    let local_id = UNIT_POS;
    let local_id_u32 = local_id as u32;

    let lx = local_id_u32 % 16u32;
    let ly = local_id_u32 / 16u32;

    let px = tx * 16u32 + lx;
    let py = ty * 16u32 + ly;
    let inside_screen = px < screen_width && py < screen_height;

    let tile_id = ty * tiles_x + tx;
    let range_start = tile_ranges[tile_id as usize * 2usize];
    let range_end = tile_ranges[tile_id as usize * 2usize + 1usize];
    let is_empty = range_start >= range_end;

    // Allocate 256-element shared memory arrays
    let mut shared_xy_x = Shared::<[f32]>::new_slice(256usize);
    let mut shared_xy_y = Shared::<[f32]>::new_slice(256usize);
    let mut shared_conic_a = Shared::<[f32]>::new_slice(256usize);
    let mut shared_conic_b = Shared::<[f32]>::new_slice(256usize);
    let mut shared_conic_c = Shared::<[f32]>::new_slice(256usize);
    let mut shared_opacity = Shared::<[f32]>::new_slice(256usize);
    let mut shared_rgb_r = Shared::<[f32]>::new_slice(256usize);
    let mut shared_rgb_g = Shared::<[f32]>::new_slice(256usize);
    let mut shared_rgb_b = Shared::<[f32]>::new_slice(256usize);
    let mut active_threads = Shared::<[u32]>::new_slice(256usize);

    let mut c_r = 0.0f32;
    let mut c_g = 0.0f32;
    let mut c_b = 0.0f32;
    let mut t = 1.0f32;
    let mut saturated = false;

    let mut total_in_tile = 0u32;
    if !is_empty {
        total_in_tile = range_end - range_start;
    }
    let num_batches = (total_in_tile + 255u32) / 256u32;

    let pix_center_x = px as f32 + 0.5f32;
    let pix_center_y = py as f32 + 0.5f32;

    for b in 0u32..num_batches {
        // Step 1: Cooperative loading into shared memory
        let entry_idx = range_start + b * 256u32 + local_id_u32;
        if entry_idx < range_end {
            let g = sorted_entries[entry_idx as usize * 3usize + 2usize] as usize;
            shared_xy_x[local_id as usize] = points_xy[g * 2usize];
            shared_xy_y[local_id as usize] = points_xy[g * 2usize + 1usize];
            shared_conic_a[local_id as usize] = conics[g * 3usize];
            shared_conic_b[local_id as usize] = conics[g * 3usize + 1usize];
            shared_conic_c[local_id as usize] = conics[g * 3usize + 2usize];
            shared_opacity[local_id as usize] = opacities[g];

            // Evaluate SH colour along the world-space view direction.
            let pos_x = positions_3d[g * 3usize];
            let pos_y = positions_3d[g * 3usize + 1usize];
            let pos_z = positions_3d[g * 3usize + 2usize];

            let vx = pos_x - cam_pos_x;
            let vy = pos_y - cam_pos_y;
            let vz = pos_z - cam_pos_z;
            let dist_sq = vx * vx + vy * vy + vz * vz;
            let inv_len = 1.0f32 / f32::max(0.0001f32, f32::sqrt(dist_sq));
            let dir_x = vx * inv_len;
            let dir_y = vy * inv_len;
            let dir_z = vz * inv_len;

            let sh_base = g * 48usize;

            // Degree 0 (DC term)
            let mut color_r = 0.282094791f32 * sh_coeffs[sh_base];
            let mut color_g = 0.282094791f32 * sh_coeffs[sh_base + 1usize];
            let mut color_b = 0.282094791f32 * sh_coeffs[sh_base + 2usize];

            // Degree 1 (3 basis functions)
            if sh_degree >= 1u32 {
                let y1_m1 = -0.488602511f32 * dir_y;
                let y1_0 = 0.488602511f32 * dir_z;
                let y1_p1 = -0.488602511f32 * dir_x;

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

            // Degree 2 (5 basis functions)
            if sh_degree >= 2u32 {
                let xx = dir_x * dir_x;
                let yy = dir_y * dir_y;
                let zz = dir_z * dir_z;
                let xy = dir_x * dir_y;
                let yz = dir_y * dir_z;
                let xz = dir_x * dir_z;

                let y2_m2 = 1.092548431f32 * xy;
                let y2_m1 = -1.092548431f32 * yz;
                let y2_0 = 0.315391565f32 * (2.0f32 * zz - xx - yy);
                let y2_p1 = -1.092548431f32 * xz;
                let y2_p2 = 0.546274215f32 * (xx - yy);

                color_r += y2_m2 * sh_coeffs[sh_base + 6usize]
                    + y2_m1 * sh_coeffs[sh_base + 7usize]
                    + y2_0 * sh_coeffs[sh_base + 8usize]
                    + y2_p1 * sh_coeffs[sh_base + 9usize]
                    + y2_p2 * sh_coeffs[sh_base + 10usize];
                color_g += y2_m2 * sh_coeffs[sh_base + 21usize]
                    + y2_m1 * sh_coeffs[sh_base + 22usize]
                    + y2_0 * sh_coeffs[sh_base + 23usize]
                    + y2_p1 * sh_coeffs[sh_base + 24usize]
                    + y2_p2 * sh_coeffs[sh_base + 25usize];
                color_b += y2_m2 * sh_coeffs[sh_base + 36usize]
                    + y2_m1 * sh_coeffs[sh_base + 37usize]
                    + y2_0 * sh_coeffs[sh_base + 38usize]
                    + y2_p1 * sh_coeffs[sh_base + 39usize]
                    + y2_p2 * sh_coeffs[sh_base + 40usize];

                // Degree 3 (7 basis functions)
                if sh_degree >= 3u32 {
                    let y3_m3 = -0.59004359f32 * dir_y * (3.0f32 * xx - yy);
                    let y3_m2 = 2.89061144f32 * xy * dir_z;
                    let y3_m1 = -0.457045794f32 * dir_y * (4.0f32 * zz - xx - yy);
                    let y3_0 = 0.373176336f32 * dir_z * (2.0f32 * zz - 3.0f32 * xx - 3.0f32 * yy);
                    let y3_p1 = -0.457045794f32 * dir_x * (4.0f32 * zz - xx - yy);
                    let y3_p2 = 1.445305721f32 * dir_z * (xx - yy);
                    let y3_p3 = -0.59004359f32 * dir_x * (xx - 3.0f32 * yy);

                    color_r += y3_m3 * sh_coeffs[sh_base + 11usize]
                        + y3_m2 * sh_coeffs[sh_base + 12usize]
                        + y3_m1 * sh_coeffs[sh_base + 13usize]
                        + y3_0 * sh_coeffs[sh_base + 14usize]
                        + y3_p1 * sh_coeffs[sh_base + 15usize]
                        + y3_p2 * sh_coeffs[sh_base + 16usize]
                        + y3_p3 * sh_coeffs[sh_base + 17usize];
                    color_g += y3_m3 * sh_coeffs[sh_base + 26usize]
                        + y3_m2 * sh_coeffs[sh_base + 27usize]
                        + y3_m1 * sh_coeffs[sh_base + 28usize]
                        + y3_0 * sh_coeffs[sh_base + 29usize]
                        + y3_p1 * sh_coeffs[sh_base + 30usize]
                        + y3_p2 * sh_coeffs[sh_base + 31usize]
                        + y3_p3 * sh_coeffs[sh_base + 32usize];
                    color_b += y3_m3 * sh_coeffs[sh_base + 41usize]
                        + y3_m2 * sh_coeffs[sh_base + 42usize]
                        + y3_m1 * sh_coeffs[sh_base + 43usize]
                        + y3_0 * sh_coeffs[sh_base + 44usize]
                        + y3_p1 * sh_coeffs[sh_base + 45usize]
                        + y3_p2 * sh_coeffs[sh_base + 46usize]
                        + y3_p3 * sh_coeffs[sh_base + 47usize];
                }
            }

            shared_rgb_r[local_id as usize] = f32::clamp(color_r + 0.5f32, 0.0f32, 1.0f32);
            shared_rgb_g[local_id as usize] = f32::clamp(color_g + 0.5f32, 0.0f32, 1.0f32);
            shared_rgb_b[local_id as usize] = f32::clamp(color_b + 0.5f32, 0.0f32, 1.0f32);
        } else {
            shared_opacity[local_id as usize] = 0.0f32;
        }

        // Barrier: Wait for cooperative load to complete
        sync_cube();

        // Step 2: Conic evaluation and alpha blending
        let batch_count = u32::min(256u32, total_in_tile - b * 256u32);

        if inside_screen && !saturated {
            for k in 0u32..batch_count {
                let op_base = shared_opacity[k as usize];
                if op_base > 0.003921569f32 {
                    let mu_x = shared_xy_x[k as usize];
                    let mu_y = shared_xy_y[k as usize];

                    let dx = pix_center_x - mu_x;
                    let dy = pix_center_y - mu_y;

                    let a = shared_conic_a[k as usize];
                    let b_coef = shared_conic_b[k as usize];
                    let c = shared_conic_c[k as usize];

                    let mut q = -0.5f32 * (a * dx * dx + 2.0f32 * b_coef * dx * dy + c * dy * dy);
                    if q > 0.0f32 {
                        q = 0.0f32;
                    }

                    if q >= -4.0f32 {
                        let alpha = f32::min(0.99f32, op_base * f32::exp(q));

                        if alpha >= 0.003921569f32 {
                            let w = alpha * t;
                            c_r += w * shared_rgb_r[k as usize];
                            c_g += w * shared_rgb_g[k as usize];
                            c_b += w * shared_rgb_b[k as usize];

                            t *= 1.0f32 - alpha;
                            if t < 0.001f32 {
                                saturated = true;
                                t = 0.0f32;
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Barrier: Ensure all threads finished using current shared memory before next batch
        sync_cube();

        // Tile-wide early transmittance termination (T < epsilon across all 256 pixels).
        active_threads[local_id as usize] = if inside_screen && !saturated {
            1u32
        } else {
            0u32
        };
        sync_cube();
        if local_id_u32 < 128u32 {
            active_threads[local_id as usize] += active_threads[local_id as usize + 128usize];
        }
        sync_cube();
        if local_id_u32 < 64u32 {
            active_threads[local_id as usize] += active_threads[local_id as usize + 64usize];
        }
        sync_cube();
        if local_id_u32 < 32u32 {
            active_threads[local_id as usize] += active_threads[local_id as usize + 32usize];
        }
        sync_cube();
        if local_id_u32 < 16u32 {
            active_threads[local_id as usize] += active_threads[local_id as usize + 16usize];
        }
        sync_cube();
        if local_id_u32 < 8u32 {
            active_threads[local_id as usize] += active_threads[local_id as usize + 8usize];
        }
        sync_cube();
        if local_id_u32 < 4u32 {
            active_threads[local_id as usize] += active_threads[local_id as usize + 4usize];
        }
        sync_cube();
        if local_id_u32 < 2u32 {
            active_threads[local_id as usize] += active_threads[local_id as usize + 2usize];
        }
        sync_cube();
        if local_id_u32 < 1u32 {
            active_threads[local_id as usize] += active_threads[local_id as usize + 1usize];
        }
        sync_cube();

        if active_threads[0usize] == 0u32 {
            break;
        }
    }

    // Final color write
    if inside_screen {
        let final_r = f32::clamp(c_r + t * bg_r, 0.0f32, 1.0f32);
        let final_g = f32::clamp(c_g + t * bg_g, 0.0f32, 1.0f32);
        let final_b = f32::clamp(c_b + t * bg_b, 0.0f32, 1.0f32);
        let mut final_a = f32::clamp(1.0f32 - t, 0.0f32, 1.0f32);
        if is_empty {
            final_a = bg_a;
        }

        let out_base = (py * screen_width + px) as usize * 4usize;
        output_pixels[out_base] = final_r;
        output_pixels[out_base + 1usize] = final_g;
        output_pixels[out_base + 2usize] = final_b;
        output_pixels[out_base + 3usize] = final_a;
    }
}

#[cube]
fn unpack_f16_comp(bits: u32) -> f32 {
    crate::render::kernels::project::unpack_f16(bits)
}

#[cube]
fn unpack_sh_coeff(sh_f16_packed: &[u32], word_base: usize, coeff_idx: usize) -> f32 {
    let word = sh_f16_packed[word_base + (coeff_idx / 2usize)];
    let bits = if (coeff_idx % 2usize) == 0usize {
        word & 0xFFFFu32
    } else {
        (word >> 16u32) & 0xFFFFu32
    };
    unpack_f16_comp(bits)
}

/// Quantized variant of [`tile_compositor_kernel`] reading `f16`-packed SH coefficients
/// (`sh_words_per_splat` `u32` words per splat) with tile-wide early transmittance termination.
#[allow(
    clippy::unnecessary_cast,
    clippy::manual_div_ceil,
    clippy::too_many_arguments
)]
#[cube(launch)]
pub fn tile_compositor_quantized_kernel(
    points_xy: &[f32],
    conics: &[f32],
    opacities: &[f32],
    sh_coeffs_f16_packed: &[u32],
    positions_3d: &[f32],
    sorted_entries: &[u32],
    tile_ranges: &[u32],
    output_pixels: &mut [f32],
    screen_width: u32,
    screen_height: u32,
    tiles_x: u32,
    bg_r: f32,
    bg_g: f32,
    bg_b: f32,
    bg_a: f32,
    cam_pos_x: f32,
    cam_pos_y: f32,
    cam_pos_z: f32,
    sh_degree: u32,
) {
    let tx = CUBE_POS_X;
    let ty = CUBE_POS_Y;
    let local_id = UNIT_POS;
    let local_id_u32 = local_id as u32;

    let lx = local_id_u32 % 16u32;
    let ly = local_id_u32 / 16u32;

    let px = tx * 16u32 + lx;
    let py = ty * 16u32 + ly;
    let inside_screen = px < screen_width && py < screen_height;

    let tile_id = ty * tiles_x + tx;
    let range_start = tile_ranges[tile_id as usize * 2usize];
    let range_end = tile_ranges[tile_id as usize * 2usize + 1usize];
    let is_empty = range_start >= range_end;

    let bases_per_ch = if sh_degree == 0u32 {
        1usize
    } else if sh_degree == 1u32 {
        4usize
    } else if sh_degree == 2u32 {
        9usize
    } else {
        16usize
    };
    let sh_words_per_splat = (bases_per_ch * 3usize + 1usize) / 2usize;
    let rest_per_ch = bases_per_ch - 1usize;
    let r_rest = 3usize;
    let g_rest = 3usize + rest_per_ch;
    let b_rest = 3usize + rest_per_ch * 2usize;

    let mut shared_xy_x = Shared::<[f32]>::new_slice(256usize);
    let mut shared_xy_y = Shared::<[f32]>::new_slice(256usize);
    let mut shared_conic_a = Shared::<[f32]>::new_slice(256usize);
    let mut shared_conic_b = Shared::<[f32]>::new_slice(256usize);
    let mut shared_conic_c = Shared::<[f32]>::new_slice(256usize);
    let mut shared_opacity = Shared::<[f32]>::new_slice(256usize);
    let mut shared_rgb_r = Shared::<[f32]>::new_slice(256usize);
    let mut shared_rgb_g = Shared::<[f32]>::new_slice(256usize);
    let mut shared_rgb_b = Shared::<[f32]>::new_slice(256usize);
    let mut active_threads = Shared::<[u32]>::new_slice(256usize);

    let mut c_r = 0.0f32;
    let mut c_g = 0.0f32;
    let mut c_b = 0.0f32;
    let mut t = 1.0f32;
    let mut saturated = false;

    let mut total_in_tile = 0u32;
    if !is_empty {
        total_in_tile = range_end - range_start;
    }
    let num_batches = (total_in_tile + 255u32) / 256u32;

    let pix_center_x = px as f32 + 0.5f32;
    let pix_center_y = py as f32 + 0.5f32;

    for b in 0u32..num_batches {
        let entry_idx = range_start + b * 256u32 + local_id_u32;
        if entry_idx < range_end {
            let g = sorted_entries[entry_idx as usize * 3usize + 2usize] as usize;
            shared_xy_x[local_id as usize] = points_xy[g * 2usize];
            shared_xy_y[local_id as usize] = points_xy[g * 2usize + 1usize];
            shared_conic_a[local_id as usize] = conics[g * 3usize];
            shared_conic_b[local_id as usize] = conics[g * 3usize + 1usize];
            shared_conic_c[local_id as usize] = conics[g * 3usize + 2usize];
            shared_opacity[local_id as usize] = opacities[g];

            let pos_x = positions_3d[g * 3usize];
            let pos_y = positions_3d[g * 3usize + 1usize];
            let pos_z = positions_3d[g * 3usize + 2usize];

            let vx = pos_x - cam_pos_x;
            let vy = pos_y - cam_pos_y;
            let vz = pos_z - cam_pos_z;
            let inv_len = 1.0f32 / f32::max(f32::sqrt(vx * vx + vy * vy + vz * vz), 1e-8f32);
            let x = vx * inv_len;
            let y = vy * inv_len;
            let z = vz * inv_len;

            let word_base = g * sh_words_per_splat;
            let mut color_r =
                0.28209479177387814f32 * unpack_sh_coeff(sh_coeffs_f16_packed, word_base, 0usize);
            let mut color_g =
                0.28209479177387814f32 * unpack_sh_coeff(sh_coeffs_f16_packed, word_base, 1usize);
            let mut color_b =
                0.28209479177387814f32 * unpack_sh_coeff(sh_coeffs_f16_packed, word_base, 2usize);

            if sh_degree >= 1u32 {
                let y1_0 = -0.4886025119029199f32 * y;
                let y1_1 = 0.4886025119029199f32 * z;
                let y1_2 = -0.4886025119029199f32 * x;
                color_r += y1_0 * unpack_sh_coeff(sh_coeffs_f16_packed, word_base, r_rest)
                    + y1_1 * unpack_sh_coeff(sh_coeffs_f16_packed, word_base, r_rest + 1usize)
                    + y1_2 * unpack_sh_coeff(sh_coeffs_f16_packed, word_base, r_rest + 2usize);
                color_g += y1_0 * unpack_sh_coeff(sh_coeffs_f16_packed, word_base, g_rest)
                    + y1_1 * unpack_sh_coeff(sh_coeffs_f16_packed, word_base, g_rest + 1usize)
                    + y1_2 * unpack_sh_coeff(sh_coeffs_f16_packed, word_base, g_rest + 2usize);
                color_b += y1_0 * unpack_sh_coeff(sh_coeffs_f16_packed, word_base, b_rest)
                    + y1_1 * unpack_sh_coeff(sh_coeffs_f16_packed, word_base, b_rest + 1usize)
                    + y1_2 * unpack_sh_coeff(sh_coeffs_f16_packed, word_base, b_rest + 2usize);

                if sh_degree >= 2u32 {
                    let xx = x * x;
                    let yy = y * y;
                    let zz = z * z;
                    let xy = x * y;
                    let yz = y * z;
                    let xz = x * z;
                    let y2_0 = 1.0925484305920792f32 * xy;
                    let y2_1 = -1.0925484305920792f32 * yz;
                    let y2_2 = 0.31539156525252005f32 * (2.0f32 * zz - xx - yy);
                    let y2_3 = -1.0925484305920792f32 * xz;
                    let y2_4 = 0.5462742152960396f32 * (xx - yy);

                    color_r += y2_0
                        * unpack_sh_coeff(sh_coeffs_f16_packed, word_base, r_rest + 3usize)
                        + y2_1 * unpack_sh_coeff(sh_coeffs_f16_packed, word_base, r_rest + 4usize)
                        + y2_2 * unpack_sh_coeff(sh_coeffs_f16_packed, word_base, r_rest + 5usize)
                        + y2_3 * unpack_sh_coeff(sh_coeffs_f16_packed, word_base, r_rest + 6usize)
                        + y2_4 * unpack_sh_coeff(sh_coeffs_f16_packed, word_base, r_rest + 7usize);
                    color_g += y2_0
                        * unpack_sh_coeff(sh_coeffs_f16_packed, word_base, g_rest + 3usize)
                        + y2_1 * unpack_sh_coeff(sh_coeffs_f16_packed, word_base, g_rest + 4usize)
                        + y2_2 * unpack_sh_coeff(sh_coeffs_f16_packed, word_base, g_rest + 5usize)
                        + y2_3 * unpack_sh_coeff(sh_coeffs_f16_packed, word_base, g_rest + 6usize)
                        + y2_4 * unpack_sh_coeff(sh_coeffs_f16_packed, word_base, g_rest + 7usize);
                    color_b += y2_0
                        * unpack_sh_coeff(sh_coeffs_f16_packed, word_base, b_rest + 3usize)
                        + y2_1 * unpack_sh_coeff(sh_coeffs_f16_packed, word_base, b_rest + 4usize)
                        + y2_2 * unpack_sh_coeff(sh_coeffs_f16_packed, word_base, b_rest + 5usize)
                        + y2_3 * unpack_sh_coeff(sh_coeffs_f16_packed, word_base, b_rest + 6usize)
                        + y2_4 * unpack_sh_coeff(sh_coeffs_f16_packed, word_base, b_rest + 7usize);

                    if sh_degree >= 3u32 {
                        let y3_m3 = -0.5900435899266435f32 * y * (3.0f32 * xx - yy);
                        let y3_m2 = 2.890611442640554f32 * xy * z;
                        let y3_m1 = -0.4570457994644658f32 * y * (4.0f32 * zz - xx - yy);
                        let y3_0 =
                            0.3731763325901154f32 * z * (2.0f32 * zz - 3.0f32 * xx - 3.0f32 * yy);
                        let y3_p1 = -0.4570457994644658f32 * x * (4.0f32 * zz - xx - yy);
                        let y3_p2 = 1.445305721320277f32 * z * (xx - yy);
                        let y3_p3 = -0.5900435899266435f32 * x * (xx - 3.0f32 * yy);

                        color_r += y3_m3
                            * unpack_sh_coeff(sh_coeffs_f16_packed, word_base, r_rest + 8usize)
                            + y3_m2
                                * unpack_sh_coeff(sh_coeffs_f16_packed, word_base, r_rest + 9usize)
                            + y3_m1
                                * unpack_sh_coeff(
                                    sh_coeffs_f16_packed,
                                    word_base,
                                    r_rest + 10usize,
                                )
                            + y3_0
                                * unpack_sh_coeff(
                                    sh_coeffs_f16_packed,
                                    word_base,
                                    r_rest + 11usize,
                                )
                            + y3_p1
                                * unpack_sh_coeff(
                                    sh_coeffs_f16_packed,
                                    word_base,
                                    r_rest + 12usize,
                                )
                            + y3_p2
                                * unpack_sh_coeff(
                                    sh_coeffs_f16_packed,
                                    word_base,
                                    r_rest + 13usize,
                                )
                            + y3_p3
                                * unpack_sh_coeff(
                                    sh_coeffs_f16_packed,
                                    word_base,
                                    r_rest + 14usize,
                                );
                        color_g += y3_m3
                            * unpack_sh_coeff(sh_coeffs_f16_packed, word_base, g_rest + 8usize)
                            + y3_m2
                                * unpack_sh_coeff(sh_coeffs_f16_packed, word_base, g_rest + 9usize)
                            + y3_m1
                                * unpack_sh_coeff(
                                    sh_coeffs_f16_packed,
                                    word_base,
                                    g_rest + 10usize,
                                )
                            + y3_0
                                * unpack_sh_coeff(
                                    sh_coeffs_f16_packed,
                                    word_base,
                                    g_rest + 11usize,
                                )
                            + y3_p1
                                * unpack_sh_coeff(
                                    sh_coeffs_f16_packed,
                                    word_base,
                                    g_rest + 12usize,
                                )
                            + y3_p2
                                * unpack_sh_coeff(
                                    sh_coeffs_f16_packed,
                                    word_base,
                                    g_rest + 13usize,
                                )
                            + y3_p3
                                * unpack_sh_coeff(
                                    sh_coeffs_f16_packed,
                                    word_base,
                                    g_rest + 14usize,
                                );
                        color_b += y3_m3
                            * unpack_sh_coeff(sh_coeffs_f16_packed, word_base, b_rest + 8usize)
                            + y3_m2
                                * unpack_sh_coeff(sh_coeffs_f16_packed, word_base, b_rest + 9usize)
                            + y3_m1
                                * unpack_sh_coeff(
                                    sh_coeffs_f16_packed,
                                    word_base,
                                    b_rest + 10usize,
                                )
                            + y3_0
                                * unpack_sh_coeff(
                                    sh_coeffs_f16_packed,
                                    word_base,
                                    b_rest + 11usize,
                                )
                            + y3_p1
                                * unpack_sh_coeff(
                                    sh_coeffs_f16_packed,
                                    word_base,
                                    b_rest + 12usize,
                                )
                            + y3_p2
                                * unpack_sh_coeff(
                                    sh_coeffs_f16_packed,
                                    word_base,
                                    b_rest + 13usize,
                                )
                            + y3_p3
                                * unpack_sh_coeff(
                                    sh_coeffs_f16_packed,
                                    word_base,
                                    b_rest + 14usize,
                                );
                    }
                }
            }

            shared_rgb_r[local_id as usize] = f32::clamp(color_r + 0.5f32, 0.0f32, 1.0f32);
            shared_rgb_g[local_id as usize] = f32::clamp(color_g + 0.5f32, 0.0f32, 1.0f32);
            shared_rgb_b[local_id as usize] = f32::clamp(color_b + 0.5f32, 0.0f32, 1.0f32);
        } else {
            shared_opacity[local_id as usize] = 0.0f32;
        }

        sync_cube();

        let batch_count = u32::min(256u32, total_in_tile - b * 256u32);
        if inside_screen && !saturated {
            for k in 0u32..batch_count {
                let op_base = shared_opacity[k as usize];
                if op_base > 0.003921569f32 {
                    let mu_x = shared_xy_x[k as usize];
                    let mu_y = shared_xy_y[k as usize];
                    let dx = pix_center_x - mu_x;
                    let dy = pix_center_y - mu_y;
                    let a = shared_conic_a[k as usize];
                    let b_coef = shared_conic_b[k as usize];
                    let c = shared_conic_c[k as usize];

                    let mut q = -0.5f32 * (a * dx * dx + 2.0f32 * b_coef * dx * dy + c * dy * dy);
                    if q > 0.0f32 {
                        q = 0.0f32;
                    }
                    if q >= -4.0f32 {
                        let alpha = f32::min(0.99f32, op_base * f32::exp(q));
                        if alpha >= 0.003921569f32 {
                            let w = alpha * t;
                            c_r += w * shared_rgb_r[k as usize];
                            c_g += w * shared_rgb_g[k as usize];
                            c_b += w * shared_rgb_b[k as usize];
                            t *= 1.0f32 - alpha;
                            if t < 0.001f32 {
                                saturated = true;
                                t = 0.0f32;
                                break;
                            }
                        }
                    }
                }
            }
        }

        sync_cube();

        active_threads[local_id as usize] = if inside_screen && !saturated {
            1u32
        } else {
            0u32
        };
        sync_cube();
        if local_id_u32 < 128u32 {
            active_threads[local_id as usize] += active_threads[local_id as usize + 128usize];
        }
        sync_cube();
        if local_id_u32 < 64u32 {
            active_threads[local_id as usize] += active_threads[local_id as usize + 64usize];
        }
        sync_cube();
        if local_id_u32 < 32u32 {
            active_threads[local_id as usize] += active_threads[local_id as usize + 32usize];
        }
        sync_cube();
        if local_id_u32 < 16u32 {
            active_threads[local_id as usize] += active_threads[local_id as usize + 16usize];
        }
        sync_cube();
        if local_id_u32 < 8u32 {
            active_threads[local_id as usize] += active_threads[local_id as usize + 8usize];
        }
        sync_cube();
        if local_id_u32 < 4u32 {
            active_threads[local_id as usize] += active_threads[local_id as usize + 4usize];
        }
        sync_cube();
        if local_id_u32 < 2u32 {
            active_threads[local_id as usize] += active_threads[local_id as usize + 2usize];
        }
        sync_cube();
        if local_id_u32 < 1u32 {
            active_threads[local_id as usize] += active_threads[local_id as usize + 1usize];
        }
        sync_cube();

        if active_threads[0usize] == 0u32 {
            break;
        }
    }

    if inside_screen {
        let final_r = f32::clamp(c_r + t * bg_r, 0.0f32, 1.0f32);
        let final_g = f32::clamp(c_g + t * bg_g, 0.0f32, 1.0f32);
        let final_b = f32::clamp(c_b + t * bg_b, 0.0f32, 1.0f32);
        let mut final_a = f32::clamp(1.0f32 - t, 0.0f32, 1.0f32);
        if is_empty {
            final_a = bg_a;
        }

        let out_base = (py * screen_width + px) as usize * 4usize;
        output_pixels[out_base] = final_r;
        output_pixels[out_base + 1usize] = final_g;
        output_pixels[out_base + 2usize] = final_b;
        output_pixels[out_base + 3usize] = final_a;
    }
}

/// High-throughput Stage 4 tile compositor reading pre-evaluated Stage 2 view-dependent RGB
/// (`splat_attrs` = 10 floats/splat) and flat `sorted_splat_ids`, packing quantized RGBA8 directly
/// into `u32` rows aligned to `padded_row_u32` (zero-copy WGPU texture transfer ready).
///
/// Uses double-buffered shared `Atomic<u32>` early tile termination (with subgroup `plane_any`
/// ballot coalescing when `use_subgroups` is true).
#[allow(
    clippy::unnecessary_cast,
    clippy::manual_div_ceil,
    clippy::too_many_arguments
)]
#[cube(launch)]
pub fn tile_compositor_v2_kernel(
    splat_attrs: &[f32],
    sorted_splat_ids: &[u32],
    tile_ranges: &[u32],
    output_rgba8: &mut [u32],
    screen_width: u32,
    screen_height: u32,
    tiles_x: u32,
    padded_row_u32: u32,
    bg_r: f32,
    bg_g: f32,
    bg_b: f32,
    bg_a: f32,
    #[comptime] use_subgroups: bool,
) {
    let tx = CUBE_POS_X;
    let ty = CUBE_POS_Y;
    let local_id = UNIT_POS;
    let local_id_u32 = local_id as u32;

    let lx = local_id_u32 % 16u32;
    let ly = local_id_u32 / 16u32;
    let px = tx * 16u32 + lx;
    let py = ty * 16u32 + ly;
    let inside_screen = px < screen_width && py < screen_height;

    let tile_id = ty * tiles_x + tx;
    let range_start = tile_ranges[tile_id as usize * 2usize];
    let range_end = tile_ranges[tile_id as usize * 2usize + 1usize];
    let is_empty = range_start >= range_end;

    let mut shared_xy_x = Shared::<[f32]>::new_slice(256usize);
    let mut shared_xy_y = Shared::<[f32]>::new_slice(256usize);
    let mut shared_conic_a = Shared::<[f32]>::new_slice(256usize);
    let mut shared_conic_b = Shared::<[f32]>::new_slice(256usize);
    let mut shared_conic_c = Shared::<[f32]>::new_slice(256usize);
    let mut shared_opacity = Shared::<[f32]>::new_slice(256usize);
    let mut shared_rgb_r = Shared::<[f32]>::new_slice(256usize);
    let mut shared_rgb_g = Shared::<[f32]>::new_slice(256usize);
    let mut shared_rgb_b = Shared::<[f32]>::new_slice(256usize);
    let tile_alive = Shared::<[Atomic<u32>]>::new_slice(2usize);

    if local_id_u32 < 2u32 {
        tile_alive[local_id as usize].store(0u32);
    }
    sync_cube();

    let mut c_r = 0.0f32;
    let mut c_g = 0.0f32;
    let mut c_b = 0.0f32;
    let mut t = 1.0f32;
    let mut saturated = false;

    let total_in_tile = if is_empty {
        0u32
    } else {
        range_end - range_start
    };
    let num_batches = (total_in_tile + 255u32) / 256u32;

    let pix_center_x = px as f32 + 0.5f32;
    let pix_center_y = py as f32 + 0.5f32;

    for b in 0u32..num_batches {
        let entry_idx = range_start + b * 256u32 + local_id_u32;
        if entry_idx < range_end {
            let g = sorted_splat_ids[entry_idx as usize] as usize;
            let attr_base = g * 10usize;
            shared_xy_x[local_id as usize] = splat_attrs[attr_base];
            shared_xy_y[local_id as usize] = splat_attrs[attr_base + 1usize];
            shared_conic_a[local_id as usize] = splat_attrs[attr_base + 2usize];
            shared_conic_b[local_id as usize] = splat_attrs[attr_base + 3usize];
            shared_conic_c[local_id as usize] = splat_attrs[attr_base + 4usize];
            shared_opacity[local_id as usize] = splat_attrs[attr_base + 5usize];
            shared_rgb_r[local_id as usize] = splat_attrs[attr_base + 6usize];
            shared_rgb_g[local_id as usize] = splat_attrs[attr_base + 7usize];
            shared_rgb_b[local_id as usize] = splat_attrs[attr_base + 8usize];
        } else {
            shared_opacity[local_id as usize] = 0.0f32;
        }
        sync_cube();

        let slot = (b & 1u32) as usize;
        let next_slot = ((b + 1u32) & 1u32) as usize;
        if local_id_u32 == 0u32 {
            tile_alive[next_slot].store(0u32);
        }

        let batch_count = u32::min(256u32, total_in_tile - b * 256u32);
        if inside_screen && !saturated {
            for k in 0u32..batch_count {
                let op_base = shared_opacity[k as usize];
                if op_base > 0.003921569f32 {
                    let mu_x = shared_xy_x[k as usize];
                    let mu_y = shared_xy_y[k as usize];
                    let dx = pix_center_x - mu_x;
                    let dy = pix_center_y - mu_y;
                    let a = shared_conic_a[k as usize];
                    let b_coef = shared_conic_b[k as usize];
                    let c = shared_conic_c[k as usize];

                    let mut q = -0.5f32 * (a * dx * dx + 2.0f32 * b_coef * dx * dy + c * dy * dy);
                    if q > 0.0f32 {
                        q = 0.0f32;
                    }
                    if q >= -4.0f32 {
                        let alpha = f32::min(0.99f32, op_base * f32::exp(q));
                        if alpha >= 0.003921569f32 {
                            let w = alpha * t;
                            c_r += w * shared_rgb_r[k as usize];
                            c_g += w * shared_rgb_g[k as usize];
                            c_b += w * shared_rgb_b[k as usize];
                            t *= 1.0f32 - alpha;
                            if t < 0.001f32 {
                                saturated = true;
                                t = 0.0f32;
                                break;
                            }
                        }
                    }
                }
            }
        }

        let is_alive = inside_screen && !saturated;
        if comptime!(use_subgroups) {
            let lane_any = plane_any(is_alive);
            if lane_any && UNIT_POS_PLANE == 0u32 {
                tile_alive[slot].store(1u32);
            }
        } else if is_alive {
            tile_alive[slot].store(1u32);
        }
        sync_cube();

        if tile_alive[slot].load() == 0u32 {
            break;
        }
    }

    if inside_screen {
        let final_r = f32::clamp(c_r + t * bg_r, 0.0f32, 1.0f32);
        let final_g = f32::clamp(c_g + t * bg_g, 0.0f32, 1.0f32);
        let final_b = f32::clamp(c_b + t * bg_b, 0.0f32, 1.0f32);
        let mut final_a = f32::clamp(1.0f32 - t, 0.0f32, 1.0f32);
        if is_empty {
            final_a = bg_a;
        }

        let r8 = f32::clamp(f32::round(final_r * 255.0f32), 0.0f32, 255.0f32) as u32;
        let g8 = f32::clamp(f32::round(final_g * 255.0f32), 0.0f32, 255.0f32) as u32;
        let b8 = f32::clamp(f32::round(final_b * 255.0f32), 0.0f32, 255.0f32) as u32;
        let a8 = f32::clamp(f32::round(final_a * 255.0f32), 0.0f32, 255.0f32) as u32;
        output_rgba8[(py * padded_row_u32 + px) as usize] =
            r8 | (g8 << 8u32) | (b8 << 16u32) | (a8 << 24u32);
    }
}

/// Native `half::f16` SH variant of [`tile_compositor_quantized_kernel`] when
/// `SHADER_F16` is available on the adapter.
#[allow(
    clippy::unnecessary_cast,
    clippy::manual_div_ceil,
    clippy::too_many_arguments
)]
#[cube(launch)]
pub fn tile_compositor_quantized_f16_kernel(
    points_xy: &[f32],
    conics: &[f32],
    opacities: &[f32],
    sh_coeffs_f16: &[half::f16],
    positions_3d: &[f32],
    sorted_entries: &[u32],
    tile_ranges: &[u32],
    output_pixels: &mut [f32],
    screen_width: u32,
    screen_height: u32,
    tiles_x: u32,
    bg_r: f32,
    bg_g: f32,
    bg_b: f32,
    bg_a: f32,
    cam_pos_x: f32,
    cam_pos_y: f32,
    cam_pos_z: f32,
    sh_degree: u32,
) {
    let tx = CUBE_POS_X;
    let ty = CUBE_POS_Y;
    let local_id = UNIT_POS;
    let local_id_u32 = local_id as u32;

    let lx = local_id_u32 % 16u32;
    let ly = local_id_u32 / 16u32;

    let px = tx * 16u32 + lx;
    let py = ty * 16u32 + ly;
    let inside_screen = px < screen_width && py < screen_height;

    let tile_id = ty * tiles_x + tx;
    let range_start = tile_ranges[tile_id as usize * 2usize];
    let range_end = tile_ranges[tile_id as usize * 2usize + 1usize];
    let is_empty = range_start >= range_end;

    let bases_per_ch = if sh_degree == 0u32 {
        1usize
    } else if sh_degree == 1u32 {
        4usize
    } else if sh_degree == 2u32 {
        9usize
    } else {
        16usize
    };
    let sh_f16_stride = ((bases_per_ch * 3usize + 1usize) / 2usize) * 2usize;

    let mut shared_xy_x = Shared::<[f32]>::new_slice(256usize);
    let mut shared_xy_y = Shared::<[f32]>::new_slice(256usize);
    let mut shared_conic_a = Shared::<[f32]>::new_slice(256usize);
    let mut shared_conic_b = Shared::<[f32]>::new_slice(256usize);
    let mut shared_conic_c = Shared::<[f32]>::new_slice(256usize);
    let mut shared_opacity = Shared::<[f32]>::new_slice(256usize);
    let mut shared_rgb_r = Shared::<[f32]>::new_slice(256usize);
    let mut shared_rgb_g = Shared::<[f32]>::new_slice(256usize);
    let mut shared_rgb_b = Shared::<[f32]>::new_slice(256usize);
    let tile_alive = Shared::<[Atomic<u32>]>::new_slice(2usize);

    if local_id_u32 < 2u32 {
        tile_alive[local_id as usize].store(0u32);
    }
    sync_cube();

    let mut c_r = 0.0f32;
    let mut c_g = 0.0f32;
    let mut c_b = 0.0f32;
    let mut t = 1.0f32;
    let mut saturated = false;

    let total_in_tile = if is_empty {
        0u32
    } else {
        range_end - range_start
    };
    let num_batches = (total_in_tile + 255u32) / 256u32;

    let pix_center_x = px as f32 + 0.5f32;
    let pix_center_y = py as f32 + 0.5f32;

    for b in 0u32..num_batches {
        let entry_idx = range_start + b * 256u32 + local_id_u32;
        if entry_idx < range_end {
            let g = sorted_entries[entry_idx as usize * 3usize + 2usize] as usize;
            shared_xy_x[local_id as usize] = points_xy[g * 2usize];
            shared_xy_y[local_id as usize] = points_xy[g * 2usize + 1usize];
            shared_conic_a[local_id as usize] = conics[g * 3usize];
            shared_conic_b[local_id as usize] = conics[g * 3usize + 1usize];
            shared_conic_c[local_id as usize] = conics[g * 3usize + 2usize];
            shared_opacity[local_id as usize] = opacities[g];

            let pos_x = positions_3d[g * 3usize];
            let pos_y = positions_3d[g * 3usize + 1usize];
            let pos_z = positions_3d[g * 3usize + 2usize];

            let vx = pos_x - cam_pos_x;
            let vy = pos_y - cam_pos_y;
            let vz = pos_z - cam_pos_z;
            let dist_sq = vx * vx + vy * vy + vz * vz;
            let inv_len = 1.0f32 / f32::max(0.0001f32, f32::sqrt(dist_sq));
            let dir_x = vx * inv_len;
            let dir_y = vy * inv_len;
            let dir_z = vz * inv_len;

            let (rgb_r, rgb_g, rgb_b) = crate::render::kernels::project::eval_sh_native_f16(
                sh_coeffs_f16,
                g * sh_f16_stride,
                dir_x,
                dir_y,
                dir_z,
                sh_degree,
            );
            shared_rgb_r[local_id as usize] = rgb_r;
            shared_rgb_g[local_id as usize] = rgb_g;
            shared_rgb_b[local_id as usize] = rgb_b;
        } else {
            shared_opacity[local_id as usize] = 0.0f32;
        }

        sync_cube();

        let slot = (b & 1u32) as usize;
        let next_slot = ((b + 1u32) & 1u32) as usize;
        if local_id_u32 == 0u32 {
            tile_alive[next_slot].store(0u32);
        }

        let batch_count = u32::min(256u32, total_in_tile - b * 256u32);
        if inside_screen && !saturated {
            for k in 0u32..batch_count {
                let op_base = shared_opacity[k as usize];
                if op_base > 0.003921569f32 {
                    let mu_x = shared_xy_x[k as usize];
                    let mu_y = shared_xy_y[k as usize];
                    let dx = pix_center_x - mu_x;
                    let dy = pix_center_y - mu_y;
                    let a = shared_conic_a[k as usize];
                    let b_coef = shared_conic_b[k as usize];
                    let c = shared_conic_c[k as usize];

                    let mut q = -0.5f32 * (a * dx * dx + 2.0f32 * b_coef * dx * dy + c * dy * dy);
                    if q > 0.0f32 {
                        q = 0.0f32;
                    }
                    if q >= -4.0f32 {
                        let alpha = f32::min(0.99f32, op_base * f32::exp(q));
                        if alpha >= 0.003921569f32 {
                            let w = alpha * t;
                            c_r += w * shared_rgb_r[k as usize];
                            c_g += w * shared_rgb_g[k as usize];
                            c_b += w * shared_rgb_b[k as usize];
                            t *= 1.0f32 - alpha;
                            if t < 0.001f32 {
                                saturated = true;
                                t = 0.0f32;
                                break;
                            }
                        }
                    }
                }
            }
        }

        if inside_screen && !saturated {
            tile_alive[slot].store(1u32);
        }
        sync_cube();

        if tile_alive[slot].load() == 0u32 {
            break;
        }
    }

    if inside_screen {
        let final_r = f32::clamp(c_r + t * bg_r, 0.0f32, 1.0f32);
        let final_g = f32::clamp(c_g + t * bg_g, 0.0f32, 1.0f32);
        let final_b = f32::clamp(c_b + t * bg_b, 0.0f32, 1.0f32);
        let mut final_a = f32::clamp(1.0f32 - t, 0.0f32, 1.0f32);
        if is_empty {
            final_a = bg_a;
        }

        let out_base = (py * screen_width + px) as usize * 4usize;
        output_pixels[out_base] = final_r;
        output_pixels[out_base + 1usize] = final_g;
        output_pixels[out_base + 2usize] = final_b;
        output_pixels[out_base + 3usize] = final_a;
    }
}
