use super::{evaluate_sh, CompositorUniforms};
use crate::render::tiles::{SortEntry, TileRange};
use glam::Vec3;
use rayon::prelude::*;

/// In-memory rasterized image output in both RGBA8 and 32-bit floating point formats.
#[derive(Clone, Debug, PartialEq)]
pub struct RasterizedImage {
    /// 8-bit quantized RGBA pixels (length `width * height * 4`).
    pub rgba8: Vec<u8>,

    /// 32-bit unquantized float RGBA pixels (length `width * height * 4`).
    pub rgba_f32: Vec<f32>,

    /// Image width in pixels.
    pub width: u32,

    /// Image height in pixels.
    pub height: u32,
}

impl RasterizedImage {
    /// Allocates an image buffer initialized with the given background color.
    pub fn new(width: u32, height: u32, background_color: [f32; 4]) -> Self {
        let pixel_count = (width * height) as usize;
        let mut rgba8 = vec![0u8; pixel_count * 4];
        let mut rgba_f32 = vec![0.0f32; pixel_count * 4];

        let r_u8 = (background_color[0] * 255.0).round().clamp(0.0, 255.0) as u8;
        let g_u8 = (background_color[1] * 255.0).round().clamp(0.0, 255.0) as u8;
        let b_u8 = (background_color[2] * 255.0).round().clamp(0.0, 255.0) as u8;
        let a_u8 = (background_color[3] * 255.0).round().clamp(0.0, 255.0) as u8;

        for i in 0..pixel_count {
            rgba8[i * 4] = r_u8;
            rgba8[i * 4 + 1] = g_u8;
            rgba8[i * 4 + 2] = b_u8;
            rgba8[i * 4 + 3] = a_u8;

            rgba_f32[i * 4] = background_color[0];
            rgba_f32[i * 4 + 1] = background_color[1];
            rgba_f32[i * 4 + 2] = background_color[2];
            rgba_f32[i * 4 + 3] = background_color[3];
        }

        Self {
            rgba8,
            rgba_f32,
            width,
            height,
        }
    }

    /// Fetches quantized RGBA8 pixel at `(px, py)`.
    #[inline]
    pub fn get_pixel_rgba8(&self, px: u32, py: u32) -> [u8; 4] {
        if px >= self.width || py >= self.height {
            return [0, 0, 0, 0];
        }
        let idx = ((py * self.width + px) * 4) as usize;
        [
            self.rgba8[idx],
            self.rgba8[idx + 1],
            self.rgba8[idx + 2],
            self.rgba8[idx + 3],
        ]
    }

    /// Fetches floating point RGBA pixel at `(px, py)`.
    #[inline]
    pub fn get_pixel_rgba_f32(&self, px: u32, py: u32) -> [f32; 4] {
        if px >= self.width || py >= self.height {
            return [0.0, 0.0, 0.0, 0.0];
        }
        let idx = ((py * self.width + px) * 4) as usize;
        [
            self.rgba_f32[idx],
            self.rgba_f32[idx + 1],
            self.rgba_f32[idx + 2],
            self.rgba_f32[idx + 3],
        ]
    }
}

/// Multi-threaded CPU tile rasterizer matching the GPU tile logic exactly.
#[allow(clippy::too_many_arguments)]
pub fn rasterize_scene_cpu(
    points_xy: &[f32],
    conics: &[f32],
    opacities: &[f32],
    sh_coeffs: &[f32],
    positions_3d: &[f32],
    sorted_entries: &[SortEntry],
    tile_ranges: &[TileRange],
    uniforms: &CompositorUniforms,
) -> RasterizedImage {
    let width = uniforms.screen_width;
    let height = uniforms.screen_height;
    let tiles_x = uniforms.tiles_x;
    let tiles_y = uniforms.tiles_y;
    let total_tiles = (tiles_x * tiles_y) as usize;

    let mut image = RasterizedImage::new(width, height, uniforms.background_color);
    if total_tiles == 0 || width == 0 || height == 0 {
        return image;
    }

    let cam_pos = Vec3::from_slice(&uniforms.camera_position);
    let bg = uniforms.background_color;

    // Parallelize across tiles
    let rgba8_ptr = image.rgba8.as_mut_ptr() as usize;
    let rgba_f32_ptr = image.rgba_f32.as_mut_ptr() as usize;

    (0..total_tiles).into_par_iter().for_each(|tile_id| {
        let tx = (tile_id as u32) % tiles_x;
        let ty = (tile_id as u32) / tiles_x;
        let range = if tile_id < tile_ranges.len() {
            tile_ranges[tile_id]
        } else {
            TileRange::EMPTY
        };

        let has_gaussians = !range.is_empty();

        for ly in 0..16 {
            let py = ty * 16 + ly;
            if py >= height {
                continue;
            }

            for lx in 0..16 {
                let px = tx * 16 + lx;
                if px >= width {
                    continue;
                }

                let pixel_idx = ((py * width + px) * 4) as usize;

                if !has_gaussians {
                    // Empty tile: write background
                    unsafe {
                        let p8 = (rgba8_ptr as *mut u8).add(pixel_idx);
                        let pf = (rgba_f32_ptr as *mut f32).add(pixel_idx);

                        *p8 = (bg[0] * 255.0).round().clamp(0.0, 255.0) as u8;
                        *p8.add(1) = (bg[1] * 255.0).round().clamp(0.0, 255.0) as u8;
                        *p8.add(2) = (bg[2] * 255.0).round().clamp(0.0, 255.0) as u8;
                        *p8.add(3) = (bg[3] * 255.0).round().clamp(0.0, 255.0) as u8;

                        *pf = bg[0];
                        *pf.add(1) = bg[1];
                        *pf.add(2) = bg[2];
                        *pf.add(3) = bg[3];
                    }
                    continue;
                }

                // Sample at pixel center (+0.5)
                let pixel_center_x = px as f32 + 0.5;
                let pixel_center_y = py as f32 + 0.5;

                let mut c_r = 0.0f32;
                let mut c_g = 0.0f32;
                let mut c_b = 0.0f32;
                let mut t = 1.0f32;

                for entry_idx in range.start..range.end {
                    if t < 0.001 {
                        t = 0.0;
                        break;
                    }

                    let g = sorted_entries[entry_idx as usize].gaussian_id as usize;

                    let mu_x = points_xy[g * 2];
                    let mu_y = points_xy[g * 2 + 1];

                    let dx = pixel_center_x - mu_x;
                    let dy = pixel_center_y - mu_y;

                    let a = conics[g * 3];
                    let b = conics[g * 3 + 1];
                    let c = conics[g * 3 + 2];

                    // Quadratic conic evaluation: -0.5 * (a*dx^2 + 2b*dx*dy + c*dy^2)
                    let mut q = -0.5 * (a * dx * dx + 2.0 * b * dx * dy + c * dy * dy);
                    if q > 0.0 {
                        q = 0.0;
                    }
                    if q < -4.0 {
                        continue;
                    }

                    let opacity_base = opacities[g];
                    let alpha = (opacity_base * q.exp()).min(0.99);
                    if alpha < (1.0 / 255.0) {
                        continue;
                    }

                    // Evaluate SH color along view direction
                    let pos_g = Vec3::new(
                        positions_3d[g * 3],
                        positions_3d[g * 3 + 1],
                        positions_3d[g * 3 + 2],
                    );
                    let to_cam = pos_g - cam_pos;
                    let dir = if to_cam.length_squared() > 1e-8 {
                        to_cam.normalize()
                    } else {
                        Vec3::Z
                    };

                    let sh_slice = &sh_coeffs[g * 48..(g + 1) * 48];
                    let rgb = evaluate_sh(sh_slice, dir, uniforms.sh_degree);

                    let w = alpha * t;
                    c_r += w * rgb[0];
                    c_g += w * rgb[1];
                    c_b += w * rgb[2];

                    t *= 1.0 - alpha;
                }

                // Final color composition with background
                let final_r = (c_r + t * bg[0]).clamp(0.0, 1.0);
                let final_g = (c_g + t * bg[1]).clamp(0.0, 1.0);
                let final_b = (c_b + t * bg[2]).clamp(0.0, 1.0);
                let final_a = (1.0 - t).clamp(0.0, 1.0);

                unsafe {
                    let p8 = (rgba8_ptr as *mut u8).add(pixel_idx);
                    let pf = (rgba_f32_ptr as *mut f32).add(pixel_idx);

                    *p8 = (final_r * 255.0).round().clamp(0.0, 255.0) as u8;
                    *p8.add(1) = (final_g * 255.0).round().clamp(0.0, 255.0) as u8;
                    *p8.add(2) = (final_b * 255.0).round().clamp(0.0, 255.0) as u8;
                    *p8.add(3) = (final_a * 255.0).round().clamp(0.0, 255.0) as u8;

                    *pf = final_r;
                    *pf.add(1) = final_g;
                    *pf.add(2) = final_b;
                    *pf.add(3) = final_a;
                }
            }
        }
    });

    image
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::tiles::{SortKey, TileRange};

    #[test]
    fn test_half_pixel_center_alignment() {
        // Splat exactly at (0.5, 0.5)
        let points_xy = [0.5, 0.5];
        // Conic circle: a = 1, b = 0, c = 1
        let conics = [1.0, 0.0, 1.0];
        let opacities = [1.0];
        let mut sh = [0.0f32; 48];
        sh[0] = 1.0; // Positive red DC
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

        // Pixel (0, 0) center is (0.5, 0.5) -> dx=0, dy=0 -> q=0 -> alpha = min(0.99, 1*exp(0)) = 0.99
        let p00 = img.get_pixel_rgba_f32(0, 0);
        assert!(p00[0] > 0.7, "Center pixel must receive maximum splat color");
    }

    #[test]
    fn test_early_ray_termination_saturation() {
        // 5 fully opaque overlapping splats at the same pixel
        let points_xy = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let conics = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
        let opacities = [0.99, 0.99, 0.99, 0.99, 0.99];
        let mut sh = vec![0.0f32; 48 * 5];
        // First splat is pure red (G=0, B=0)
        sh[0] = 2.0; // Red
        sh[1] = -2.0; // Green = 0
        sh[2] = -2.0; // Blue = 0
        // Rest are pure green (R=0, G=1, B=0)
        for i in 1..5 {
            sh[i * 48] = -2.0; // Red = 0
            sh[i * 48 + 1] = 2.0; // Green = 1
            sh[i * 48 + 2] = -2.0; // Blue = 0
        }
        let positions = vec![0.0f32; 15];
        let sorted_entries = (0..5).map(|i| SortEntry::new(SortKey::new(0, i as f32), i)).collect::<Vec<_>>();
        let tile_ranges = [TileRange::new(0, 5)];

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
        // After splat 0: T = 1 - 0.99 = 0.01.
        // After splat 1: T = 0.01 * 0.01 = 0.0001 < 0.001 -> saturation triggers!
        assert!(p00[0] > 0.9, "Red must dominate due to front-to-back compositing");
        assert!(p00[1] < 0.1, "Subsequent green splats should be almost completely occluded");
    }

    #[test]
    fn test_non_multiple_of_16_screen_boundaries() {
        // Screen resolution 19x19 (spans across 2x2 tiles: 4 tiles total)
        let uniforms = CompositorUniforms::new(19, 19, [0.1, 0.2, 0.3, 1.0], [0.0, 0.0, 0.0], 0);
        assert_eq!(uniforms.tiles_x, 2);
        assert_eq!(uniforms.tiles_y, 2);

        let img = rasterize_scene_cpu(&[], &[], &[], &[], &[], &[], &[], &uniforms);
        assert_eq!(img.width, 19);
        assert_eq!(img.height, 19);
        assert_eq!(img.rgba8.len(), 19 * 19 * 4);

        // Border pixel (18, 18) must have background color
        let p_edge = img.get_pixel_rgba_f32(18, 18);
        assert!((p_edge[0] - 0.1).abs() < 1e-5);
        assert!((p_edge[1] - 0.2).abs() < 1e-5);
    }
}
