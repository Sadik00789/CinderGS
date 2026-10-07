mod cpu_reference;
pub mod keys;

pub use cpu_reference::*;
pub use keys::*;

use bytemuck::{Pod, Zeroable};

/// Fixed screen tile dimension (16x16 pixels).
pub const TILE_SIZE: u32 = 16;

/// Fixed screen tile dimension as floating point.
pub const TILE_SIZE_F32: f32 = 16.0;

/// Uniform buffer defining screen tiling geometry and counts.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct TileConfigUniforms {
    /// Screen width in pixels.
    pub screen_width: u32,

    /// Screen height in pixels.
    pub screen_height: u32,

    /// Number of horizontal tiles: $\lceil \text{width} / 16.0 \rceil$.
    pub tiles_x: u32,

    /// Number of vertical tiles: $\lceil \text{height} / 16.0 \rceil$.
    pub tiles_y: u32,

    /// Total tile count: $T_x \times T_y$.
    pub total_tiles: u32,

    /// Number of Gaussians in the input buffer.
    pub gaussian_count: u32,
}

impl TileConfigUniforms {
    /// Constructs tile configuration for a given screen resolution and Gaussian count.
    pub fn new(screen_width: u32, screen_height: u32, gaussian_count: u32) -> Self {
        let tiles_x = screen_width.div_ceil(TILE_SIZE);
        let tiles_y = screen_height.div_ceil(TILE_SIZE);
        let total_tiles = tiles_x * tiles_y;

        Self {
            screen_width,
            screen_height,
            tiles_x,
            tiles_y,
            total_tiles,
            gaussian_count,
        }
    }

    /// Computes linear tile ID from 2D tile coordinates: $t_y \times T_x + t_x$.
    #[inline]
    pub fn tile_id(&self, tx: u32, ty: u32) -> u32 {
        ty * self.tiles_x + tx
    }

    /// Decomposes linear tile ID into 2D tile coordinates: $(t_x, t_y)$.
    #[inline]
    pub fn tile_coord(&self, tile_id: u32) -> (u32, u32) {
        if self.tiles_x == 0 {
            return (0, 0);
        }
        (tile_id % self.tiles_x, tile_id / self.tiles_x)
    }

    /// Returns pixel bounding box of a tile: `(min_x, min_y, max_x, max_y)`.
    #[inline]
    pub fn tile_pixel_bounds(&self, tx: u32, ty: u32) -> (u32, u32, u32, u32) {
        let min_x = tx * TILE_SIZE;
        let min_y = ty * TILE_SIZE;
        let max_x = ((tx + 1) * TILE_SIZE).min(self.screen_width);
        let max_y = ((ty + 1) * TILE_SIZE).min(self.screen_height);
        (min_x, min_y, max_x, max_y)
    }

    /// Returns byte view for GPU uniform buffer upload.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }

    /// Returns slice of 32-bit unsigned integers.
    #[inline]
    pub fn as_slice(&self) -> &[u32] {
        bytemuck::cast_slice(self.as_bytes())
    }
}

/// Calculates clamped tile bounding box $[tx_{\min}, tx_{\max}] \times [ty_{\min}, ty_{\max}]$ for a Gaussian.
///
/// Returns `None` if the Gaussian is culled (radius <= 0) or entirely off-screen.
#[inline]
pub fn compute_gaussian_tile_bounds(
    center_x: f32,
    center_y: f32,
    radius: f32,
    tiles_x: u32,
    tiles_y: u32,
    screen_width: u32,
    screen_height: u32,
) -> Option<(u32, u32, u32, u32)> {
    if radius <= 0.0 || tiles_x == 0 || tiles_y == 0 {
        return None;
    }

    let min_x = center_x - radius;
    let max_x = center_x + radius;
    let min_y = center_y - radius;
    let max_y = center_y + radius;

    // Viewport culling
    if max_x < 0.0
        || min_x >= screen_width as f32
        || max_y < 0.0
        || min_y >= screen_height as f32
    {
        return None;
    }

    let tx_min = ((min_x / TILE_SIZE_F32).floor() as i32).clamp(0, tiles_x as i32 - 1) as u32;
    let tx_max = ((max_x / TILE_SIZE_F32).floor() as i32).clamp(0, tiles_x as i32 - 1) as u32;
    let ty_min = ((min_y / TILE_SIZE_F32).floor() as i32).clamp(0, tiles_y as i32 - 1) as u32;
    let ty_max = ((max_y / TILE_SIZE_F32).floor() as i32).clamp(0, tiles_y as i32 - 1) as u32;

    if tx_max < tx_min || ty_max < ty_min {
        return None;
    }

    Some((tx_min, tx_max, ty_min, ty_max))
}

/// Computes the number of tiles touched by a Gaussian.
///
/// Returns 0 if culled or off-screen.
#[inline]
pub fn count_tiles_touched(
    center_x: f32,
    center_y: f32,
    radius: f32,
    tiles_x: u32,
    tiles_y: u32,
    screen_width: u32,
    screen_height: u32,
) -> u32 {
    match compute_gaussian_tile_bounds(
        center_x,
        center_y,
        radius,
        tiles_x,
        tiles_y,
        screen_width,
        screen_height,
    ) {
        Some((tx_min, tx_max, ty_min, ty_max)) => {
            (tx_max - tx_min + 1) * (ty_max - ty_min + 1)
        }
        None => 0,
    }
}

// Compile-time layout assertions
const _: () = {
    assert!(std::mem::size_of::<TileConfigUniforms>() == 24);
    assert!(std::mem::align_of::<TileConfigUniforms>() == 4);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_config_uniforms_geometry() {
        let config = TileConfigUniforms::new(800, 600, 1000);
        assert_eq!(config.screen_width, 800);
        assert_eq!(config.screen_height, 600);
        // ceil(800 / 16) = 50
        assert_eq!(config.tiles_x, 50);
        // ceil(600 / 16) = 38 (37 * 16 = 592 < 600)
        assert_eq!(config.tiles_y, 38);
        assert_eq!(config.total_tiles, 50 * 38);
        assert_eq!(config.gaussian_count, 1000);

        let (tx, ty) = config.tile_coord(53);
        assert_eq!(tx, 3);
        assert_eq!(ty, 1);
        assert_eq!(config.tile_id(3, 1), 53);
    }

    #[test]
    fn test_single_gaussian_touching_1x1_tile() {
        // Splat well inside tile (0, 0)
        let bounds = compute_gaussian_tile_bounds(8.0, 8.0, 2.0, 50, 38, 800, 600);
        assert_eq!(bounds, Some((0, 0, 0, 0)));
        assert_eq!(count_tiles_touched(8.0, 8.0, 2.0, 50, 38, 800, 600), 1);
    }

    #[test]
    fn test_straddling_gaussian_overlapping_2x2_tiles() {
        // Splat right at junction of tile (0,0), (1,0), (0,1), (1,1) at pixel (16, 16)
        let bounds = compute_gaussian_tile_bounds(16.0, 16.0, 2.0, 50, 38, 800, 600);
        assert_eq!(bounds, Some((0, 1, 0, 1)));
        assert_eq!(count_tiles_touched(16.0, 16.0, 2.0, 50, 38, 800, 600), 4);
    }

    #[test]
    fn test_culled_gaussian_radius_zero() {
        let bounds = compute_gaussian_tile_bounds(100.0, 100.0, 0.0, 50, 38, 800, 600);
        assert_eq!(bounds, None);
        assert_eq!(count_tiles_touched(100.0, 100.0, 0.0, 50, 38, 800, 600), 0);

        let bounds_neg = compute_gaussian_tile_bounds(100.0, 100.0, -5.0, 50, 38, 800, 600);
        assert_eq!(bounds_neg, None);
        assert_eq!(count_tiles_touched(100.0, 100.0, -5.0, 50, 38, 800, 600), 0);
    }

    #[test]
    fn test_culled_gaussian_offscreen() {
        // Way to the left
        assert_eq!(count_tiles_touched(-50.0, 100.0, 10.0, 50, 38, 800, 600), 0);
        // Way to the right
        assert_eq!(count_tiles_touched(900.0, 100.0, 10.0, 50, 38, 800, 600), 0);
    }
}
