//! Minimal `wasm32` entry point.
//!
//! The desktop renderer depends on `winit`/surface presentation, which is not
//! available here. This module exposes the CPU-side ingestion path so a browser
//! drop-zone can validate and inspect a PLY entirely in WebAssembly.

use wasm_bindgen::prelude::*;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Parses a PLY byte buffer and returns a short human-readable summary.
///
/// Returns a JavaScript error string when the file is not a supported PLY.
#[wasm_bindgen]
pub fn describe_ply(bytes: &[u8]) -> Result<String, JsValue> {
    let scene = crate::scene::PlyLoader::load_from_bytes(bytes)
        .map_err(|err| JsValue::from_str(&err.to_string()))?;
    let framing = crate::scene::autoframe::compute_scene_framing(&scene);
    Ok(format!(
        "{} Gaussians | hero box {:?} .. {:?} | camera distance {:.2}",
        scene.len(),
        framing.hero_min,
        framing.hero_max,
        framing.camera_distance
    ))
}

/// Parses a PLY buffer and reports quantized VRAM memory footprint (`f16` covariances/SH
/// plus octahedral-encoded rotations) relative to full `f32` SoA storage.
#[wasm_bindgen]
pub fn describe_ply_quantized(bytes: &[u8], sh_degree: u32) -> Result<String, JsValue> {
    let scene = crate::scene::PlyLoader::load_from_bytes(bytes)
        .map_err(|err| JsValue::from_str(&err.to_string()))?;
    let quantized = scene.quantize_with_degree(sh_degree);
    Ok(format!(
        "{} Gaussians | f32: {} bytes -> quantized (SH{}): {} bytes ({:.2}x smaller)",
        scene.len(),
        quantized.unquantized_bytes(),
        quantized.sh_degree,
        quantized.total_bytes(),
        quantized.compression_ratio()
    ))
}
