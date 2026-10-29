//! Thin, testable wrappers around the CubeCL compute kernels.
//!
//! Every function takes an already-initialised [`Client`]; none of them create
//! devices or rebuild runtimes. Readback is explicit so callers can measure real
//! end-to-end latency instead of a dispatch-only lower bound.

use crate::deformation::{GaussianBinding, TetMesh};
use crate::render::ProjectionOutputs;
use crate::render::camera::CameraUniforms;
use crate::render::kernels::{
    compute_tet_jacobians_kernel, deform_gaussians_kernel, project_gaussians_kernel,
    tile_compositor_kernel,
};
use crate::render::rasterizer::{CompositorUniforms, RasterizedImage};
use crate::render::tiles::TileBinningOutputs;
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl_common::bytes::Bytes;
use thiserror::Error;

/// Errors produced while marshalling data to and from the GPU.
#[derive(Debug, Error)]
pub enum GpuBridgeError {
    /// A device-to-host copy failed.
    #[error("failed to read back {buffer} from the GPU: {message}")]
    Readback {
        /// Logical buffer name.
        buffer: &'static str,
        /// Backend error description.
        message: String,
    },

    /// The GPU returned fewer bytes than requested.
    #[error("GPU returned {actual} bytes for {buffer}, expected at least {expected}")]
    ShortReadback {
        /// Logical buffer name.
        buffer: &'static str,
        /// Required byte count.
        expected: usize,
        /// Actual byte count.
        actual: usize,
    },
}

/// Deformed Gaussian geometry produced by the GPU deformation stage.
#[derive(Clone, Debug, PartialEq)]
pub struct DeformedGeometry {
    /// Deformed world positions `[N * 3]`.
    pub positions: Vec<f32>,
    /// Deformed 3D covariances `[N * 6]`.
    pub covariances: Vec<f32>,
    /// Deformed activations `[N]`.
    pub opacities: Vec<f32>,
}

fn upload_f32(client: &Client, data: &[f32]) -> Handle {
    client.create(Bytes::from_bytes_vec(bytemuck::cast_slice(data).to_vec()))
}

fn upload_u32(client: &Client, data: &[u32]) -> Handle {
    client.create(Bytes::from_bytes_vec(bytemuck::cast_slice(data).to_vec()))
}

fn read_bytes(
    client: &Client,
    handle: Handle,
    buffer: &'static str,
    expected: usize,
) -> Result<Vec<u8>, GpuBridgeError> {
    let bytes = client
        .read_one(handle)
        .map_err(|err| GpuBridgeError::Readback {
            buffer,
            message: format!("{err:?}"),
        })?;
    if bytes.len() < expected {
        return Err(GpuBridgeError::ShortReadback {
            buffer,
            expected,
            actual: bytes.len(),
        });
    }
    Ok(bytes[..expected].to_vec())
}

fn read_f32(
    client: &Client,
    handle: Handle,
    buffer: &'static str,
    len: usize,
) -> Result<Vec<f32>, GpuBridgeError> {
    let bytes = read_bytes(client, handle, buffer, len * 4)?;
    Ok(bytemuck::cast_slice(&bytes).to_vec())
}

fn read_i32(
    client: &Client,
    handle: Handle,
    buffer: &'static str,
    len: usize,
) -> Result<Vec<i32>, GpuBridgeError> {
    let bytes = read_bytes(client, handle, buffer, len * 4)?;
    Ok(bytemuck::cast_slice(&bytes).to_vec())
}

/// Runs the tetrahedral Jacobian + per-Gaussian deformation kernels.
///
/// SH coefficients are not touched: deformation is geometry-only on the GPU.
#[allow(clippy::too_many_arguments)]
pub fn deform_geometry_gpu(
    client: &Client,
    scene_positions: &[f32],
    scene_covariances: &[f32],
    scene_opacities: &[f32],
    mesh: &TetMesh,
    bindings: &[GaussianBinding],
    cage_vertices: &[[f32; 3]],
) -> Result<DeformedGeometry, GpuBridgeError> {
    let count = scene_opacities.len();
    let num_tets = mesh.num_tets();

    if count == 0 {
        return Ok(DeformedGeometry {
            positions: Vec::new(),
            covariances: Vec::new(),
            opacities: Vec::new(),
        });
    }

    if num_tets == 0 || cage_vertices.is_empty() {
        return Ok(DeformedGeometry {
            positions: scene_positions.to_vec(),
            covariances: scene_covariances.to_vec(),
            opacities: scene_opacities.to_vec(),
        });
    }

    let flat_cage: Vec<f32> = cage_vertices
        .iter()
        .flat_map(|v| [v[0], v[1], v[2]])
        .collect();
    let flat_elems: Vec<u32> = mesh.elements.iter().flat_map(|e| e.indices).collect();
    let flat_inv_dm: Vec<f32> = mesh.precomputed.iter().flat_map(|p| p.inv_dm).collect();

    let cage_handle = upload_f32(client, &flat_cage);
    let elems_handle = upload_u32(client, &flat_elems);
    let inv_dm_handle = upload_f32(client, &flat_inv_dm);
    let jac_handle = client.empty((num_tets * 9 * 4).max(4));

    let cube_dim_tet = CubeDim::new_1d(64.min(num_tets.max(1) as u32));
    let cube_count_tet = CubeCount::Static((num_tets as u32).div_ceil(64), 1, 1);

    compute_tet_jacobians_kernel::launch(
        client,
        cube_count_tet,
        cube_dim_tet,
        unsafe { BufferArg::from_raw_parts(cage_handle.clone(), flat_cage.len()) },
        unsafe { BufferArg::from_raw_parts(elems_handle.clone(), flat_elems.len()) },
        unsafe { BufferArg::from_raw_parts(inv_dm_handle, flat_inv_dm.len()) },
        unsafe { BufferArg::from_raw_parts(jac_handle.clone(), num_tets * 9) },
        num_tets as u32,
    );

    let bind_tets: Vec<u32> = bindings.iter().map(|b| b.tet_index).collect();
    let bind_weights: Vec<f32> = bindings.iter().flat_map(|b| b.weights).collect();
    let bind_tets_h = upload_u32(client, &bind_tets);
    let bind_weights_h = upload_f32(client, &bind_weights);
    let rest_pos_h = upload_f32(client, scene_positions);
    let rest_cov_h = upload_f32(client, scene_covariances);
    let rest_opa_h = upload_f32(client, scene_opacities);

    let out_pos_h = client.empty((count * 3 * 4).max(4));
    let out_cov_h = client.empty((count * 6 * 4).max(4));
    let out_opa_h = client.empty((count * 4).max(4));

    let cube_dim_g = CubeDim::new_1d(256);
    let cube_count_g = CubeCount::Static(((count as u32).div_ceil(256)).max(1), 1, 1);

    deform_gaussians_kernel::launch(
        client,
        cube_count_g,
        cube_dim_g,
        unsafe { BufferArg::from_raw_parts(cage_handle, flat_cage.len()) },
        unsafe { BufferArg::from_raw_parts(elems_handle, flat_elems.len()) },
        unsafe { BufferArg::from_raw_parts(jac_handle, num_tets * 9) },
        unsafe { BufferArg::from_raw_parts(bind_tets_h, bind_tets.len()) },
        unsafe { BufferArg::from_raw_parts(bind_weights_h, bind_weights.len()) },
        unsafe { BufferArg::from_raw_parts(rest_pos_h, scene_positions.len()) },
        unsafe { BufferArg::from_raw_parts(rest_cov_h, scene_covariances.len()) },
        unsafe { BufferArg::from_raw_parts(rest_opa_h, scene_opacities.len()) },
        unsafe { BufferArg::from_raw_parts(out_pos_h.clone(), count * 3) },
        unsafe { BufferArg::from_raw_parts(out_cov_h.clone(), count * 6) },
        unsafe { BufferArg::from_raw_parts(out_opa_h.clone(), count) },
        count as u32,
        num_tets as u32,
    );

    Ok(DeformedGeometry {
        positions: read_f32(client, out_pos_h, "deformed positions", count * 3)?,
        covariances: read_f32(client, out_cov_h, "deformed covariances", count * 6)?,
        opacities: read_f32(client, out_opa_h, "deformed opacities", count)?,
    })
}

/// Runs the EWA projection kernel and downloads its outputs.
pub fn project_gaussians_gpu(
    client: &Client,
    positions: &[f32],
    covariances: &[f32],
    camera: &CameraUniforms,
) -> Result<ProjectionOutputs, GpuBridgeError> {
    // Derive the element count from the position buffer, not the viewport.
    let gaussians = positions.len() / 3;
    if gaussians == 0 {
        return Ok(ProjectionOutputs::empty());
    }

    let pos_handle = upload_f32(client, positions);
    let cov_handle = upload_f32(client, covariances);
    let cam_handle = upload_f32(client, camera.as_slice());

    let pts_h = client.empty((gaussians * 2 * 4).max(4));
    let depths_h = client.empty((gaussians * 4).max(4));
    let radii_h = client.empty((gaussians * 4).max(4));
    let conics_h = client.empty((gaussians * 3 * 4).max(4));

    let cube_dim = CubeDim::new_1d(256);
    let cube_count = CubeCount::Static(((gaussians as u32).div_ceil(256)).max(1), 1, 1);

    project_gaussians_kernel::launch(
        client,
        cube_count,
        cube_dim,
        unsafe { BufferArg::from_raw_parts(pos_handle, positions.len()) },
        unsafe { BufferArg::from_raw_parts(cov_handle, covariances.len()) },
        unsafe { BufferArg::from_raw_parts(cam_handle, 24) },
        unsafe { BufferArg::from_raw_parts(pts_h.clone(), gaussians * 2) },
        unsafe { BufferArg::from_raw_parts(depths_h.clone(), gaussians) },
        unsafe { BufferArg::from_raw_parts(radii_h.clone(), gaussians) },
        unsafe { BufferArg::from_raw_parts(conics_h.clone(), gaussians * 3) },
        gaussians as u32,
    );

    Ok(ProjectionOutputs {
        points_xy: read_f32(client, pts_h, "projected centers", gaussians * 2)?,
        depths: read_f32(client, depths_h, "projected depths", gaussians)?,
        radii: read_i32(client, radii_h, "projected radii", gaussians)?,
        conics: read_f32(client, conics_h, "projected conics", gaussians * 3)?,
        count: gaussians,
    })
}

/// Runs the shared-memory tile compositor and downloads the framebuffer.
#[allow(clippy::too_many_arguments)]
pub fn composite_gpu(
    client: &Client,
    projection: &ProjectionOutputs,
    opacities: &[f32],
    sh_coeffs: &[f32],
    positions: &[f32],
    binning: &TileBinningOutputs,
    uniforms: &CompositorUniforms,
) -> Result<RasterizedImage, GpuBridgeError> {
    let width = uniforms.screen_width;
    let height = uniforms.screen_height;
    let pixel_count = (width as usize) * (height as usize);
    let count = projection.count;

    if count == 0 || pixel_count == 0 || uniforms.tiles_x == 0 || uniforms.tiles_y == 0 {
        return Ok(RasterizedImage::new(
            width,
            height,
            uniforms.background_color,
        ));
    }

    let sorted_entries_flat: Vec<u32> = binning
        .sorted_entries
        .iter()
        .flat_map(|e| [e.key.tile_id, e.key.depth_bits, e.gaussian_id])
        .collect();
    let tile_ranges_flat: Vec<u32> = binning
        .tile_ranges
        .iter()
        .flat_map(|r| [r.start, r.end])
        .collect();

    let pts_h = upload_f32(client, &projection.points_xy);
    let con_h = upload_f32(client, &projection.conics);
    let opa_h = upload_f32(client, opacities);
    let sh_h = upload_f32(client, sh_coeffs);
    let pos_h = upload_f32(client, positions);
    let sorted_h = upload_u32(client, &sorted_entries_flat);
    let ranges_h = upload_u32(client, &tile_ranges_flat);
    let out_pixels_h = client.empty((pixel_count * 4 * 4).max(4));

    let cube_dim = CubeDim::new_1d(256);
    let cube_count = CubeCount::Static(uniforms.tiles_x, uniforms.tiles_y, 1);

    tile_compositor_kernel::launch(
        client,
        cube_count,
        cube_dim,
        unsafe { BufferArg::from_raw_parts(pts_h, projection.points_xy.len()) },
        unsafe { BufferArg::from_raw_parts(con_h, projection.conics.len()) },
        unsafe { BufferArg::from_raw_parts(opa_h, opacities.len()) },
        unsafe { BufferArg::from_raw_parts(sh_h, sh_coeffs.len()) },
        unsafe { BufferArg::from_raw_parts(pos_h, positions.len()) },
        unsafe { BufferArg::from_raw_parts(sorted_h, sorted_entries_flat.len().max(1)) },
        unsafe { BufferArg::from_raw_parts(ranges_h, tile_ranges_flat.len().max(1)) },
        unsafe { BufferArg::from_raw_parts(out_pixels_h.clone(), pixel_count * 4) },
        width,
        height,
        uniforms.tiles_x,
        uniforms.background_color[0],
        uniforms.background_color[1],
        uniforms.background_color[2],
        uniforms.background_color[3],
        uniforms.camera_position[0],
        uniforms.camera_position[1],
        uniforms.camera_position[2],
        uniforms.sh_degree,
    );

    let pixels = read_f32(
        client,
        out_pixels_h,
        "composited framebuffer",
        pixel_count * 4,
    )?;

    let mut image = RasterizedImage::new(width, height, uniforms.background_color);
    for (i, &value) in pixels.iter().enumerate() {
        image.rgba_f32[i] = value;
        image.rgba8[i] = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
    }

    Ok(image)
}
