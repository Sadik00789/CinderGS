//! # CinderGS Engine
//!
//! Production-grade 3D Gaussian Splatting (3DGS) engine in Rust.
//!
//! - `math`: Activations, 3D covariance construction, and EWA 3D-to-2D projection.
//! - `scene`: Schema-driven binary PLY ingestion, Structure-of-Arrays storage, and
//!   outlier-robust automatic framing.
//! - `deformation`: Volumetric tetrahedral cages, exact SH rotation, spring dynamics,
//!   and the CPU reference deformation pipeline.
//! - `render`: Unified WGPU/CubeCL device management, compute kernels, tile binning,
//!   compositing, blitting, wireframe overlays, and headless rendering.

pub mod app;
pub mod deformation;
pub mod math;
pub mod parallel;
pub mod render;
pub mod scene;

#[cfg(target_arch = "wasm32")]
mod wasm;

// Re-exports
pub use app::{CagePicker, CinderApp, GuiState, OrbitCamera};
pub use deformation::*;
pub use math::*;
pub use render::*;
pub use scene::*;
