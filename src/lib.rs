//! # CinderGS Engine
//!
//! Production-grade 3D Gaussian Splatting (3DGS) engine in Rust.
//!
//! - `math`: Activations, 3D covariance construction, and EWA 3D-to-2D projection.
//! - `scene`: Binary PLY layout (`RawPlyVertex`), Structure-of-Arrays layout (`GaussianSceneSoa`),
//!   and zero-copy memory-mapped PLY parser (`PlyLoader`).
//! - `render`: Camera parameters (`CameraUniforms`), projection outputs (`ProjectionOutputs`),
//!   and CubeCL compute kernels (`project_gaussians_kernel`).

pub mod app;
pub mod deformation;
pub mod math;
pub mod render;
pub mod scene;

// Re-exports
pub use app::{CagePicker, CinderApp, GuiState, OrbitCamera};
pub use deformation::*;
pub use math::*;
pub use render::*;
pub use scene::*;


