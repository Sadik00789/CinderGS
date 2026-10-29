//! CubeCL compute kernels.
//!
//! Stage 3 (tile binning and depth ordering) intentionally runs on the CPU: the
//! kernel prototypes that once lived here required a device-side prefix scan and
//! radix sort that did not exist, so they were removed rather than left as dead
//! code. See `README.md` for the current frame graph.

pub mod compositor;
pub mod deformation;
pub mod project;

pub use compositor::*;
pub use deformation::*;
pub use project::*;
