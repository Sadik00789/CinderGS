# Changelog

All notable changes to **CinderGS** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-11-14

First stable major release of **CinderGS**, a pure-Rust 3D Gaussian Splatting (3DGS) rasterization and volumetric cage kinematics engine.

### Added
- **Unified GPU & CPU Rasterization Pipeline**:
  - Shared `wgpu` 30.0.1 device and queue model registered once with `cubecl` 0.11.0 (`GpuCompute`).
  - CubeCL GPU compute kernels for Stage 1 cage deformation, Stage 2 EWA projection & culling, and Stage 4 16×16 shared-memory tile compositing with full world-space Spherical Harmonics (degrees 0–3) and early ray termination (`alpha > 0.9999`).
  - Multi-threaded Rayon CPU reference backend for exact parity verification and fallback execution.
  - Stage 3 deterministic 64-bit `(tile_id, depth_key)` tile binning and sorting with static-frame sort caching (`Sort Cache`).
  - Stage 5 offscreen sRGB blit pass and Stage 6 wireframe cage overlay + `egui` telemetry HUD.
- **Volumetric Tetrahedral Cage Kinematics**:
  - Multi-cage `CageScene` with barycentric embedding, per-cage stiffness/damping, and dynamic addition/removal.
  - Closed-form covariance transport ($\Sigma' = J \Sigma J^\top$), optical density stabilization, and polar decomposition ($J = R S$) with exact SH rotation matrices for degrees 1–3.
  - Frame-rate independent semi-implicit Euler spring dynamics (`dynamics.rs`), screen-space vertex picking (`picker.rs`), and camera-plane dragging.
- **Schema-Driven PLY Ingestion & Auto-Framing**:
  - Binary little- and big-endian PLY parser (`PlyLoader`) supporting `float`/`double` scalars, arbitrary property ordering, compact SH degree 0–3 (`f_rest` 0/9/24/45) repacking, zero-copy `mmap` fast path, and `PlyError` diagnostics.
  - Outlier-robust 2nd/98th percentile scene auto-framing (`compute_scene_framing`) and procedural Fibonacci scene generation (`fibonacci_cluster`).
- **Production CLI Binaries & WebAssembly Tooling**:
  - `cinder_gs`: Interactive desktop viewport with orbit/pan/zoom camera, live CPU/GPU backend switching, SH degree clamping, and keyframed timeline playback (`Timeline`).
  - `render`: Headless batch renderer supporting JSON `CameraPath` trajectories or automatic 360° orbits and PNG frame output.
  - `record_demo`: Autonomous demo recorder generating PNG frames and encoding `MP4`/`GIF` outputs when `ffmpeg` is available.
  - `benchmark`: Self-measuring CPU vs. CubeCL GPU stage and end-to-end throughput benchmark suite.
  - `wasm32` PLY inspection module (`wasm.rs`) paired with the zero-upload browser drop-zone (`web/index.html`).
- **Verification & Release Automation**:
  - 143 automated tests covering unit tests, integration deliverables, GPU/CPU kernel parity, and deterministic golden-image regression, plus Criterion CPU stage microbenchmarks (`benches/stages.rs`).
  - Cross-platform GitHub Actions CI (`ci.yml`) and tagged release packaging workflow (`release.yml`) for Linux, macOS, and Windows.
