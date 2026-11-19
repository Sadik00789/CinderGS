# Changelog

All notable changes to **CinderGS** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-11-20

Major GPU architecture release moving Stages 1–5 completely onto persistent VRAM buffers with zero-copy swapchain presentation, hardware `f16`/`SUBGROUP`/`TIMESTAMP_QUERY` acceleration, full GPU Stage 3 prefix scan + radix sort, tight ellipse culling, and inverse-ray SH evaluation — cutting 1080p frame time from `125.35 ms` (`8.0 FPS`) in `v1.0.1` to `9.19 ms` (`108.9 FPS` zero-copy dynamic cage) / `1.42 ms` (`704.3 FPS` static cached).

### Added
- **Persistent VRAM Residency & Zero-Copy Swapchain Blit (`render/gpu_state.rs`, `render/pipeline.rs`, `render/headless.rs`)**:
  - Added `GpuSceneState` (persistent rest positions, covariances, opacities, SH coefficients, flattened multi-cage topology, and dirty-flag deformation skipping), `GpuFrameArena` (1.5× exponential buffer reuse), `GpuPipelineState` (Stage 1 + Stage 3 GPU temporal caching), and `GpuStageTimings`.
  - Added zero-copy GPU-to-GPU presentation via `copy_framebuffer_to_texture` (`padded_rgba8_row_u32` aligned to `256` bytes, `client.get_resource::<WgpuServer<AutoCompiler>>()` -> `encoder.copy_buffer_to_texture` into `Rgba8UnormSrgb`) and fast tight readback `readback_framebuffer_rgba8` / `HeadlessRenderer::render_zero_copy` / `HeadlessRenderer::render_rgba8`.
- **Full GPU Stage 3: Prefix Scan, 4-bit LSD Radix Sort & Tight Ellipse Binning (`kernels/scan.rs`, `kernels/radix_sort.rs`, `kernels/binning.rs`)**:
  - 3-level 512-element Blelloch exclusive prefix sum (`scan_block_512_kernel`, `scan_add_block_offsets_kernel`, `gpu_exclusive_scan_total`).
  - Stable 4-bit LSD radix sort (`radix_histogram_kernel`, `radix_scatter_kernel` with hardware `plane_exclusive_sum` subgroup ranking + Blelloch fallback, `gpu_radix_sort_u32_pairs`, and two-pass `gpu_radix_sort_u64_pairs` for `> 65,535` tiles).
  - GPU `count_tiles_kernel`, `emit_tile_keys_kernel`, `clear_tile_ranges_kernel`, and `identify_tile_ranges_kernel` with dynamic relative-float depth precision (`compute_packed_key_bits`, `pack_relative_depth_bits`).
- **Stage 2 Inverse-Ray SH, Polar Rotation & Tight Ellipse AABBs (`kernels/deformation.rs`, `kernels/project.rs`, `math/projection.rs`, `render/tiles/mod.rs`)**:
  - Added `compute_tet_transforms_kernel` (Jacobian, 5-step Higham Newton-Schulz polar rotation $R \in \mathrm{SO}(3)$, max stretch) and `deform_and_split_compact_kernel` (atomic split compaction).
  - Pre-evaluated view-dependent RGB (degrees 0–3) once per surviving splat inside `project_gaussians_v2_kernel` / `project_gaussians_v2_f16_kernel` using local inverse-ray direction $R^\top d$, shrinking `tile_compositor_v2_kernel` to 4 storage buffers (~16× less bandwidth) with double-buffered `plane_any` early termination.
  - Replaced square `3√λ_max` tile bounds with opacity-aware tight ellipse extents (`opacity_sigma_cutoff`, `compute_conic_and_extents`, `is_frustum_culled_xy`, `compute_gaussian_tile_bounds_xy`) and exact quadratic `ellipse_intersects_tile` culling (`1,082,262` vs `1,505,993` entries at 1080p).
- **Hardware Feature Negotiation, HUD VSync & MVC Gradients (`render/gpu.rs`, `app/gui.rs`, `app/mod.rs`, `deformation/mvc.rs`)**:
  - Added `preferred_wgpu_features`, `supports_f16()`, `supports_subgroups()`, `supports_timestamps()`, live `vsync` checkbox (`select_present_mode`), and precomputed parallel MVC `weight_gradients` (`deform_scene_cpu`).

### Changed
- **Fast Bit-Twiddling `unpack_f16` (`kernels/project.rs`, `kernels/compositor.rs`)**: Replaced software `powf` bit decoding with IEEE-754 integer bit-twiddling; added native `half::f16` kernel variants when `SHADER_F16` is enabled.
- **High-Throughput Demo Recorder (`bin/record_demo.rs`)**: Switched to `HeadlessRenderer::render_rgba8` streamed over raw stdin pipe into a single-pass dual-output `ffmpeg` filter graph (MP4 + GIF encoded simultaneously with zero PNG disk round-trips by default; `--save-frames` keeps fast PNGs). Also improved GIF palette sampling (`max_colors=256:stats_mode=full`, `bayer_scale=5`) and updated per-frame `compositor.camera_position`.

### Fixed
- Eliminated Stage 3 16-bit depth quantization shimmer/black-dot popping by dynamically allocating `(tile_id << depth_bits) | depth_q` with `near_plane` exponent anchoring (`0.03 mm` resolution at 540p, `0.12 mm` at 1080p vs `~7.8 mm` previously).
- Invalidated persistent GPU scene residency on cage rebinds (`orchestrator.invalidate_gpu_scene()`).

### Benchmarks & Verification
- **RTX 3050 Laptop GPU (`Vulkan`, 100k Gaussians @ 1920×1080, `TIMESTAMP_QUERY`)**: Stage 1 `0.066 ms`, Stage 2 `0.242 ms`, Stage 3 `10.018 ms`, Stage 4 `1.669 ms` (`11.996 ms` sum / `83.4 FPS`); Camera Orbit `8.072 ms` (`123.9 FPS`); Static Cached `1.420 ms` (`704.3 FPS`); Zero-Copy Headless Dynamic Cage `9.19 ms` (`108.9 FPS`).
- **Demo Recorder (`assets/bonsai.ply`, 1,073,930 Gaussians, 180 frames @ 960×540)**: Completed end-to-end render + simultaneous MP4/GIF encode in `3.98s` (`22.14 ms/frame` incl. encode).
- **Automated Verification**: `177` passing tests (`127` lib unit + `15` GPU/CPU parity & golden image + `35` integration deliverables, plus `5` Criterion bench smoke runs), `0` clippy warnings (`-D warnings`), clean `cargo fmt --all -- --check`, and clean `cargo build --release --locked`.

## [1.0.1] - 2026-11-15

Feature, performance, and authoring release expanding deformation fidelity, GPU culling/quantization efficiency, and standalone export workflows.

### Added
- **Strain-Driven Adaptive Splitting & Anti-Aliasing**:
  - Local Jacobian singular values via polar/symmetric decomposition (`jacobian_singular_values`, `JacobianStrain`) across CPU and GPU deformation stages (`polar.rs`, `cpu_reference.rs`, `kernels/deformation.rs`).
  - Transient major-axis Gaussian splitting (`split_gaussian_along_major_axis`) when stretch exceeds `strain_split_threshold`, eliminating tension needle artifacts without mutating canonical rest `.ply` buffers.
  - 3D Mip-Splatting isotropic covariance regularization (`mip_filter_3d_var`) and 2D EWA screen-space low-pass filtering (`filter_var_2d`) across CPU and GPU projection kernels.
- **GPU-Driven Culling, Indirect Dispatch & Quantized Buffers**:
  - Frustum, minimum opacity (`min_opacity`), and minimum screen-radius (`min_radius_px`) culling in `kernels/project.rs` and `project_scene_cull_cpu` writing `IndirectDispatchArgs` so culled splats never enter active tile binning (`bin_and_sort_active_gaussians_cpu`).
  - Compact `QuantizedGaussianSceneSoa` (`scene/soa.rs`) packing symmetric 3D covariances and degree-trimmed SH coefficients into `f16` pairs plus 32-bit octahedral rotations (`encode_octahedral_rotation` / `decode_octahedral_rotation`) for ~1.8×–4.1× memory reduction across SH degrees 3–1, paired with GPU quantized projection/compositor kernels (`gpu_bridge.rs`, `gpu_pool.rs`) and quantized scene telemetry in `wasm.rs`.
  - Tile-wide early transmittance termination ($T < \epsilon$) in `kernels/compositor.rs`.
- **Richer Cage Authoring, Constraints & Deterministic Exports**:
  - Arbitrary closed polygonal cage support via Ju–Schaefer–Warren 3D Mean Value Coordinates (`deformation/mvc.rs`: `PolyCageMesh`, manifold/winding checks, procedural polyhedra, Wavefront `.obj` parser, finite-difference local Jacobians, and `to_tet_mesh()`).
  - Persistent multi-handle pinning (`pinned_handles`, `pinned_vertices`) and half-space `CollisionPlane` obstacle constraints in `deformation/dynamics.rs` and `app/picker.rs`.
  - Binary deformed `.ply` snapshot serializer (`PlyLoader::write_binary_ply_bytes`, `PlyLoader::save_file`, `GaussianSceneSoa::to_raw_vertices`) and deterministic timeline sequence rendering (`Timeline::export_deformed_ply_at`, `Timeline::render_deterministic_frames`, `Timeline::export_deterministic_renders`) wired into the `egui` HUD and interactive viewport.

### Fixed
- Invalidated `SortCache` properly when filter/splitting parameters or cage deformations change.
- Preserved GPU view-dependent SH rotation parity across deformed cages (`prepare_gpu_sh_coeffs`) and enforced active-only tile binning for culled splats.

### Verification
- 167 automated tests passing (`cargo test --all-targets`: 125 library unit tests, 7 GPU/CPU parity & golden-image tests, 35 integration tests), zero warnings under `cargo clippy --all-targets -- -D warnings`, clean formatting via `cargo fmt --all -- --check`, and regenerated `docs/cindergs_demo.gif` verified against reference frames.

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
