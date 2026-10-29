# CinderGS

Native 3D Gaussian Splatting with volumetric cage kinematics, written in pure Rust.

[![Rust Edition 2024](https://img.shields.io/badge/Rust_Edition-2024-orange.svg?style=flat-square&logo=rust)](https://www.rust-lang.org/)
[![WGPU 30](https://img.shields.io/badge/WGPU-v30-blue.svg?style=flat-square&logo=webgpu)](https://wgpu.rs/)
[![CubeCL 0.11](https://img.shields.io/badge/CubeCL-v0.11-darkgreen.svg?style=flat-square)](https://cubecl.org/)
[![Tests](https://img.shields.io/badge/Tests-142%20passing-brightgreen.svg?style=flat-square)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg?style=flat-square)](LICENSE)
[![CI](https://github.com/Sadik00789/CinderGS/actions/workflows/ci.yml/badge.svg)](https://github.com/Sadik00789/CinderGS/actions/workflows/ci.yml)

<div align="center">
  <img src="docs/cindergs_demo.gif" alt="CinderGS real-time Gaussian Splatting deformation demo" width="100%">
  <p><em>Volumetric cage kinematics on the Mip-NeRF 360 bonsai capture: the canopy ruffles while the trunk and pot stay anchored. Rendered with the CubeCL backend on an NVIDIA GeForce RTX 3050 Laptop GPU.</em></p>
</div>

CinderGS is a 3D Gaussian Splatting renderer and deformation engine written entirely
in Rust. It ingests standard 3DGS PLY captures, deforms Gaussians with tetrahedral
cages, orders them per 16×16 tile, and composites them with either a CPU reference
backend or CubeCL compute kernels sharing a single WGPU device with the viewport.

## Features

| Area | Implementation |
| :--- | :--- |
| PLY ingestion | Schema-driven header parser: `binary_little_endian` and `binary_big_endian`, `float`/`double`, arbitrary property order, SH degrees 0–3, zero-copy `mmap` fast path for the canonical 62-float layout |
| Automatic framing | 2nd/98th percentile per axis, plus explicit `--look-at` / `--distance` / `--bounds` overrides |
| Stage 1 — deformation | CPU (all cages) or GPU (one pass per cage): barycentric position mapping, covariance transport, opacity stabilization, SH rotation on the CPU path |
| Stage 2 — projection | EWA 3D-to-2D projection, conic inversion, frustum culling on CPU or GPU |
| Stage 3 — binning | 16×16 tile overlap counting, 64-bit `(tile, depth)` keys, parallel sort, tile ranges (CPU) |
| Stage 4 — compositor | 256-thread shared-memory tile compositor on CPU or GPU, world-space SH degrees 0–3, early ray termination |
| Multi-cage | Independent cages with their own meshes, bindings, spring parameters and wireframe colours |
| Sort cache | Reuses tile ordering across frames while camera and cages are static |
| Viewport | winit + egui HUD, orbit/pan/zoom, cage picking and dragging, HiDPI input |
| Timeline | Camera + cage keyframes with playback and scrubbing |
| Headless | Render farm CLI with camera-path JSON, demo recorder to PNG/MP4/GIF |
| WebAssembly | PLY ingestion entry point and browser drop-zone (`web/index.html`) |

## Frame graph

```
 PLY capture ──► schema parser ──► GaussianSceneSoa
                                        │
                                        ▼
                         ┌───────────────────────────────┐
                         │  CageScene (N cages)          │
                         │  per-cage spring simulation   │
                         └──────────────┬────────────────┘
                                        │ barycentric bindings
                                        ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ Stage 1  Deformation                                                 │
 │   x' = Σ wₖ Xₖ                                                       │
 │   Σ' = J Σ Jᵀ,   J = Dₛ Dₘ⁻¹                                        │
 │   α' = 1 − (1 − α)^(1 / clamp(det J, 0.6, 1.8))                      │
 │   SH coefficients rotated by the polar rotation R (CPU path)         │
 ├──────────────────────────────────────────────────────────────────────┤
 │ Stage 2  EWA projection + conic radius                               │
 ├──────────────────────────────────────────────────────────────────────┤
 │ Stage 3  16×16 tile binning + 64-bit (tile, depth) sort      [CPU]   │
 │   cached while camera and cages are static                           │
 ├──────────────────────────────────────────────────────────────────────┤
 │ Stage 4  Shared-memory 16×16 tile compositor                         │
 │   world-space SH evaluation, degrees 0–3                             │
 ├──────────────────────────────────────────────────────────────────────┤
 │ Stage 5  sRGB offscreen texture + fullscreen blit                    │
 ├──────────────────────────────────────────────────────────────────────┤
 │ Stage 6  Cage wireframe + handles, then egui HUD                     │
 └──────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                                WGPU swapchain
```

Stages 1, 2 and 4 run on either backend. Stage 3 executes on the CPU, which keeps
the 64-bit key construction, sorting and tile-range extraction deterministic and
lets the cache reuse them without a GPU round-trip.

### Device model

`wgpu` is pinned to 30.0.1 — the version linked by CubeCL 0.11. `GpuCompute`
registers the presentation `wgpu::Device` and `Queue` with the CubeCL runtime, so
one instance, adapter, device and queue serve both the swapchain and the compute
kernels. The compute client is created once and reused for the lifetime of the
process.

## Measured performance

Measured with `cargo run --release --bin benchmark` on:

- 100,000 synthetic Gaussians, 1920×1080
- 12 CPU threads (Rayon)
- NVIDIA GeForce RTX 3050 Laptop GPU, Vulkan backend, CubeCL 0.11

| Stage | CPU reference | CubeCL GPU (dispatch + readback) |
| :--- | ---: | ---: |
| 1 — Cage deformation | 5.31 ms | 9.48 ms |
| 2 — EWA projection | 1.14 ms | 7.20 ms |
| 3 — Tile binning + sort (CPU) | 31.12 ms | 28.91 ms |
| 4 — Tile compositor | 239.06 ms | 91.38 ms |
| **Sum of stages** | **276.63 ms** | **136.98 ms** |
| End-to-end frame (upload + blit) | 252.93 ms (4.0 FPS) | 108.36 ms (9.2 FPS) |

End-to-end is lower than the stage sum because the benchmark reuses the Stage 3
sort while the camera and cages are static. This workload produces ~1.5 million
tile entries, so the compositor dominates. `cargo bench --bench stages` measures
the individual CPU stages under Criterion.

## Build and run

Requirements:

- Rust 1.85 or newer (edition 2024)
- A Vulkan, Metal or DirectX 12 driver

```bash
# Test suite
cargo test --all-targets

# Zero-warning lint gate
cargo clippy --all-targets -- -D warnings

# Interactive viewport with the procedural scene
cargo run --release

# Interactive viewport with a capture
cargo run --release -- path/to/scene.ply

# Benchmark suite
cargo run --release --bin benchmark

# Criterion stage benchmarks
cargo bench --bench stages

# Headless render farm
cargo run --release --bin render -- \
    --input scene.ply --camera-path camera.json --out frames/

# Headless demo recorder (PNG frames, plus MP4/GIF when ffmpeg is on PATH)
cargo run --release --bin record_demo
```

### Camera path JSON

```json
{
  "width": 960,
  "height": 540,
  "fps": 30,
  "frames": 120,
  "keyframes": [
    { "time": 0.0, "eye": [0.0, 0.5, -4.0], "target": [0.0, 0.0, 0.0] },
    { "time": 4.0, "eye": [0.0, 0.5, 4.0],  "target": [0.0, 0.0, 0.0],
      "fov_y_degrees": 50.0 }
  ]
}
```

Keyframes are interpolated linearly; yaw takes the shortest path. Without
`--camera-path`, the renderer generates a one-revolution orbit around the
auto-framed subject.

## PLY schema support

The loader parses the ASCII header and supports:

- `format binary_little_endian` and `format binary_big_endian`
- `float` and `double` scalar properties
- arbitrary property order and ignored extra properties
- required properties `x y z f_dc_0..2 opacity scale_0..2 rot_0..3`
- optional `f_rest_*` with exactly 0, 3, 8 or 45 entries → SH degree 0–3
- a zero-copy `mmap` path for the canonical 62-float little-endian layout

Unsupported inputs (ASCII, list properties, truncated payloads, unknown formats,
invalid SH layouts) return descriptive `PlyError` variants.

## Automatic framing

`scene::autoframe` samples up to 200,000 positions, takes the 2nd and 98th
percentile of every axis to reject distant outliers, and derives the hero cage
and orbit distance from the trimmed box. A specific capture can be framed
explicitly from the CLI with `--look-at`, `--distance` and `--bounds`, which is
useful for room-scale captures where the subject is much smaller than the scene.

## Deformation and dynamics

- Positions: `x' = Σ wₖ Xₖ` using barycentric weights from the owning tet.
- Covariances: `Σ' = J Σ Jᵀ`, symmetrized and regularized to at least `1e-4` on
  the diagonal.
- Opacity: `α' = 1 − (1 − α)^(1 / clamp(det J, 0.6, 1.8))`.
- SH rotation: coefficients are rotated by the polar rotation `R` with exact
  monomial rotation matrices for degrees 1–3
  (`render/rasterizer/sh_rotation.rs`). Matrices are built once per tetrahedron
  and reused for every Gaussian bound to it.
- Springs: semi-implicit Euler with exponential damping
  `γ = exp(−c Δt)`, `c = −ln(γ₆₀)·60`, so trajectories are frame-rate
  independent. Timesteps are clamped to 33 ms.
- Cages are independent: each owns its mesh, bindings, stiffness, damping, mass
  and colour. Overlaps resolve first-cage-wins.

## Viewport controls

| Input | Action |
| :--- | :--- |
| Right mouse drag | Orbit |
| Middle mouse drag | Pan |
| Scroll | Zoom |
| Left click on a handle | Select a cage vertex |
| Left drag | Move the selected vertex |
| Release | Let the spring settle |

The HUD controls the execution backend, SH degree, wireframe visibility,
per-cage stiffness and damping, jiggle/reset, add/remove cage, and timeline
bake/play/scrub.

## Repository layout

```
src/math/         activations, covariance construction, EWA conic math
src/scene/        PLY schema parser, SoA storage, auto-framing, synthetic scenes
src/deformation/  tet cages, SH rotation, polar decomposition, springs, CPU deformation
src/render/       unified WGPU/CubeCL device, kernels, binning, compositor, blit
src/app/          winit event loop, egui HUD, orbit camera, picker, timeline
src/bin/          benchmark, headless render farm, demo recorder
tests/            integration, kernel parity, golden image
benches/          Criterion per-stage CPU benchmarks
web/              browser PLY drop-zone (wasm-bindgen)
```

`memmap2` and `rayon` are compiled for non-`wasm32` targets; the parallel helpers
fall back to sequential iteration on WebAssembly.

## Demo media

`docs/cindergs_demo.gif` and `docs/cindergs_demo.mp4` are recorded from the
Mip-NeRF 360 bonsai capture (1,073,930 Gaussians) with:

```bash
cargo run --release --bin record_demo -- \
    --input assets/bonsai.ply \
    --look-at 0.38,0.88,1.35 --distance 3.2 \
    --bounds -0.15,0.43,0.50,1.15,1.10,2.00
```

The cage encloses only the canopy. Its base ring is pinned, so displacement
tapers to zero at the foliage/trunk junction: the flowers ruffle while the
trunk, pot and table stay anchored.

`assets/bonsai.ply` is not committed (`*.ply` is ignored) because of its size and
the GraphDeco research license. Without the file the recorder renders a synthetic
scene. Encoding uses `ffmpeg`; without it the PNG frames remain in
`target/demo_frames/`.

Large media is tracked through Git LFS patterns in `.gitattributes`. To store the
blobs already in history with LFS, run
`git lfs migrate import --include="docs/*.gif,docs/*.mp4"`.

## Architecture notes

- Stage 3 (tile binning and sorting) runs on the CPU.
- GPU deformation updates geometry only; SH rotation is applied by the CPU
  deformation path.
- The composited frame is read back once per frame to upload the sRGB offscreen
  texture. CubeCL exposes storage handles rather than textures, so the blit
  source is a texture upload.
- The WebAssembly target covers PLY ingestion and validation; the interactive
  viewport is native.
- The renderer is a forward splat compositor with no shadow or ray-traced paths.

## License

MIT — see [LICENSE](LICENSE).
