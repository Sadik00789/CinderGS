# CinderGS

<div align="center">

**High-Performance 3D Gaussian Splatting & Volumetric Cage Kinematics in Pure Rust**

[![CI](https://github.com/Sadik00789/CinderGS/actions/workflows/ci.yml/badge.svg)](https://github.com/Sadik00789/CinderGS/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/Sadik00789/CinderGS?style=flat-square)](https://github.com/Sadik00789/CinderGS/releases)
[![Rust Edition 2024](https://img.shields.io/badge/Rust_Edition-2024-orange.svg?style=flat-square&logo=rust)](https://www.rust-lang.org/)
[![MSRV](https://img.shields.io/badge/MSRV-1.85+-blue.svg?style=flat-square&logo=rust)](Cargo.toml)
[![WGPU 30](https://img.shields.io/badge/WGPU-v30.0-blue.svg?style=flat-square&logo=webgpu)](https://wgpu.rs/)
[![CubeCL 0.11](https://img.shields.io/badge/CubeCL-v0.11-darkgreen.svg?style=flat-square)](https://cubecl.org/)
[![Tests](https://img.shields.io/badge/Tests-167%20passing-brightgreen.svg?style=flat-square)](tests/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS%20%7C%20WASM-lightgrey.svg?style=flat-square)](Cargo.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg?style=flat-square)](LICENSE)

<br />

[Overview](#-overview) •
[Key Features](#-key-features) •
[Architecture](#-architecture--pipeline) •
[Performance](#-measured-performance) •
[Quick Start](#-quick-start) •
[Production Tooling](#-production-tooling) •
[Controls](#-viewport-controls) •
[Layout](#-repository-layout)

---

<img src="docs/cindergs_demo.gif" alt="CinderGS real-time Gaussian Splatting deformation demo" width="100%">

<p><em>Real-time volumetric cage kinematics on the Mip-NeRF 360 bonsai capture (1,073,930 Gaussians). The canopy flexes and settles via damped springs while the trunk and base remain anchored. Rendered with the CubeCL backend on an NVIDIA GeForce RTX 3050 Laptop GPU.</em></p>

</div>

---

## 🔭 Overview

**CinderGS** is a production-grade 3D Gaussian Splatting (3DGS) rasterization and physics deformation engine written entirely in Rust. It eliminates complex C++/CUDA toolchain dependencies in favor of a modern, multi-backend compute architecture powered by **[CubeCL](https://cubecl.org/)** and **[WGPU](https://wgpu.rs/)**, while introducing physically grounded **tetrahedral cage kinematics** for real-time volumetric manipulation of Gaussian scenes.

Whether ingesting multi-million Gaussian captures from raw PLY files, deforming complex organic structures with volume preservation and spherical harmonics rotation, or rendering high-resolution frames headlessly across camera trajectories, CinderGS provides an uncompromising balance of mathematical precision, memory safety, and high-throughput rendering.

---

## ✨ Key Features

| Domain | Capability Highlights |
| :--- | :--- |
| **GPU Compute & Rasterization** | • Native compute kernels via **CubeCL 0.11** targeting Vulkan, Metal, and DirectX 12.<br>• Exact **CPU reference backend** with multi-threaded Rayon execution.<br>• **Unified device model**: single `wgpu::Device` and `Queue` shared across compute kernels and swapchain surface.<br>• GPU-driven frustum, opacity, and screen-radius culling writing `IndirectDispatchArgs` so culled splats skip tile binning.<br>• Compact `QuantizedGaussianSceneSoa` (`f16` covariance & SH packing + 32-bit octahedral rotations) cutting scene footprint up to ~1.8×–4.1× across SH degrees 3–1.<br>• 256-thread shared-memory 16×16 tile compositor with **tile-wide early transmittance termination** ($`T < \epsilon`$) and 3D Mip-Splatting / 2D EWA anti-aliasing filter.<br>• Full world-space Spherical Harmonics (SH) evaluation across degrees 0 to 3. |
| **Volumetric Cage Kinematics** | • Volumetric tetrahedral cages AND **arbitrary closed low-poly cage meshes via 3D Mean Value Coordinates (MVC)** (`PolyCageMesh`, Wavefront `.obj` parser, winding-number containment).<br>• **Strain-driven transient adaptive splitting**: computes local Jacobian singular values and splits over-stretched splats along their major axis without mutating canonical rest-state `.ply` data.<br>• Closed-form covariance transport `Σ' = J Σ Jᵀ` with 3D Mip-Splatting and diagonal regularization.<br>• Optical density stabilization: `α' = 1 − (1 − α)^(1 / clamp(det J, 0.6, 1.8))`.<br>• Exact monomial SH rotation matrices for bands 1–3 via polar decomposition `J = R S`.<br>• Multi-cage support with individual stiffness, damping, rest configurations, and wireframes. |
| **Physics & Dynamics** | • Semi-implicit Euler spring solver with frame-rate independent exponential damping (`γ = exp(−c Δt)`).<br>• **Persistent multi-handle vertex pinning** (`pinned_handles`) and **half-space collision planes** (`CollisionPlane`) in `dynamics.rs` and `picker.rs`.<br>• Real-time screen-space cage vertex picking and camera-plane dragging.<br>• Automated jiggle excitation and settling simulations. |
| **Sorting & Caching** | • Deterministic 64-bit `(tile_id, depth_key)` parallel active-splat sorting.<br>• **Sort Cache Optimization**: Reuses tile ordering across frames when camera and cages are static, avoiding redundant passes during inspection. |
| **PLY Ingestion & Export** | • Robust schema-driven header parser and **binary `.ply` serializer** (`PlyLoader::save_file` for deformed snapshots).<br>• Handles `binary_little_endian` / `binary_big_endian`, `float` and `double` scalars, arbitrary property orderings, and custom attributes.<br>• Spherical harmonics degrees 0–3 (9/24/45 `f_rest` coefficients) repacked cleanly between canonical `.ply` and internal SoA layouts.<br>• High-throughput zero-copy `mmap` fast path for canonical 62-float buffers.<br>• Rich error diagnostics through dedicated `PlyError` variants. |
| **Intelligent Auto-Framing** | • Statistical outlier rejection via 2nd/98th percentile bounding box trimming.<br>• Automatic derivation of hero cage dimensions, orbit pivot, and focal distance.<br>• Explicit CLI override flags (`--look-at`, `--distance`, `--bounds`) for room-scale captures. |
| **Production Tooling** | • **Headless Render Farm** (`render`): Scriptable offline renderer reading JSON camera paths.<br>• **Autonomous Demo Recorder** (`record_demo`): Frame generator with automatic MP4/GIF encoding.<br>• **Interactive Timeline & Deterministic Export**: Keyframed camera trajectories and cage poses with live scrubbing, deformed `.ply` snapshots, and deterministic image sequence exports.<br>• **WebAssembly Drop-Zone**: In-browser PLY validation and inspection (including quantized memory footprint telemetry) with zero network transmission. |

---

## 🏗️ Architecture & Pipeline

CinderGS processes scenes through a staged pipeline designed to maximize throughput while preserving deterministic ordering and state caching:

```text
 PLY Capture ──► Schema Parser ──► GaussianSceneSoa
                                          │
                                          ▼
                           ┌─────────────────────────────────────────┐
                           │   CageScene (Tet & MVC Poly Cages)      │
                           │   Spring Solver, Pins & Collision Planes│
                           └────────────────────┬────────────────────┘
                                                │ Deformed Vertices
                                                ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │ Stage 1: Cage Deformation & Transient Strain Splitting        [CPU/GPU]│
   │   • Vertex mapping:       x' = Σ wₖ Xₖ  (Tet barycentric or 3D MVC)    │
   │   • Covariance transport: Σ' = J Σ Jᵀ + Mip₃D I,  J = Dₛ Dₘ⁻¹          │
   │   • Strain splitting:     σ_max(J) > τ ⇒ transient major-axis child    │
   │   • Opacity & SH rot:     α' stabilization + polar rotation J = R S    │
   ├────────────────────────────────────────────────────────────────────────┤
   │ Stage 2: EWA Projection, Culling & Quantization              [CPU/GPU]│
   │   • Frustum, opacity, and min-radius culling ⇒ IndirectDispatchArgs    │
   │   • 3D-to-2D covariance projection: Σ₂D = J_cam Σ' J_camᵀ + filter₂D I │
   │   • Optional f16 + octahedral rotation scene packing                   │
   ├────────────────────────────────────────────────────────────────────────┤
   │ Stage 3: Active Tile Binning & 64-bit Sort                        [CPU]│
   │   • Overlap counting over surviving active splats only                 │
   │   • Parallel sort on composite keys: (tile_id << 32) | depth           │
   │   • Sort Cache: bypassed when camera, cages, and filter settings hold  │
   ├────────────────────────────────────────────────────────────────────────┤
   │ Stage 4: Tile Compositor                                      [CPU/GPU]│
   │   • 256-thread shared-memory workgroups (16×16 pixels/tile)            │
   │   • World-space Spherical Harmonics evaluation (degrees 0–3 / f16)     │
   │   • Front-to-back blending with tile-wide early T < ε termination      │
   ├────────────────────────────────────────────────────────────────────────┤
   │ Stage 5: Offscreen Blit Pass                                     [WGPU]│
   │   • sRGB transfer, gamma correction, and fullscreen presentation blit  │
   ├────────────────────────────────────────────────────────────────────────┤
   │ Stage 6: Overlays & Telemetry                                    [WGPU]│
   │   • Wireframe cage rendering, vertex handles, and immediate-mode HUD   │
   └────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                                   WGPU Swapchain
```

### Mathematical Foundations

#### 1. Barycentric Embedding
Each Gaussian center $`\mathbf{x} \in \mathbb{R}^3`$ inside a cage tetrahedron with rest vertices $`\mathbf{X}_k`$ is mapped to deformed coordinates $`\mathbf{x}'`$ via precomputed weights $`w_k`$:

```math
\mathbf{x}' = \sum_{k=1}^4 w_k \mathbf{X}'_k, \quad \sum_{k=1}^4 w_k = 1
```

#### 2. Covariance Transport
With deformed edge matrix $`\mathbf{D}_s`$ and rest reference edge matrix $`\mathbf{D}_m`$, the Jacobi deformation gradient is $`\mathbf{J} = \mathbf{D}_s \mathbf{D}_m^{-1}`$. The 3D Gaussian covariance matrix $`\mathbf{\Sigma}`$ transports as:

```math
\mathbf{\Sigma}' = \mathbf{J} \mathbf{\Sigma} \mathbf{J}^\top
```

The resulting tensor is symmetrized and regularized with a minimum diagonal floor of $`10^{-4}`$ to avoid degenerate splats.

#### 3. Opacity Stabilization
When a tetrahedron expands or compresses, the apparent volumetric density changes. CinderGS preserves visual optical thickness by scaling opacity $`\alpha`$:

```math
\alpha' = 1 - (1 - \alpha)^{\frac{1}{\mathrm{clamp}(\det \mathbf{J},\, 0.6,\, 1.8)}}
```

#### 4. Spherical Harmonics Rotation
Monomial rotation representations for degree 1 (3×3), degree 2 (5×5), and degree 3 (7×7) are evaluated using the polar rotation factor $`\mathbf{R}`$ from $`\mathbf{J} = \mathbf{R} \mathbf{S}`$, ensuring view-dependent color stays consistent under large rotational deformations.

#### 5. Strain-Driven Adaptive Splitting & Anti-Aliasing
Under strong tensile deformation ($`\sigma_{\max}(\mathbf{J}) > \tau`$), a splat's deformed 3D covariance $`\mathbf{\Sigma}'`$ is factored via eigen-decomposition along its major axis $`\mathbf{u}_{\max}`$ and transiently split into two overlapping child Gaussians without modifying the rest-state `.ply`. Isotropic 3D Mip-Splatting and 2D EWA low-pass covariance filters ($`\mathbf{\Sigma}' + v_{\mathrm{mip}} \mathbf{I}`$, $`\mathbf{\Sigma}_{2\mathrm{D}} + v_{2\mathrm{D}} \mathbf{I}`$) prevent sub-pixel compression shimmer.

#### 6. 3D Mean Value Coordinates (Closed Polygonal Cages)
In addition to tetrahedral cages, closed triangular cages (`PolyCageMesh`, loadable via Wavefront `.obj` or procedural polyhedra) evaluate smooth harmonic-like 3D Mean Value Coordinates (Ju–Schaefer–Warren) with finite-difference local Jacobians $`\mathbf{J}(\mathbf{x})`$ and optional tetrahedralization (`to_tet_mesh()`) for full GPU deformation parity.

### Unified Device Model

`wgpu` is pinned to `30.0.1`, matching the runtime bundled with `cubecl 0.11`. `GpuCompute` registers the presentation `wgpu::Device` and `wgpu::Queue` directly with CubeCL's compute client. As a result:
- A single hardware instance, adapter, device, and queue serve both compute dispatches and graphics presentation.
- Memory round-trips and context synchronization bottlenecks are eliminated.
- Compute clients are initialized once and persist for the application lifetime.

---

## ⚡ Measured Performance

Benchmarked via `cargo run --release --bin benchmark` under identical workload conditions:

- **Workload**: 100,000 synthetic Gaussians at 1920×1080 resolution (~1.5 million generated tile entries)
- **CPU**: 12 execution threads (Rayon work-stealing pool)
- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (Vulkan backend, CubeCL 0.11)

| Pipeline Stage | CPU Reference | CubeCL GPU (Dispatch + Readback) | Speedup |
| :--- | :---: | :---: | :---: |
| **Stage 1 — Cage Deformation** | 5.31 ms | 9.48 ms | Reference faster on small mesh |
| **Stage 2 — EWA Projection** | 1.14 ms | 7.20 ms | Reference faster (no dispatch overhead) |
| **Stage 3 — Tile Binning & Sort (CPU)** | 31.12 ms | 28.91 ms | Deterministic CPU sort |
| **Stage 4 — Tile Compositor** | 239.06 ms | 91.38 ms | **2.62× GPU speedup** |
| **Sum of Stages** | **276.63 ms** | **136.98 ms** | **2.02× aggregate speedup** |
| **End-to-End Frame (Static View / Cached Sort)** | **252.93 ms** (4.0 FPS) | **108.36 ms** (9.2 FPS) | **2.33× total throughput** |

> [!NOTE]
> During static camera views or minor cage movements, the **Sort Cache** reuses Stage 3 binning structures, reducing end-to-end frame time below the raw stage summation. Micro-benchmarks for individual CPU stages can be verified using `cargo bench --bench stages`.

---

## 🚀 Quick Start

### Prerequisites

- **Rust**: Version 1.85 or newer (Edition 2024 supported).
- **Graphics Driver**: Vulkan (Linux/Windows), Metal (macOS), or DirectX 12 (Windows).

### Building & Running

```bash
# Clone repository
git clone https://github.com/Sadik00789/CinderGS.git
cd CinderGS

# Run unit and integration tests (167 tests)
cargo test --all-targets

# Strict zero-warning lint check
cargo clippy --all-targets -- -D warnings

# Launch interactive viewport with procedural Fibonacci scene (2,000 splats)
cargo run --release

# Launch interactive viewport with a custom 3DGS capture
cargo run --release -- path/to/capture.ply
```

---

## 🛠️ Production Tooling

### 1. Headless Render Farm (`render`)

Render publication-quality frames without opening an interactive window or requiring a window compositor:

```bash
cargo run --release --bin render -- \
    --input assets/bonsai.ply \
    --camera-path camera_path.json \
    --backend cubecl \
    --out output_frames/
```

#### Camera Path Specification (`camera_path.json`)

```json
{
  "width": 1920,
  "height": 1080,
  "fps": 30,
  "frames": 120,
  "keyframes": [
    {
      "time": 0.0,
      "eye": [0.0, 0.5, -4.0],
      "target": [0.0, 0.0, 0.0],
      "fov_y_degrees": 50.0
    },
    {
      "time": 4.0,
      "eye": [0.0, 0.5, 4.0],
      "target": [0.0, 0.0, 0.0],
      "fov_y_degrees": 50.0
    }
  ]
}
```

*Keyframes undergo spherical interpolation for yaw and linear interpolation for target and camera positions. If `--camera-path` is omitted, the renderer generates a smooth 360° orbit around the automatically framed target.*

---

### 2. Autonomous Demo Recorder (`record_demo`)

Generates video artifacts demonstrating orbit paths and cage physics release:

```bash
cargo run --release --bin record_demo -- \
    --input assets/bonsai.ply \
    --look-at 0.38,0.88,1.35 \
    --distance 3.2 \
    --bounds -0.15,0.43,0.50,1.15,1.10,2.00
```

- When `ffmpeg` is present on `PATH`, the recorder outputs both `docs/cindergs_demo.mp4` and `docs/cindergs_demo.gif`.
- If `ffmpeg` is unavailable, uncompressed PNG frames are preserved in `target/demo_frames/`.

---

### 3. WebAssembly Inspection Portal (`web/index.html`)

CinderGS compiles to `wasm32-unknown-unknown` for client-side PLY parsing and validation.

```bash
# Serve the web portal locally
cd web
python3 -m http.server 8080
```

Drop any 3DGS `.ply` file onto the browser window to inspect Gaussian counts, schema headers, bounding boxes, and framing recommendations instantly without server uploads.

---

## 🎮 Viewport Controls

| Input Interaction | Action Performed |
| :--- | :--- |
| **Right Mouse Button + Drag** | Orbit camera around pivot target |
| **Middle Mouse Button + Drag** | Pan camera plane (sensor shift) |
| **Scroll Wheel** | Proportional distance zoom |
| **Left Mouse Click (on Handle)** | Select a cage vertex |
| **Left Mouse Button + Drag** | Translate selected vertex in camera-aligned eye plane |
| **Release Drag** | Release vertex into semi-implicit spring settling simulation |

### HUD Control Panel (`egui`)

- **Backend Toggle**: Switch live between `CubeCL GPU` and `CPU Reference`.
- **Controls & Appearance**: Dynamically clamp `SH Degree` (0–3), adjust transient `Split Threshold` and `3D Mip Var` anti-aliasing, or toggle `Show Cage Wireframe`.
- **Volumetric Cages**: Select active cages, adjust spring stiffness ($`k`$) and exponential damping ($`\gamma`$), pin/unpin vertex handles (`Pin / Unpin Handle`, `Clear Pins`), toggle `Ground Plane Y` collision constraints, trigger `Jiggle Impulse`, or add/remove/reset cages.
- **Keyframe Timeline & Export**: `Bake Keyframe`, scrub playhead, play animations, write deformed 3DGS captures (`Export Deformed PLY`), and export deterministic image sequences (`Export Renders`).

---

## 📂 Repository Layout

```text
CinderGS/
├── assets/                 # Example PLY captures (ignored in VCS)
├── benches/
│   └── stages.rs           # Criterion microbenchmarks for CPU stages
├── docs/
│   ├── cindergs_demo.gif   # Visual demonstration artifact
│   └── cindergs_demo.mp4   # Encoded MP4 demonstration
├── src/
│   ├── app/                # Application layer: winit loop, orbit camera, picker, timeline, egui HUD
│   │   ├── camera.rs       # Orbit camera controller with gimbal-lock protection
│   │   ├── gui.rs          # Immediate-mode telemetry overlay
│   │   ├── picker.rs       # Screen-space vertex picking and eye-plane dragging
│   │   └── timeline.rs     # Keyframe interpolation and animation playback
│   ├── bin/                # Executable CLI targets
│   │   ├── benchmark.rs    # Dual-backend throughput benchmark suite
│   │   ├── record_demo.rs  # Automated demo capture and encoding binary
│   │   └── render.rs       # Headless batch render farm binary
│   ├── deformation/        # Physics engine & volumetric kinematics
│   │   ├── cpu_reference.rs# Reference deformation & strain-splitting pipeline
│   │   ├── dynamics.rs     # Frame-rate independent spring solver, pins & collision planes
│   │   ├── mod.rs          # Cage meshes, barycentric bindings & multi-cage scene
│   │   ├── mvc.rs          # Closed polygonal cages via 3D Mean Value Coordinates (.obj)
│   │   └── polar.rs        # Polar decomposition, singular values & SH rotation matrices
│   ├── math/               # Mathematical primitives: activations, covariances, splat splitting, EWA
│   ├── parallel.rs         # Abstract parallel iterators (Rayon / Sequential fallback)
│   ├── render/             # Unified graphics & compute subsystem
│   │   ├── blit.rs         # Fullscreen sRGB presentation blit pass
│   │   ├── camera.rs       # Camera uniforms and view-projection math
│   │   ├── camera_path.rs  # JSON camera trajectory interpolation for batch rendering
│   │   ├── gpu.rs          # WGPU device, queue, and swapchain initialization
│   │   ├── gpu_bridge.rs   # CubeCL-to-WGPU buffer synchronization & IndirectDispatchArgs
│   │   ├── gpu_pool.rs     # Persistent GPU scratch & quantized scene buffer pools
│   │   ├── headless.rs     # Offscreen WGPU/CubeCL batch renderer
│   │   ├── kernels/        # CubeCL GPU compute kernels (deformation, culling, compositor)
│   │   ├── rasterizer/     # 16x16 tile compositor, SH math, rotation, and sort caching
│   │   ├── tiles/          # Active-splat tile binning and 64-bit depth/tile sorting
│   │   └── wireframe.rs    # Dynamic cage edge and vertex handle rendering
│   ├── scene/              # Data ingestion, quantization & analysis
│   │   ├── autoframe.rs    # Percentile-based bounding box & camera framing
│   │   ├── ply_loader.rs   # Schema-driven binary PLY reader & deformed snapshot writer
│   │   ├── raw_vertex.rs   # Canonical 62-float PLY record layout
│   │   ├── soa.rs          # Structure-of-Arrays & f16/octahedral quantized GPU storage
│   │   └── synthetic.rs    # Procedural Fibonacci sphere scene generator
│   ├── lib.rs              # Engine library root and public API
│   ├── main.rs             # Desktop application entry point
│   └── wasm.rs             # WebAssembly entry point for browser inspection
├── tests/                  # Integration, GPU-CPU parity, and golden image tests
├── web/
│   └── index.html          # Browser drag-and-drop PLY inspector
├── Cargo.toml              # Rust dependency definitions and profiles
├── CHANGELOG.md            # Version history and release notes
└── LICENSE                 # MIT License
```

---

## 📚 References & Acknowledgments

- **3D Gaussian Splatting**: Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis. *3D Gaussian Splatting for Real-Time Radiance Field Rendering*. ACM Transactions on Graphics (SIGGRAPH 2023).
- **CubeCL Project**: High-performance, multi-backend compute kernel framework in pure Rust.
- **WGPU Ecosystem**: Portable graphics standard implementation in Rust.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for complete details.
