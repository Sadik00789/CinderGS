# CinderGS

<div align="center">

**High-Performance 3D Gaussian Splatting & Volumetric Cage Kinematics in Pure Rust**

[![CI](https://github.com/Sadik00789/CinderGS/actions/workflows/ci.yml/badge.svg)](https://github.com/Sadik00789/CinderGS/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/Sadik00789/CinderGS?style=flat-square)](https://github.com/Sadik00789/CinderGS/releases)
[![Rust Edition 2024](https://img.shields.io/badge/Rust_Edition-2024-orange.svg?style=flat-square&logo=rust)](https://www.rust-lang.org/)
[![MSRV](https://img.shields.io/badge/MSRV-1.85+-blue.svg?style=flat-square&logo=rust)](Cargo.toml)
[![WGPU 30](https://img.shields.io/badge/WGPU-v30.0-blue.svg?style=flat-square&logo=webgpu)](https://wgpu.rs/)
[![CubeCL 0.11](https://img.shields.io/badge/CubeCL-v0.11-darkgreen.svg?style=flat-square)](https://cubecl.org/)
[![Tests](https://img.shields.io/badge/Tests-177%20passing-brightgreen.svg?style=flat-square)](tests/)
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
| **GPU Compute & Rasterization** | • **100% GPU-resident persistent pipeline** (`GpuSceneState` + `GpuFrameArena`) via **CubeCL 0.11** targeting Vulkan, Metal, and DirectX 12 — eliminating per-frame re-uploads and PCIe framebuffer readbacks.<br>• **Unified shared device & hardware feature negotiation**: requests `SHADER_F16`, `SUBGROUP`, and `TIMESTAMP_QUERY` when supported on a single shared `wgpu::Device` and `Queue`.<br>• **Zero-copy compute-to-texture blit**: packs `RGBA8` rows aligned to `256` bytes and copies directly GPU-to-GPU via `copy_buffer_to_texture` into `Rgba8UnormSrgb` swapchain surfaces.<br>• **Pre-evaluated inverse-ray SH**: Stage 2 transforms view directions by $`\mathbf{R}^\top \mathbf{d}`$ and evaluates degrees 0–3 once per surviving splat, cutting Stage 4 compositor bandwidth by ~16×.<br>• Cooperative 256-thread shared-memory 16×16 tile compositor with double-buffered `plane_any` subgroup early transmittance termination ($`T < \epsilon`$) and 3D/2D Mip-Splatting EWA filtering.<br>• Exact multi-threaded Rayon **CPU reference backend** and `QuantizedGaussianSceneSoa` (`f16` / octahedral) support. |
| **Volumetric Cage Kinematics** | • Volumetric tetrahedral cages AND **arbitrary closed low-poly cage meshes via 3D Mean Value Coordinates (MVC)** (`PolyCageMesh`, Wavefront `.obj` parser, precomputed `weight_gradients`, parallel `deform_scene_cpu`).<br>• **GPU polar decomposition & atomic split compaction**: computes per-tet Jacobians, 5-step Higham Newton-Schulz polar rotation $`\mathbf{R} \in \mathrm{SO}(3)`$, and atomically compacts strain-split children along their major axis without mutating canonical `.ply` data.<br>• Closed-form covariance transport `Σ' = J Σ Jᵀ` with 3D Mip-Splatting and diagonal regularization.<br>• Optical density stabilization: `α' = 1 − (1 − α)^(1 / clamp(det J, 0.6, 1.8))`.<br>• Multi-cage support with per-cage stiffness, damping, rest configurations, and dirty-flag GPU skip when stationary. |
| **Physics & Dynamics** | • Semi-implicit Euler spring solver with frame-rate independent exponential damping (`γ = exp(−c Δt)`).<br>• **Persistent multi-handle vertex pinning** (`pinned_handles`) and **half-space collision planes** (`CollisionPlane`) in `dynamics.rs` and `picker.rs`.<br>• Real-time screen-space cage vertex picking and camera-plane dragging.<br>• Automated jiggle excitation and settling simulations. |
| **Sorting & Caching** | • **Full GPU Stage 3**: opacity-aware tight ellipse AABBs + exact quadratic ellipse-vs-tile intersection, 3-level Blelloch exclusive prefix scan (`scan.rs`), and 4-bit LSD radix sort (`radix_sort.rs`) with dynamic relative-float depth precision (and 64-bit fallback for `> 65,535` tiles).<br>• **Temporal Persistence**: dirty-flag Stage 1 skip during camera orbits plus GPU Stage 3 sort/range cache reuse during static views. |
| **PLY Ingestion & Export** | • Robust schema-driven header parser and **binary `.ply` serializer** (`PlyLoader::save_file` for deformed snapshots).<br>• Handles `binary_little_endian` / `binary_big_endian`, `float` and `double` scalars, arbitrary property orderings, and custom attributes.<br>• Spherical harmonics degrees 0–3 (9/24/45 `f_rest` coefficients) repacked cleanly between canonical `.ply` and internal SoA layouts.<br>• High-throughput zero-copy `mmap` fast path for canonical 62-float buffers.<br>• Rich error diagnostics through dedicated `PlyError` variants. |
| **Intelligent Auto-Framing** | • Statistical outlier rejection via 2nd/98th percentile bounding box trimming.<br>• Automatic derivation of hero cage dimensions, orbit pivot, and focal distance.<br>• Explicit CLI override flags (`--look-at`, `--distance`, `--bounds`) for room-scale captures. |
| **Production Tooling** | • **Headless Render Farm** (`render`): Scriptable offline renderer reading JSON camera paths.<br>• **Autonomous Demo Recorder** (`record_demo`): Streams raw `RGBA8` frames over a pipe into a single-pass dual-output `ffmpeg` encoder for fast simultaneous MP4 + GIF generation (plus optional `--save-frames`).<br>• **Interactive Timeline & Deterministic Export**: Keyframed camera trajectories and cage poses with live scrubbing, deformed `.ply` snapshots, and deterministic image sequence exports.<br>• **WebAssembly Drop-Zone**: In-browser PLY validation and inspection (including quantized memory footprint telemetry) with zero network transmission. |

---

## 🏗️ Architecture & Pipeline

CinderGS processes scenes through a **persistent GPU-resident pipeline** (`GpuSceneState` + `GpuFrameArena`) that uploads rest geometry and cage topology once on load/rebind and keeps Stages 1–5 resident in VRAM with zero per-frame host readbacks during interactive rendering:

```text
 PLY Capture ──► Schema Parser ──► GaussianSceneSoa ──► Persistent VRAM (GpuSceneState)
                                                               │
                                                               ▼
                           ┌─────────────────────────────────────────┐
                           │   CageScene (Tet & MVC Poly Cages)      │
                           │   Spring Solver, Pins & Collision Planes│
                           └────────────────────┬────────────────────┘
                                                │ Cage Vertices (skipped when stationary)
                                                ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │ Stage 1: Deformation, Polar Rotation & Split Compaction      [GPU/CPU]│
   │   • Per-tet Jacobian + 5-step Newton-Schulz polar rotation R ∈ SO(3)   │
   │   • Vertex mapping:       x' = Σ wₖ Xₖ  (Tet barycentric or 3D MVC)    │
   │   • Covariance transport: Σ' = J Σ Jᵀ + Mip₃D I,  α' stabilization     │
   │   • Atomic split compact: σ_max(J) > τ ⇒ transient major-axis child    │
   ├────────────────────────────────────────────────────────────────────────┤
   │ Stage 2: EWA Projection, Inverse-Ray SH & Compaction         [GPU/CPU]│
   │   • Camera projection, 3D/2D Mip-Splatting EWA, & workgroup scan cull  │
   │   • Tight opacity-aware ellipse AABBs (extent_x, extent_y)             │
   │   • Exact inverse-ray SH (Rᵀ d, degrees 0–3) ⇒ compact splat_attrs     │
   ├────────────────────────────────────────────────────────────────────────┤
   │ Stage 3: GPU Prefix Scan, Radix Sort & Tile Ranges           [GPU/CPU]│
   │   • Tight quadratic ellipse-vs-tile count & dynamic-precision key emit │
   │   • 3-level Blelloch scan + 4-bit LSD radix sort (subgroups/Blelloch)  │
   │   • Temporal Cache: skipped when camera & cages are unchanged          │
   ├────────────────────────────────────────────────────────────────────────┤
   │ Stage 4: Cooperative Tile Compositor -> Packed RGBA8         [GPU/CPU]│
   │   • 256-thread shared-memory 16×16 tiles over 4 storage buffers (~16×  │
   │     less bandwidth via pre-evaluated RGB)                              │
   │   • Double-buffered plane_any early T < ε termination & RGBA8 packing  │
   ├────────────────────────────────────────────────────────────────────────┤
   │ Stage 5: Zero-Copy Buffer-to-Texture Blit                       [WGPU]│
   │   • GPU-to-GPU copy_buffer_to_texture into Rgba8UnormSrgb swapchain    │
   ├────────────────────────────────────────────────────────────────────────┤
   │ Stage 6: Overlays & Telemetry                                   [WGPU]│
   │   • Wireframe cage rendering, vertex handles, and egui HUD (+ VSync)   │
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
In real-time GPU rendering, each tetrahedron's $`\mathrm{SO}(3)`$ polar rotation $`\mathbf{R}`$ (from $`\mathbf{J} = \mathbf{R}\mathbf{S}`$) transforms view directions into rest-frame coordinates $`\mathbf{d}_{\mathrm{local}} = \mathbf{R}^\top \mathbf{d}`$ before evaluating static VRAM SH coefficients once per surviving splat in Stage 2 — mathematically exact across degrees 0–3 while avoiding per-frame coefficient rotation. For deformed `.ply` snapshot exports, `ShRotator` rotates coefficients explicitly via exact degree 1 (3×3), degree 2 (5×5), and degree 3 (7×7) Wigner/monomial matrices.

#### 5. Strain-Driven Adaptive Splitting & Anti-Aliasing
Under strong tensile deformation ($`\sigma_{\max}(\mathbf{J}) > \tau`$), a splat's deformed 3D covariance $`\mathbf{\Sigma}'`$ is factored via eigen-decomposition along its major axis $`\mathbf{u}_{\max}`$ and transiently split into two overlapping child Gaussians without modifying the rest-state `.ply`. Isotropic 3D Mip-Splatting and 2D EWA low-pass covariance filters ($`\mathbf{\Sigma}' + v_{\mathrm{mip}} \mathbf{I}`$, $`\mathbf{\Sigma}_{2\mathrm{D}} + v_{2\mathrm{D}} \mathbf{I}`$) prevent sub-pixel compression shimmer, while tight opacity-aware ellipse bounds ($`\sqrt{2\ln(255\,\alpha_{2\mathrm{D}})} \sqrt{\Sigma'_{ii}}`$) and exact quadratic tile intersection eliminate wasted square-box tile pairs.

#### 6. 3D Mean Value Coordinates (Closed Polygonal Cages)
In addition to tetrahedral cages, closed triangular cages (`PolyCageMesh`, loadable via Wavefront `.obj` or procedural polyhedra) evaluate smooth harmonic-like 3D Mean Value Coordinates (Ju–Schaefer–Warren) with precomputed `weight_gradients` for closed-form parallel deformation Jacobians (`deform_scene_cpu`) and optional tetrahedralization (`to_tet_mesh()`) for full GPU deformation parity.

### Unified Device Model

`wgpu` is pinned to `30.0.1`, matching the runtime bundled with `cubecl 0.11`. `GpuCompute` registers the presentation `wgpu::Device` and `wgpu::Queue` directly with CubeCL's compute client via `preferred_wgpu_features(&adapter)`. As a result:
- Hardware compute accelerators (`SHADER_F16`, `SUBGROUP`, and `TIMESTAMP_QUERY`) are enabled automatically when supported by the adapter.
- A single hardware instance, adapter, device, and queue serve both compute dispatches and graphics presentation.
- `GpuPipelineState` reads CubeCL's underlying `WgpuServer` resource buffer directly and records a zero-copy `copy_buffer_to_texture` straight into the `Rgba8UnormSrgb` offscreen texture.
- Compute clients and VRAM allocations (`GpuSceneState` + `GpuFrameArena`) persist across frames with exponential growth (`1.5×`) only on capacity increases.

---

## ⚡ Measured Performance

Benchmarked on `v2.0.0` via `cargo run --release --bin benchmark` using hardware `WGPU TIMESTAMP_QUERY` counters:

- **Workload**: 100,000 synthetic Gaussians at 1920×1080 (`1,082,262` Stage 3 tile entries with tight ellipse AABBs vs `1,505,993` square-bound entries in `v1.0.1` — **28.1% fewer tile entries**)
- **CPU**: 12 execution threads (Rayon work-stealing pool)
- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (`Vulkan`, CubeCL `0.11`, `SHADER_F16`, `SUBGROUP`, `TIMESTAMP_QUERY`, shared `wgpu::Device`)

| Pipeline Stage | CPU Reference (`v2.0.0`) | GPU `v1.0.1` (Readback) | GPU `v2.0.0` (Dynamic Cage) | GPU `v2.0.0` Speedup |
| :--- | :---: | :---: | :---: | :---: |
| **Stage 1 — Deformation & Polar Rot** | 5.85 ms | 10.55 ms | **0.066 ms** (`0.000 ms` static cage) | **88.6×** vs CPU / **159.8×** vs `v1.0.1` |
| **Stage 2 — EWA Proj + Rᵀ d SH** | 1.53 ms | 7.46 ms | **0.242 ms** (`0.200 ms` orbit/static) | **6.3×** vs CPU / **30.8×** vs `v1.0.1` |
| **Stage 3 — Prefix Scan, Radix Sort & Binning** | 46.84 ms | 30.51 ms *(CPU)* | **10.018 ms** (`6.661 ms` orbit) | **4.7×** vs CPU / **3.0×** vs `v1.0.1` |
| **Stage 4 — Cooperative Tile Compositor** | 287.05 ms | 96.19 ms | **1.669 ms** (`1.211 ms` orbit/static) | **172.0×** vs CPU / **57.6×** vs `v1.0.1` |
| **Sum of Active Stages** | **341.27 ms** | **144.72 ms** | **11.996 ms** *(83.4 FPS GPU time)* | **28.4×** vs CPU / **12.1×** vs `v1.0.1` |

#### Persistent Pipeline & Zero-Copy End-to-End Throughput (`1920×1080`, 100k Gaussians)

| Execution Mode / Workload Regime | Measured Frame Time | Throughput | Speedup vs `v1.0.1` |
| :--- | :---: | :---: | :---: |
| **Zero-Copy End-to-End GPU (Dynamic Cage)** *(incl. `copy_buffer_to_texture` + submit)* | **9.19 ms** | **108.9 FPS** | — |
| **Zero-Copy End-to-End GPU (Camera Orbit)** *(incl. `copy_buffer_to_texture` + submit)* | **9.51 ms** | **105.2 FPS** | — |
| **GPU Camera Orbit (Stationary Cage, Stage 1 skipped, `TIMESTAMP_QUERY`)** | **8.072 ms** | **123.9 FPS** | — |
| **GPU Static View (Full Temporal Cache: Stage 1 + Stage 3 skipped)** | **1.420 ms** | **704.3 FPS** | **88.3×** vs `v1.0.1` (125.35 ms / 8.0 FPS) |
| **CPU Reference End-to-End Frame** *(incl. upload + blit)* | **263.29 ms** | **3.8 FPS** | Baseline |

> [!NOTE]
> When per-stage hardware `TIMESTAMP_QUERY` resolution is active on Dynamic Cage frames, CPU wall time is `15.729 ms` due to synchronous query readback per stage; normal rendering (`HeadlessRenderer::render_zero_copy` and interactive swapchain presentation) runs without per-stage query stalls at **9.19 ms (108.9 FPS)**. On the 1,073,930-Gaussian Mip-NeRF 360 `bonsai.ply` capture at 960×540 (`50,293` bound Gaussians, `~182k–232k` surviving splats), `record_demo` renders steady frames in `~9.3–11.1 ms` and exports the full 180-frame MP4 + GIF sequence in **`3.98s` total (`22.14 ms/frame` incl. simultaneous `ffmpeg` encoding)**.

### Criterion CPU Stage Microbenchmarks

Measured via `cargo bench --bench stages` (20,000 synthetic Gaussians at 1280×720, median estimate across 10 Criterion samples):

| Criterion Benchmark | Median Time | 95% Confidence Interval |
| :--- | :---: | :---: |
| `stage1_deformation_20k` | 1.58 ms | 1.56 ms – 1.60 ms |
| `stage2_projection_20k_720p` | 267.1 µs | 260.0 µs – 278.1 µs |
| `stage3_binning_20k_720p` | 5.66 ms | 5.63 ms – 5.69 ms |
| `stage4_compositor_720p` | 85.56 ms | 84.49 ms – 86.88 ms |
| `end_to_end_cpu_720p` | 94.02 ms | 91.67 ms – 96.20 ms |

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

# Run unit and integration tests (177 tests)
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

- When `ffmpeg` is present on `PATH`, the recorder pipes tight `RGBA8` GPU frames directly into a single-pass dual-output `ffmpeg` graph to generate `docs/cindergs_demo.mp4` and `docs/cindergs_demo.gif` simultaneously with zero PNG disk I/O (`--save-frames` keeps fast PNGs in `target/demo_frames/`).
- If `ffmpeg` is unavailable, parallel `CompressionType::Fast` PNG frames are written to `target/demo_frames/`.

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

- **Backend & Presentation**: Switch live between `CubeCL GPU` and `CPU Reference`, and toggle `VSync` (`Fifo` / `AutoNoVsync`) on the swapchain.
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
│   │   ├── gpu.rs          # WGPU device, hardware feature negotiation & swapchain init
│   │   ├── gpu_bridge.rs   # Standalone GPU bridge helpers, IndirectDispatchArgs & parity
│   │   ├── gpu_pool.rs     # Persistent GPU scratch & exponential growth capacity helpers
│   │   ├── gpu_state.rs    # Persistent VRAM GpuSceneState, GpuFrameArena & zero-copy blit
│   │   ├── headless.rs     # Offscreen WGPU/CubeCL renderer (render, zero-copy, rgba8)
│   │   ├── kernels/        # CubeCL kernels (binning, compositor, deformation, project, radix_sort, scan)
│   │   ├── pipeline.rs     # Staged orchestrator connecting GPU/CPU Stages 1–5 + overlays
│   │   ├── rasterizer/     # 16x16 tile compositor, SH math, rotation, and sort caching
│   │   ├── tiles/          # Tight ellipse bounds, quadratic tile intersection & CPU binning
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
