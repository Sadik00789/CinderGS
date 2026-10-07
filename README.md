# CinderGS: Native 3D Gaussian Splatting Engine with Volumetric Cage Kinematics

[![Rust Edition 2024](https://img.shields.io/badge/Rust_Edition-2024-orange.svg?style=flat-square&logo=rust)](https://www.rust-lang.org/)
[![WGPU 22](https://img.shields.io/badge/WGPU-v22-blue.svg?style=flat-square&logo=webgpu)](https://wgpu.rs/)
[![CubeCL 0.11](https://img.shields.io/badge/CubeCL-v0.11-darkgreen.svg?style=flat-square)](https://cubecl.org/)
[![CUDA/C++ Free](https://img.shields.io/badge/C%2B%2B%2FCUDA-Zero_Dependencies-success.svg?style=flat-square)](https://github.com/Sadik00789/CinderGS)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg?style=flat-square)](LICENSE)
[![CI](https://github.com/Sadik00789/CinderGS/actions/workflows/ci.yml/badge.svg)](https://github.com/Sadik00789/CinderGS/actions/workflows/ci.yml)
[![Tests Passing](https://img.shields.io/badge/Tests-86%20Passed-brightgreen.svg?style=flat-square)](tests/integration_tests.rs)

**CinderGS** is an ultra-high-performance, state-of-the-art 3D Gaussian Splatting (3DGS) rasterization and volumetric deformation engine written entirely in pure Rust. Built on top of **CubeCL** and **WGPU**, CinderGS completely eliminates legacy C++, CUDA toolkit, LibTorch, and Inria native extensions in favor of fully portable, multi-backend GPU compute kernels that run natively across Linux (Vulkan), macOS (Metal), and Windows (DirectX 12 / Vulkan).

---

## Key Architectural Innovations

1. **Pure Rust & CubeCL Compute Pipeline:**
   - Full elimination of external CUDA runtime, `nvcc`, Python bindings, and LibTorch.
   - Compute kernels written in CubeCL compile just-in-time to SPIR-V, WGSL, and native GPU bytecode across any hardware vendor.
2. **Zero-Copy Memory-Mapped PLY Ingestion:**
   - Sequential little-endian binary header and attribute streaming via `memmap2` and `bytemuck`.
   - Ingests over **2,000,000 raw Gaussian splats in under 85 ms** (62 properties per vertex, 248 bytes/vertex).
3. **Real-Time Volumetric Cage Kinematics:**
   - Arbitrary mesh deformation using a bounding 5-tetrahedron lattice with barycentric coordinates $\mathbf{w}_i = (w_0, w_1, w_2, w_3)$.
   - Affine covariance deformation: $\Sigma' = J \Sigma_{\text{rest}} J^T$ computed per splat.
   - Energy-conserving volumetric opacity adjustment: $\alpha' = 1 - (1 - \alpha)^{1 / \det(J)}$.
4. **Inverse-Ray Spherical Harmonics Evaluation:**
   - Rotates the viewing direction into local tetrahedral reference coordinates ($\mathbf{d}_{\text{local}} = R^T \mathbf{d}_{\text{world}}$) using Higham polar decomposition $R \in \mathrm{SO}(3)$, bypassing costly Wigner D-matrix rotations on 48 SH coefficients.
5. **Cooperative Shared-Memory Tile Compositor:**
   - Viewport partitioned into $16 \times 16$ pixel workgroups.
   - 256-thread cooperative tile fetching into GPU shared memory with deadlock-free barrier synchronization and early ray termination ($\mathcal{T} < 10^{-4}$).
6. **Damped Spring Cage Dynamics:**
   - Interactive Hookean spring physics with timestep clamping ($\Delta t_{\text{sim}} \le 0.033\text{ s}$) preventing numerical instability during hitching.
   - Unpinned vertices physically oscillate and settle when dragged cage handles are released, accompanied by radial jiggle impulse perturbations.

---

## System Architecture & Frame Graph

### A. High-Level Engine Pipeline

```
  +--------------------------------------------------------------------------+
  |                             CinderGS Engine                              |
  +--------------------------------------------------------------------------+
       |
       | 1. Memory-Mapped Streaming
       v
  +-------------------------+      Padded AABB      +------------------------+
  |    PlyLoader (mmap)     | --------------------> | 5-Tet Cage Generation  |
  |  (248B/vtx, Zero-Copy)  |                       |  (TetMesh / Bounding)  |
  +-------------------------+                       +------------------------+
       |                                                         |
       | Structure-of-Arrays (SoA)                               | Barycentric Bind
       v                                                         v
  +--------------------------------------------------------------------------+
  |                      RenderPipelineOrchestrator                          |
  |  [ExecutionBackend: CpuReference  <--->  ExecutionBackend: CubeClGpu]    |
  +--------------------------------------------------------------------------+
       |
       +---> Stage 1: Volumetric Cage Deformation (J = Ds * Dm^-1, R in SO(3))
       |
       +---> Stage 2: Camera Frustum EWA 3D-to-2D Projection & Conics
       |
       +---> Stage 3: Monotonic 64-bit Tile-Depth Key Generation & Radix Sort
       |
       +---> Stage 4: Shared-Memory 16x16 Cooperative Tile Compositor
       |
       +---> Stage 5: Fullscreen Blit to Presentation Swapchain
       |
       +---> Stage 6: Line-List Wireframe & Diamond Billboard Handles Overlay
       |
       v
  +--------------------------------------------------------------------------+
  |                    WGPU Swapchain Surface + egui HUD                     |
  +--------------------------------------------------------------------------+
```

### B. Frame Execution Graph

```
[Start Frame Tick]
        |
        v
 [Orbit Camera / Cage Picker] ---> Process Window Mouse & Pan Events
        |
        v
 [Cage Spring Simulator] --------> Timestep Clamped Semi-Implicit Euler
        |                          x_k^(t+1) = x_k^t + v_k^(t+1) * dt_sim
        v
 [Compute Tet Jacobians] --------> J = D_s * D_m^-1, R = HighamPolar(J)
        |
        v
 [Deform Gaussians Kernel] ------> x' = Sum(w_k * x_vk), Sigma' = J * Sigma * J^T
        |                          alpha' = 1 - (1 - alpha)^(1/det(J))
        v
 [Project Gaussians Kernel] -----> View transform, J_proj, Sigma_2D, Conic (a,b,c)
        |
        v
 [Tile Binning & Sort] ----------> 16x16 tile overlap count, 64-bit Radix Sort
        |
        v
 [Tile Compositor Kernel] -------> Shared-memory tile rasterization into offscreen buffer
        |
        v
 [Fullscreen Blit Pipeline] -----> Texture upload & fullscreen quad blit
        |
        v
 [Wireframe Pass] ---------------> Line-list cyan cage edges + gold diamond handles
        |
        v
 [egui HUD Overlay] -------------> FPS, telemetry, spring physics sliders, backend switch
        |
        v
  [Present Frame]
```

---

## Mathematical Formulations

### 1. 3D Covariance Matrix Construction

From optimized raw quaternion rotation $\mathbf{q} \in \mathbb{H}$ and logarithmic scale factors $\mathbf{s} \in \mathbb{R}^3$:

$$
\hat{\mathbf{q}} = \begin{cases} 
\begin{bmatrix} 0 & 0 & 0 & 1 \end{bmatrix}^T & \text{if } \|\mathbf{q}\| < 10^{-6} \\ 
\frac{\mathbf{q}}{\|\mathbf{q}\|} & \text{otherwise} 
\end{cases}
$$

$$
\mathbf{s}_{\text{clamped}} = \max(\exp(\mathbf{s}), 10^{-4})
$$

$$
\Sigma = R(\hat{\mathbf{q}}) \cdot \mathrm{diag}(\mathbf{s}_{\text{clamped}}^2) \cdot R(\hat{\mathbf{q}})^T \succ 0
$$

### 2. EWA Perspective Projection & Conics

Given world-to-camera transform $W \in \mathbb{R}^{4 \times 4}$, camera-space position $\mathbf{t} = W \cdot \mathbf{p}$, and projection Jacobian $J_{\text{proj}}$:

$$
J_{\text{proj}} = \begin{bmatrix} 
\frac{f_x}{t_z} & 0 & -\frac{f_x t_x}{t_z^2} \\ 
0 & \frac{f_y}{t_z} & -\frac{f_y t_y}{t_z^2} 
\end{bmatrix}
$$

$$
\Sigma_{\text{cam}} = W_{3 \times 3} \Sigma W_{3 \times 3}^T, \quad \Sigma_{2D} = J_{\text{proj}} \Sigma_{\text{cam}} J_{\text{proj}}^T + \begin{bmatrix} 0.3 & 0 \\ 0 & 0.3 \end{bmatrix}
$$

$$
\det(\Sigma_{2D}) = \Sigma_{2D, 00} \Sigma_{2D, 11} - \Sigma_{2D, 01}^2
$$

$$
\mathbf{c}_{\text{conic}} = \begin{bmatrix} a \\ b \\ c \end{bmatrix} = \frac{1}{\det(\Sigma_{2D})} \begin{bmatrix} \Sigma_{2D, 11} \\ -\Sigma_{2D, 01} \\ \Sigma_{2D, 00} \end{bmatrix}
$$

$$
\lambda_{\max} = \frac{\Sigma_{2D, 00} + \Sigma_{2D, 11}}{2} + \sqrt{\left(\frac{\Sigma_{2D, 00} - \Sigma_{2D, 11}}{2}\right)^2 + \Sigma_{2D, 01}^2}
$$

$$
r = \mathrm{clamp}\left(\lceil 3.0 \sqrt{\lambda_{\max}} \rceil, 1, 1024\right)
$$

### 3. Volumetric Cage Kinematics

Each Gaussian $i$ is bound to tetrahedron $\text{tet} = (v_0, v_1, v_2, v_3)$ via barycentric coordinates $\mathbf{w}_i$:

$$
\mathbf{x}'_i = \sum_{k=0}^3 w_{i, k} \mathbf{x}_{v_k}, \quad D_s = \begin{bmatrix} \mathbf{x}_{v_1} - \mathbf{x}_{v_0} & \mathbf{x}_{v_2} - \mathbf{x}_{v_0} & \mathbf{x}_{v_3} - \mathbf{x}_{v_0} \end{bmatrix}
$$

$$
J = D_s D_m^{-1}, \quad \Sigma' = J \Sigma_{\text{rest}} J^T
$$

$$
\alpha' = 1 - (1 - \alpha)^{1 / \max(\det(J), 10^{-6})}
$$

### 4. Higham Polar Decomposition for SH Rotation

To rotate viewing direction without mutating 48 SH spherical harmonics coefficients, the deformation Jacobian $J$ is decomposed into $J = R P$ where $R \in \mathrm{SO}(3)$ using scaled Newton-Schulz iterations:

$$
R_0 = J, \quad R_{k+1} = \frac{1}{2} \left( \gamma_k R_k + \frac{1}{\gamma_k} R_k^{-T} \right)
$$

Enforcing proper chirality $\det(R) = +1$, the view direction is evaluated in local reference coordinates:

$$
\mathbf{d}_{\text{local}} = R^T \left( \frac{\mathbf{p}' - \mathbf{c}_{\text{cam}}}{\|\mathbf{p}' - \mathbf{c}_{\text{cam}}\|} \right)
$$

### 5. Damped Spring Cage Dynamics

Cage elasticity update per unpinned vertex $k$:

$$
\Delta t_{\text{sim}} = \min(\Delta t, 0.033)
$$

$$
\begin{aligned}
\mathbf{a}_k &= \frac{k_{\text{stiffness}}}{m} (\mathbf{X}_{\text{rest}, k} - \mathbf{x}_k) \\
\mathbf{v}_k^{t+1} &= \gamma \cdot (\mathbf{v}_k^t + \mathbf{a}_k \Delta t_{\text{sim}}) \\
\mathbf{x}_k^{t+1} &= \mathbf{x}_k^t + \mathbf{v}_k^{t+1} \Delta t_{\text{sim}}
\end{aligned}
$$

Baseline parameters: Stiffness $k = 180.0\text{ N/m}$, damping $\gamma = 0.92$, mass $m = 1.0\text{ kg}$.

---

## Benchmark Latency Table

*Empirically measured on hardware: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM / Vulkan / CubeCL) at 1920 × 1080 resolution with 100,000 Gaussians.*

| Stage | CPU Reference Mode (Multi-Threaded) | CubeCL GPU Native Mode (RTX 3050 Laptop) |
| :--- | :---: | :---: |
| **Stage 1: Cage Deformation** | 9.90 ms | 0.52 ms |
| **Stage 2: EWA 3D-to-2D Projection** | 0.92 ms | 0.26 ms |
| **Stage 3: Tile Binning & Sort** | 57.02 ms | 2.95 ms |
| **Stage 4: Tile Compositor** | 267.93 ms | **5.78 ms** |
| **Stage 5: Fullscreen Blit & Overlays** | 0.28 ms | 0.30 ms |
| **Total Frame Latency** | **336.05 ms** | **9.81 ms** |
| **Effective Framerate** | **3.0 FPS** | **101.9 FPS** |

> *"The CPU Reference backend serves as a deterministic mathematical ground truth for headless CI and platforms without WebGPU/Vulkan compute support. The CubeCL GPU backend executes entirely in hardware with zero CPU readbacks, maintaining sustained real-time performance."*

---

## Quickstart & Build Instructions

### Prerequisites
- **Rust Toolchain:** Stable 1.80+ or Nightly (Edition 2024 compatible)
- **Vulkan / Metal / DirectX 12 GPU Drivers**

### 1. Build and Run Test Suite
```bash
# Verify Clippy with zero warnings
cargo clippy --all-targets -- -D warnings

# Execute full unit and integration test suite
cargo test --all-targets --release
```

### 2. Launch with Procedural Synthetic Cluster
```bash
cargo run --release
```

### 3. Ingest a Binary PLY Capture
```bash
cargo run --release -- path/to/capture.ply
```

### 4. Run Benchmark Suite
```bash
cargo run --release --bin benchmark
```

---

## Viewport Controls & HUD Guide

| Input | Action |
| :--- | :--- |
| **Right Mouse Button (RMB) + Drag** | Spherical orbit around pivot target |
| **Middle Mouse Button (MMB) + Drag** | Camera sensor-plane panning |
| **Scroll Wheel** | Proportional distance zooming |
| **Left Mouse Button (LMB) on Handle** | Pick and drag volumetric cage vertex |
| **LMB Release** | Release pinned vertex to activate spring oscillation |

### Immediate-Mode HUD Controls (egui)
- **Execution Backend:** Switch between `CPU Reference` (deterministic validation) and `CubeCL GPU` (hardware compute dispatch).
- **Show Cage Wireframe:** Toggle translucent cyan tetrahedral edges and glowing gold vertex pick handles.
- **Spring Stiffness (10 – 500 N/m):** Adjust cage elasticity restorative force.
- **Spring Damping (0.50 – 0.99):** Control oscillation settling rate.
- **Trigger Jiggle Impulse:** Perturb all unpinned cage vertices radially outward with an explosive velocity pulse.
- **Reset Cage / Reset Camera:** Restore rest lattice topology or orbit view.
