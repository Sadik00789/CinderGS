# Demo Artifacts

This directory contains the visual demonstration embedded in the project
[README](../README.md).

| File | Encoding | Details |
| :--- | :--- | :--- |
| `cindergs_demo.gif` | GIF | 800x450, 20 fps, 6 s, 128-colour palette |
| `cindergs_demo.mp4` | H.264 MP4 | 960x540, 30 fps, 6 s, CRF 18, `yuv420p` |

Both artifacts are produced from the same 180 rendered frames of the
Mip-NeRF 360 bonsai capture (`assets/bonsai.ply`, 1,073,930 Gaussians) using
the CubeCL GPU backend. The sequence orbits the scene, deforms the hero cage
canopy, then releases it into the damped spring simulation.

## Regenerating

```bash
cargo run --release --bin record_demo -- \
    --input assets/bonsai.ply \
    --look-at 0.38,0.88,1.35 \
    --distance 3.2 \
    --bounds -0.15,0.43,0.50,1.15,1.10,2.00
```

PNG source frames are written to `target/demo_frames/`. When `ffmpeg` is on
`PATH`, the recorder encodes both artifacts into this directory; otherwise the
PNGs are preserved for manual encoding.

The renderer and encoder settings are deterministic. Re-running the command on
the same build and ffmpeg version reproduces byte-identical artifacts (verified
by SHA-256), so regenerating without changing content intentionally produces no
Git diff.

## Storage

Both media files are tracked through Git LFS (see [`.gitattributes`](../.gitattributes)).
Run `git lfs install` once and `git lfs pull` after cloning to materialize the
binaries.
