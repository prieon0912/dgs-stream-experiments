# R*: fast rasterizer for 3DGStream (`ext/dgr_fast`)

Modified copy of SJoJoK/3DGStreamRasterizer @ 39b62ce (the rasterizer 3DGStream uses). Selected at run time by `DGR_MODE`:

| DGR_MODE | change | files |
|---|---|---|
| 0 (unset) | upstream behaviour | — |
| 1 | opacity-aware tile rect (SnugBox idea): a gaussian is binned only to tiles where alpha >= 1/255 can occur, intersected with the original 3-sigma rect. Pixels outside were already skipped by the blend loop, so the image is unchanged; radii are unchanged | `auxiliary.h` (`snugRect`), `forward.cu` (`preprocessCUDA<C,SNUG>`), `rasterizer_impl.cu` (`duplicateWithKeys<SNUG>`) |
| 2 | warp-level reduction of the backward atomics (DISTWAR idea, warp-shuffle part only): the 32 lanes of a warp sum their per-gaussian gradients with `__shfl_down_sync`, lane 0 does one `atomicAdd` | `backward.cu` (`renderCUDAWarp<C>`, `warpSum`) |
| 3 | 1 + 2 (the R* used in the pilots) | |

Own implementation of the two published ideas, not their official code. AccuTile and DISTWAR's SM/atomic balancing are not implemented.

## Build (pod, gs env, A6000)
```
cp -r /workspace/3DGStream/submodules/diff-gaussian-rasterization/third_party/glm ext/dgr_fast/third_party/
TORCH_CUDA_ARCH_LIST=8.6 pip install --no-build-isolation --no-deps --target /workspace/dgr_fast_pkg ./ext/dgr_fast
```
Use without touching the env's own rasterizer: `DGR_MODE=3 PYTHONPATH=/workspace/dgr_fast_pkg <command>`.

## Validation
- Offline (`tools/tile_probe/rbench.py`, `compare.py` -> `results/rbench_compare.csv`), frame-0 PLYs of the 4 final inits, 21 views:
  image identical in every mode (max abs diff 0); gradient relative error 1e-6 to 1e-3 (summation order of atomics);
  fwd+bwd speed-up for mode 3: opacity1m 2.91x, redundancy600k 3.26x, reproj600k 3.42x, zipsplat05 4.98x.
- 30-frame streaming pilots (`results/mstar_pilot_results.csv` neighbours, run names `final_s0_p2_<init>_AR`):
  cam00 PSNR within +-0.03 dB of mode A; train time per frame 1.49x (opacity1m), 1.54x, 1.70x, 2.23x (zipsplat05).
- Not yet: 300 frames, other scenes.
