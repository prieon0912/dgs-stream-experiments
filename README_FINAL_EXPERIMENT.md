# Final experiment (dgs_exp_v7): first-frame init x update mode, 300 frames

N3DV flame_steak, 3DGStream, RTX A6000. Running since 2026-11-24 (results will be added to `results/` when done).

## Design (16 runs + pilot)

| init (first frame) | type | gaussians |
|---|---|---|
| opacity1m | DA3 all views, opacity top-1M, +1k refine (reference) | 1,000,000 |
| redundancy600k | DA3, Papantonakis spatial redundancy pruning (iterative), +1k | 600,000 |
| reproj600k | DA3, reprojection de-duplication across views, +1k | 600,000 |
| zipsplat05 | ZipSplat feed-forward generation (ratio 0.5), +1k | 155,520 |

Update modes (which gaussians inside the fixed AABB the NTC updates; the rasterizer always renders all):

- **A**: all inside the AABB (original 3DGStream)
- **B**: motion-history freezing (window 10, warm-up 10, full re-check every 30 frames; `B_TAU_XYZ` from the pilot, target ~35% active)
- **C**: image-difference mask (DecoGS-style, tau 25, radius 48, kappa 2 views)
- **EB**: adaptive AABB from image differences of the past N frames (used as an activity mask only; NTC bounds stay fixed) + B inside it. N in {10, 30} chosen by the pilot.

Code: `stream/selective_update.py`, `stream/train_frames_select.py` (modes, per-frame log incl. Gaussian-tile pairs, optional torch.profiler breakdown), `jobs/final_s0/worker.sh` (queue: pod BE = pilot + B/EB, pod AC = pilot C + A/C), `scripts/40_decide_final.py`, `scripts/41_summarize_final.py`.

## Findings so far (rasterization cost)

`tools/tile_probe/` renders existing first-frame PLYs (no training) and logs Gaussian-tile pairs and forward/backward time; `results/tile_pairs_probe.csv`.

| init | gaussians | tile pairs | fwd ms | bwd ms | 300-frame train s/frame |
|---|---|---|---|---|---|
| 3DGS 15k | 322k | 3.23M | 3.7 | 13.7 | 11.90 |
| direction_04+1k | 652k | 9.87M | 10.3 | 36.9 | 20.33 |
| DA3+1k all views | 1M | 8.26M | 8.9 | 39.2 | 21.92 |
| DA3 all views (no refine) | 1M | 7.80M | 8.8 | 42.9 | 24.70 |
| position_08+1k | 1M | 12.87M | 13.4 | 46.4 | 24.92 |

- Train time correlates best with render **backward** time (r = 0.99; 0.95 among DA3 rows), then gaussian / AABB-inside count (0.95 / 0.96); tile pairs alone 0.82 (0.26 among DA3 rows).
- Pilot profile (torch.profiler, frames 25-26, opacity1m, mode A): per frame ~11 s GPU in rasterizer kernels (render backward ~8.9 s, forward ~1.4 s) vs ~1.1 s in tiny-cuda-nn (NTC). So after frame 1 most of the time is rasterization (render + backward), and modes B/C/EB, which only shrink the NTC input, are expected to save little time. Reducing the gaussian count, or rasterizer-side changes (exact tile intersection as in Speedy-Splat SnugBox/AccuTile, AdR-Gaussian adaptive-radius culling, ROI/tile-masked rendering of changed regions), target the dominant cost.

## Other result tables (`results/`)

- `first_frame_all_results.csv`: 34 first-frame candidates (pruning 28 + EDGS 4 + ZipSplat 2), cam00 / train-view PSNR vs the opacity-1M reference, counts, times.
- `camera_count_results.csv`: camera-subset experiment, 300-frame streams (A6000).
- `raw.csv`, `alt_init.csv`: raw outputs of the pruning and alternative-init screens.
