"""[env: zipsplat] ZipSplat feed-forward init on frame 0 (train views only) -> our world frame PLY.

Steps (timed part marked *):
  load model weights            (not timed, like DA3 weight loading)
  warm-up on 2 views            (not timed: one-off Triton / gsplat kernel compilation)
  * forward pass + TokenBA      (ZipSplat's own default test-time refinement, ba=True)
  * Sim(3) alignment            predicted camera centres -> COLMAP camera centres (20 train views, Umeyama)
  * transform + write PLY       sh degree set to 1 (3DGStream), band 1 rotated with the scene

ZipSplat predicts Gaussians in the frame of the first input view with its own scale, so the
alignment uses the camera centres it predicts together with the Gaussians. The rotation part is
checked separately (angle between aligned predicted and COLMAP camera orientations).

The Gaussian count is set by --compression (fraction of scene tokens kept); it cannot be forced
to an exact N. The count actually produced is recorded.

  python alt_init/run_zipsplat.py --frame_dir $FRAME0 --out .../init.ply --compression 1.0
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

from alt_init.gaussian_ops import apply_sim3
from common import timing
from common.colmap_io import load_cameras
from common.geom import umeyama
from common.ply_io import load_ply, save_ply, set_sh_degree, stats, subset


def rot_angle_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame_dir", required=True)
    ap.add_argument("--images", default="images_2")
    ap.add_argument("--test_cam", default="cam00")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="zipsplat", help="zipsplat (504 px finetune) or zipsplat-252")
    ap.add_argument("--compression", type=float, default=1.0)
    ap.add_argument("--no_ba", action="store_true", help="skip TokenBA (not the paper default)")
    ap.add_argument("--sh_degree", type=int, default=1)
    ap.add_argument("--max_center_residual", type=float, default=0.05,
                    help="review threshold: median camera-centre residual / camera-rig radius")
    ap.add_argument("--max_gaussians", type=int, default=1_000_000,
                    help="cap (opacity top-N, the same rule as the DA3 1M cap) if ZipSplat returns more")
    ap.add_argument("--max_views", type=int, default=24, help="ZipSplat was trained with 2..24 views")
    ap.add_argument("--max_nonfinite_frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not (0 < args.compression <= 1):
        sys.exit("--compression must be in (0, 1]")
    fail_json = os.path.splitext(args.out)[0] + "_zipsplat_failed.json"

    def fail(code, reason, **extra):
        with open(fail_json, "w") as f:
            json.dump(dict(reason=reason, compression=args.compression, **extra), f, indent=1)
        print("ERROR:", reason)
        sys.exit(code)

    import torch
    from zipsplat import ZipSplat, load_image

    torch.manual_seed(args.seed)
    cams = load_cameras(args.frame_dir, images=args.images, exclude=(args.test_cam,))
    if not (2 <= len(cams) <= args.max_views):
        fail(5, f"{len(cams)} input views outside the range ZipSplat was trained for (2..{args.max_views})")
    images = [load_image(c.image_path) for c in cams]
    model = ZipSplat(weights=args.weights).cuda().eval()
    # expected count BEFORE running: tokens per view (model input / patch 14) x views x compression x per-token
    try:   # informational only (assumes patch size 14 of the DA3 backbone); the cap below works regardless
        W, H = min(model.input_sizes,
                   key=lambda s: abs(math.log(s[0] / s[1]) - math.log(images[0].shape[-1] / images[0].shape[-2])))
        per_token = model.model.gaussian_head.gaussians_per_token
        tokens = (W // 14) * (H // 14) * len(images)
        expected = int(math.ceil(args.compression * tokens)) * per_token
    except Exception:
        W = H = tokens = per_token = expected = -1
    print(f"model input {W}x{H}, {tokens} tokens, {per_token} Gaussians/token -> expected ~{expected:,} Gaussians"
          f"{' (> cap ' + format(args.max_gaussians, ',') + ', will be capped by opacity)' if expected > args.max_gaussians else ''}")
    try:
        # warm-up outside the timed part: Triton kernels (TokenBA / PnP) and gsplat compile on first use
        model(images[:2], compression=args.compression, ba=not args.no_ba)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        exclusive = timing.gpu_exclusive()
        torch.cuda.synchronize()
        t0 = time.time()
        pred = model(images, compression=args.compression, ba=not args.no_ba)
        torch.cuda.synchronize()
        t_infer = time.time() - t0
    except torch.cuda.OutOfMemoryError as e:
        fail(6, "CUDA out of memory in ZipSplat (no automatic fallback: changing settings would change the method)",
             views=len(images), expected_gaussians=expected, detail=str(e)[:500])
    peak_mb = torch.cuda.max_memory_allocated() / 2**20

    t1 = time.time()
    gaussians = pred["gaussians"] if isinstance(pred, dict) else pred[0]
    poses = pred["poses"]                                   # c2w, frame of the first input view
    tmp_ply = os.path.splitext(args.out)[0] + "_raw_zipsplat.ply"
    gaussians.save_ply(tmp_ply)
    R_pred = poses.R.detach().float().cpu().numpy().reshape(-1, 3, 3)
    c_pred = poses.t.detach().float().cpu().numpy().reshape(-1, 3)
    c_gt = np.stack([c.center for c in cams])
    R_gt_c2w = np.stack([c.w2c[:3, :3].T for c in cams])
    s, R, t = umeyama(c_pred.astype(np.float64), c_gt.astype(np.float64))
    res = np.linalg.norm(s * c_pred @ R.T + t - c_gt, axis=1)
    rig = float(np.linalg.norm(c_gt - c_gt.mean(0), axis=1).max())
    ang = [rot_angle_deg(R @ R_pred[i], R_gt_c2w[i]) for i in range(len(cams))]

    d = load_ply(tmp_ply)
    n_raw = len(d["xyz"])
    finite = np.all(np.isfinite(np.concatenate(
        [d["xyz"], d["f_dc"], d["f_rest"], d["opacity"][:, None], d["scale"], d["rot"]], 1)), axis=1)
    n_nonfinite = int((~finite).sum())
    if n_nonfinite > args.max_nonfinite_frac * n_raw:
        fail(7, f"{n_nonfinite} of {n_raw} Gaussians have NaN/inf values", n_raw=n_raw)
    d = subset(d, np.flatnonzero(finite)) if n_nonfinite else d
    n_capped = 0
    if len(d["xyz"]) > args.max_gaussians:          # same rule as the DA3 cap: keep the most opaque
        keep = np.sort(np.argsort(-d["opacity"], kind="stable")[: args.max_gaussians])
        n_capped = len(d["xyz"]) - len(keep)
        d = subset(d, keep)
    d = apply_sim3(d, s, R, t)
    d = set_sh_degree(d, args.sh_degree)
    save_ply(d, args.out)
    t_align = time.time() - t1

    residual_rel = float(np.median(res) / rig)
    alignment_pass = bool(np.isfinite(residual_rel) and residual_rel <= args.max_center_residual
                          and np.median(ang) <= 5.0)
    report = dict(method="zipsplat", weights=args.weights, compression=args.compression, token_ba=not args.no_ba,
                  n_views=len(cams), camera_names=[c.name for c in cams], n_gaussians=int(len(d["xyz"])),
                  n_raw=int(n_raw), n_expected=int(expected), n_nonfinite_dropped=n_nonfinite,
                  n_capped_by_opacity=int(n_capped), max_gaussians=args.max_gaussians, peak_gpu_mem_mb=peak_mb,
                  sim3_scale=float(s), center_residual_median=float(np.median(res)),
                  center_residual_rel=residual_rel, rig_radius=rig,
                  rotation_error_deg_median=float(np.median(ang)), rotation_error_deg_max=float(np.max(ang)),
                  alignment_pass=alignment_pass, alignment_threshold=args.max_center_residual,
                  infer_sec=t_infer, align_sec=t_align, stats=stats(d))
    with open(os.path.splitext(args.out)[0] + "_zipsplat.json", "w") as f:
        json.dump(report, f, indent=1)
    timing.write(args.out, "zipsplat_infer_ba", t_infer, exclusive=exclusive, measurement_required=True,
                 notes=["ZipSplat forward + TokenBA (weights loading excluded)", "camera poses: predicted, then Sim3 to COLMAP"],
                 init_metadata=dict(model="zipsplat", n_views=len(cams), camera_names=[c.name for c in cams],
                                    compression=args.compression, seed=args.seed),
                 alignment_residual_rel=residual_rel, alignment_pass=alignment_pass)
    timing.write(args.out, "align_convert", t_align, src_ply=args.out)
    print(json.dumps({k: v for k, v in report.items() if k != "stats"}, indent=1))
    if not alignment_pass:
        review = os.path.splitext(args.out)[0] + "_REVIEW.ply"
        os.replace(args.out, review)
        os.replace(timing.timing_path(args.out), timing.timing_path(review))
        print(f"WARNING: alignment above threshold -> moved to {review}; not used until reviewed")
        sys.exit(3)


if __name__ == "__main__":
    main()
