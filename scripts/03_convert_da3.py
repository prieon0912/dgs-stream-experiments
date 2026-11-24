"""[env: gs or da3, CPU] DA3 raw gaussians -> 3DGStream-ready PLY in the COLMAP frame of the suite.

Coordinate fix (the "gaussian output is misaligned" issue):
  DA3 builds gaussians from ITS OWN predicted poses (gs_adapter: means = pred_cam origin + ray * depth),
  while `prediction.depth/extrinsics` are aligned to our GT poses. Gaussian i is pixel i of (v h w),
  so unprojecting the aligned depth with GT poses gives an exact correspondence -> one Umeyama Sim(3).
  Applied to: means (sR x + t), scales (log + log s), rotations (q_R * q), SH band 1 (rotated by R).

SH: DA3-GIANT predicts SH degree 2. We keep DC + band 1 (rotated) for 3DGStream's sh_degree=1
    (--sh1 zero reproduces the "pad f_rest with 0" variant of the original plan).
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from common import timing
from common.geom import quat_mul, rotate_sh1, rotmat_to_quat_wxyz, umeyama_trimmed, unproject
from common.ply_io import logit, save_ply, stats
from common.diagnostics import sources_path
from common.result_io import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="output .pt of 02_run_da3.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--align", choices=["pixel", "none"], default="pixel")
    ap.add_argument("--ray_depth", action="store_true", help="treat depth as ray length (default z-depth)")
    ap.add_argument("--sh1", choices=["keep", "zero"], default="keep")
    ap.add_argument("--border", type=float, default=8 / 256, help="trim this fraction at image borders (DA3 export default)")
    ap.add_argument("--depth_prune", type=float, default=1.0, help="drop pixels above this per-view depth quantile (DA3 export uses 0.9)")
    ap.add_argument("--min_opacity", type=float, default=0.005)
    ap.add_argument("--n_corr", type=int, default=300000)
    ap.add_argument("--max-align-residual", type=float, default=0.02,
                    help="Review threshold for median residual / scene depth; an empirical guard, not a proven quality bound")
    ap.add_argument("--allow-high-align-error", action="store_true", help="Diagnostic override, recorded as alignment_pass=False")
    ap.add_argument("--save-sources", action="store_true", help="Save originating view IDs for overlap diagnostics; excluded from conversion timing")
    args = ap.parse_args()
    if args.max_align_residual <= 0 or args.n_corr < 3:
        ap.error("Require a positive alignment threshold and at least 3 correspondences")
    if os.path.exists(args.out):
        raise FileExistsError(f"Output already exists: {args.out}; use a new condition name")

    r = torch.load(args.raw, map_location="cpu")
    t_start = time.time()   # .pt loading excluded: it only exists because we split inference/convert
    depth = r["depth"].numpy()
    V, H, W = depth.shape
    means = r["gs_means"].numpy().astype(np.float64)
    N = means.shape[0]
    assert N == V * H * W, f"gaussians {N} != V*H*W {V * H * W}; pixel correspondence impossible -> use plan B"
    K = r["intrinsics"].numpy()
    E = r["gt_extrinsics"].numpy() if "gt_extrinsics" in r else r["extrinsics"].numpy()

    # -------- pixel mask (border + depth quantile + confidence for alignment)
    keep = np.zeros((V, H, W), bool)
    bh, bw = int(args.border * H), int(args.border * W)
    keep[:, bh:H - bh, bw:W - bw] = True
    if args.depth_prune < 1:
        q = np.quantile(depth.reshape(V, -1), args.depth_prune, axis=1)
        keep &= depth <= q[:, None, None]
    keep &= np.isfinite(depth) & (depth > 0)

    s, R, t = 1.0, np.eye(3), np.zeros(3)
    report = dict(align=args.align, alignment_pass=None, alignment_threshold=args.max_align_residual,
                  alignment_guard_note="Empirical review threshold, not a guaranteed reconstruction-quality bound")
    if args.align == "pixel":
        gt_pts = np.concatenate([unproject(depth[v], K[v], E[v], z_depth=not args.ray_depth) for v in range(V)])
        sel = keep.copy()
        if r.get("conf") is not None:
            conf = r["conf"].numpy()
            sel &= conf >= np.quantile(conf.reshape(V, -1), 0.3, axis=1)[:, None, None]
        idx = np.flatnonzero(sel.reshape(-1))
        if len(idx) < 3 or not np.isfinite(means[idx]).all():
            raise ValueError("Too few or non-finite alignment correspondences")
        rng = np.random.default_rng(0)
        idx = rng.choice(idx, min(args.n_corr, len(idx)), replace=False)
        s, R, t, _ = umeyama_trimmed(means[idx], gt_pts[idx])
        # residuals on all kept pixels, per view
        res_all = np.linalg.norm(s * means @ R.T + t - gt_pts, axis=1).reshape(V, H, W)
        scene = float(np.median(depth[keep]))
        per_view = [float(np.median(res_all[v][keep[v]])) for v in range(V)]
        report.update(scale=float(s), R=R.tolist(), t=t.tolist(), scene_median_depth=scene,
                      residual_median=float(np.median(res_all[keep])),
                      residual_median_rel=float(np.median(res_all[keep]) / scene),
                      residual_per_view=dict(zip(r["names"], per_view)))
        print(f"Sim3: s={s:.4f}  residual median {report['residual_median']:.4f} "
              f"({100 * report['residual_median_rel']:.2f}% of median depth {scene:.2f})")
        worst = max(per_view)
        print("per-view residual median: min %.4f max %.4f" % (min(per_view), worst))
        alignment_pass = (np.isfinite(report["residual_median_rel"]) and np.isfinite(s) and s > 0
                          and report["residual_median_rel"] <= args.max_align_residual)
        report["alignment_pass"] = bool(alignment_pass)
        write_json(os.path.splitext(args.out)[0] + "_alignment.json", report)
        if not alignment_pass and not args.allow_high_align_error:
            raise ValueError("Alignment review threshold exceeded. Inspect training-view geometry before proceeding; "
                             "no PLY was saved. --allow-high-align-error is diagnostic only.")
        if worst > 5 * np.median(per_view):
            print("WARNING: one view is far off -> possible view-order mismatch.")

    # -------- transform gaussians
    m = keep.reshape(-1)
    op = r["gs_opacities"].numpy().reshape(N)
    m &= op >= args.min_opacity
    xyz = (s * means[m] @ R.T + t).astype(np.float32)
    scales = r["gs_scales"].numpy()[m] * s
    rots = r["gs_rotations"].numpy()[m].astype(np.float64)
    rots = rots / np.linalg.norm(rots, axis=1, keepdims=True)
    rots = quat_mul(rotmat_to_quat_wxyz(R), rots)
    sh = r["gs_harmonics"].numpy()[m]                     # (n, 3, d_sh)
    n = len(xyz)
    if not n:
        raise ValueError("All Gaussians were filtered out")
    f_dc = sh[:, :, 0]
    if sh.shape[2] >= 4 and args.sh1 == "keep":
        sh1 = rotate_sh1(sh[:, :, 1:4].astype(np.float64), R)
    else:
        sh1 = np.zeros((n, 3, 3))
    d = dict(xyz=xyz, f_dc=f_dc.astype(np.float32), f_rest=sh1.reshape(n, 9).astype(np.float32),
             opacity=logit(op[m]).astype(np.float32), scale=np.log(np.maximum(scales, 1e-8)).astype(np.float32),
             rot=rots.astype(np.float32))
    if any(not np.isfinite(value).all() for value in d.values()):
        raise ValueError("Non-finite Gaussian attributes after conversion")
    save_ply(d, args.out)
    convert_sec = time.time() - t_start
    meta_p = os.path.splitext(args.raw)[0] + "_meta.json"
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    steps = {"da3_infer": float(meta["time_sec"])} if "time_sec" in meta else {}
    steps["convert"] = convert_sec
    json.dump(dict(steps=steps, init_time_sec=float(sum(steps.values())) if "da3_infer" in steps else None,
                   gpu=meta.get("gpu") or timing.gpu_name(), gpu_exclusive=meta.get("gpu_exclusive"),
                   da3_model=meta.get("model"), da3_peak_mem_gb=meta.get("peak_mem_gb"),
                   init_metadata=dict(model=meta.get("model", r.get("model")),
                                      n_raw_gaussians=int(N), n_converted_gaussians=int(n), n_final_gaussians=int(n),
                                      n_views=V, camera_names=r["names"], selection=r.get("selection", {}),
                                      process_res=r.get("process_res"), max_gaussians=0, refine_iters=0,
                                      alignment_pass=report["alignment_pass"],
                                      alignment_residual_rel=report.get("residual_median_rel"),
                                      alignment_threshold=args.max_align_residual,
                                      filter_policy=dict(border=args.border, depth_prune=args.depth_prune,
                                                         min_opacity=args.min_opacity, sh1=args.sh1)),
                   notes=["DA3 init = model.inference() + alignment/conversion (CPU)",
                          "excluded: weight loading %.1fs" % meta.get("model_load_sec", float("nan"))]),
              open(timing.timing_path(args.out), "w"), indent=1)
    print(f"init time: DA3 {meta.get('time_sec', float('nan')):.1f}s + convert {convert_sec:.1f}s")
    st = stats(d)
    report.update(stats=st, args=vars(args), d_sh_in=int(sh.shape[2]),
                  n_views=V, camera_names=r["names"], selection=r.get("selection", {}),
                  n_raw_gaussians=int(N), n_converted_gaussians=int(n))
    with open(os.path.splitext(args.out)[0] + "_convert.json", "w") as fp:
        json.dump(report, fp, indent=1)
    print(json.dumps(st, indent=1))
    if n * 3 > 2 ** 24:
        print(f"NOTE: {n} gaussians: 3DGStream's torch.quantile in get_xyz_bound needs N*3 <= 16.7M "
              f"-> run 07_subsample.py --topk before streaming.")
    print("saved", args.out)
    if args.save_sources:
        view_ids = np.repeat(np.arange(V, dtype=np.int16), H * W)[m]
        np.savez_compressed(sources_path(args.out), view_ids=view_ids, camera_names=np.asarray(r["names"]))
        print("saved view-origin diagnostics (outside timed conversion)")


if __name__ == "__main__":
    main()
