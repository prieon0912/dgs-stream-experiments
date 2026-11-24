"""[env: gs] PLAN B: ignore DA3 gaussians, use only depth (aligned to GT pose scale) -> point cloud
in the COLMAP frame -> isotropic gaussians (graphdeco create_from_pcd style). Then train with 05_refine.py.

Note: these points are not renderable as-is in a meaningful way (no learned scale/opacity), so in
plan B the ladder becomes  "depth + N_short iters"  vs  "depth + 1k iters".  Write that in the results.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from scipy.spatial import cKDTree

from common import timing
from common.geom import unproject
from common.ply_io import logit, rgb2sh, save_ply, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf_q", type=float, default=0.3, help="drop pixels below this per-view conf quantile")
    ap.add_argument("--voxel", type=float, default=0.0, help="voxel size (scene units); 0 = auto (0.1%% of median depth)")
    ap.add_argument("--max_points", type=int, default=1_000_000)
    ap.add_argument("--ray_depth", action="store_true")
    args = ap.parse_args()

    r = torch.load(args.raw, map_location="cpu")
    t_start = time.time()
    depth = r["depth"].numpy()
    V, H, W = depth.shape
    K, E = r["intrinsics"].numpy(), r["gt_extrinsics"].numpy()
    img = r["processed_images"].numpy().astype(np.float32) / 255.0
    pts, cols = [], []
    for v in range(V):
        m = np.isfinite(depth[v]) & (depth[v] > 0)
        if r.get("conf") is not None:
            c = r["conf"][v].numpy()
            m &= c >= np.quantile(c, args.conf_q)
        p = unproject(depth[v], K[v], E[v], z_depth=not args.ray_depth).reshape(H, W, 3)
        pts.append(p[m]); cols.append(img[v][m])
    pts, cols = np.concatenate(pts), np.concatenate(cols)
    vox = args.voxel or 1e-3 * float(np.median(depth))
    key = np.floor(pts / vox).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    pts, cols = pts[first], cols[first]
    if len(pts) > args.max_points:
        sel = np.random.default_rng(0).choice(len(pts), args.max_points, replace=False)
        pts, cols = pts[sel], cols[sel]
    n = len(pts)
    dist, _ = cKDTree(pts).query(pts, k=4)
    d2 = np.maximum((dist[:, 1:] ** 2).mean(1), 1e-7)          # simple-knn: mean sq. dist of 3 NN
    d = dict(xyz=pts.astype(np.float32), f_dc=rgb2sh(cols).astype(np.float32),
             f_rest=np.zeros((n, 9), np.float32),
             opacity=np.full(n, logit(np.array(0.1)), np.float32),
             scale=np.repeat(np.log(np.sqrt(d2))[:, None], 3, 1).astype(np.float32),
             rot=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)))
    save_ply(d, args.out)
    sec = time.time() - t_start
    meta_p = os.path.splitext(args.raw)[0] + "_meta.json"
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    steps = {"da3_infer": float(meta["time_sec"])} if "time_sec" in meta else {}
    steps["depth_to_points"] = sec
    json.dump(dict(steps=steps, init_time_sec=float(sum(steps.values())) if "da3_infer" in steps else None,
                   gpu=meta.get("gpu"), gpu_exclusive=meta.get("gpu_exclusive"),
                   notes=["plan B: DA3 depth only (gaussian head output discarded)"]),
              open(timing.timing_path(args.out), "w"), indent=1)
    st = stats(d)
    with open(os.path.splitext(args.out)[0] + "_convert.json", "w") as fp:
        json.dump(dict(plan="B_depth_only", voxel=vox, args=vars(args), stats=st), fp, indent=1)
    print(json.dumps(st, indent=1)); print("saved", args.out)


if __name__ == "__main__":
    main()
