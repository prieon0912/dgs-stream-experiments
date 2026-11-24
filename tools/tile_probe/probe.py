"""[env: gs] Measure Gaussian-tile pairs (num_rendered) and render cost of existing first-frame PLYs.

No training, no file under /workspace is modified. For every PLY and every camera of frame 0:
  - num_rendered   : Gaussian-tile pairs produced by the rasterizer (what gets sorted and blended)
  - visible        : gaussians with radius > 0
  - mean_radius    : mean screen radius (px) of visible gaussians
  - fwd_ms         : forward render time
  - fwdbwd_ms      : forward + backward time (one training-iteration render, like 3DGStream stage 1/2)
Results: <out>/tile_pairs.csv (mean over views per PLY) and <out>/tile_pairs_views.jsonl.

usage: python probe.py --frame_dir /workspace/data/flame_steak/frame000000 --out /workspace/tile_probe_out \
           --ply label=/path/a.ply --glob '/workspace/runs/camera_capped/*/refined.ply' ...
"""
import argparse
import csv
import glob
import json
import os
import statistics
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import diff_gaussian_rasterization as dgr  # noqa: E402

# ---- capture num_rendered / radii from the C++ call without touching the installed package ----
_LAST = {}
_REAL_C = dgr._C


class _CProxy:
    def __getattr__(self, name):
        return getattr(_REAL_C, name)

    def rasterize_gaussians(self, *args):
        r = _REAL_C.rasterize_gaussians(*args)
        try:
            _LAST["num_rendered"] = int(r[0])
        except Exception:
            _LAST["num_rendered"] = -1
        # graphdeco: (num_rendered, color, radii, ...); 3DGStream fork: (num_rendered, color, depth, radii, ...)
        # -> the first 1-D tensor after the image (depth is (H,W) and is skipped)
        rad = next((x for x in r[2:4] if torch.is_tensor(x) and x.dim() == 1), None)
        _LAST["radii"] = rad
        return r


dgr._C = _CProxy()

from common.colmap_io import load_cameras  # noqa: E402
from common.gs_render import GS, RCam, render  # noqa: E402


def sync():
    torch.cuda.synchronize()


def measure_view(gs, rc, reps):
    # forward only
    fw = []
    with torch.no_grad():
        for _ in range(reps):
            sync(); t = time.perf_counter()
            render(gs, rc)
            sync(); fw.append((time.perf_counter() - t) * 1e3)
    nr = _LAST["num_rendered"]
    radii = _LAST["radii"]
    if radii is None:
        n_vis, mean_r = -1, -1.0
    else:
        vis = radii > 0
        n_vis = int(vis.sum().item())
        mean_r = float(radii[vis].float().mean().item()) if n_vis else 0.0
    # forward + backward
    fb = []
    for _ in range(reps):
        gs.zero_grad(set_to_none=True)
        sync(); t = time.perf_counter()
        img = render(gs, rc)
        img.sum().backward()
        sync(); fb.append((time.perf_counter() - t) * 1e3)
    return dict(num_rendered=nr, visible=n_vis, mean_radius=mean_r,
                fwd_ms=statistics.median(fw), fwdbwd_ms=statistics.median(fb))


def label_of(path):
    parts = path.rstrip("/").split("/")
    return f"{parts[-2]}/{parts[-1]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame_dir", required=True)
    ap.add_argument("--images", default="images_2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ply", action="append", default=[], help="label=path")
    ap.add_argument("--glob", action="append", default=[])
    ap.add_argument("--exclude", action="append", default=["converted.ply", "_ALL.ply"],
                    help="skip paths containing this substring")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    items = []
    for s in args.ply:
        lab, p = s.split("=", 1)
        items.append((lab, p))
    for g in args.glob:
        for p in sorted(glob.glob(g, recursive=True)):
            if any(x in p for x in args.exclude):
                continue
            items.append((label_of(p), p))
    seen, uniq = set(), []
    for lab, p in items:
        rp = os.path.realpath(p)
        if rp not in seen and os.path.exists(p):
            seen.add(rp); uniq.append((lab, p))
    print(f"{len(uniq)} PLYs", flush=True)

    os.makedirs(args.out, exist_ok=True)
    cams = load_cameras(args.frame_dir, args.images)
    rcs = [RCam(c) for c in cams]
    print(f"{len(rcs)} cameras {cams[0].width}x{cams[0].height}", flush=True)

    csv_path = os.path.join(args.out, "tile_pairs.csv")
    done = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            done = {r["path"] for r in csv.DictReader(f)}
    fields = ["label", "path", "n_gaussians", "views", "num_rendered_mean", "num_rendered_train_mean",
              "visible_mean", "pairs_per_visible", "mean_radius", "fwd_ms_mean", "fwdbwd_ms_mean",
              "fwdbwd_ms_train_mean", "status"]
    new = not os.path.exists(csv_path)
    fcsv = open(csv_path, "a", newline="")
    w = csv.DictWriter(fcsv, fieldnames=fields)
    if new:
        w.writeheader(); fcsv.flush()
    fv = open(os.path.join(args.out, "tile_pairs_views.jsonl"), "a")

    for lab, p in uniq:
        if p in done:
            print(f"skip (done) {lab}", flush=True)
            continue
        row = dict(label=lab, path=p)
        try:
            gs = GS.from_ply(p)
            n = gs.xyz.shape[0]
            render(gs, rcs[0]); sync()
            per = []
            for c, rc in zip(cams, rcs):
                m = measure_view(gs, rc, args.reps)
                m.update(label=lab, path=p, cam=c.name)
                per.append(m)
                fv.write(json.dumps(m) + "\n")
            fv.flush()
            tr = [m for m in per if m["cam"] != "cam00"]
            mean = lambda k, L=per: sum(m[k] for m in L) / len(L)
            row.update(n_gaussians=n, views=len(per),
                       num_rendered_mean=round(mean("num_rendered")),
                       num_rendered_train_mean=round(mean("num_rendered", tr)),
                       visible_mean=round(mean("visible")),
                       pairs_per_visible=round(mean("num_rendered") / max(mean("visible"), 1), 3),
                       mean_radius=round(mean("mean_radius"), 3),
                       fwd_ms_mean=round(mean("fwd_ms"), 3), fwdbwd_ms_mean=round(mean("fwdbwd_ms"), 3),
                       fwdbwd_ms_train_mean=round(mean("fwdbwd_ms", tr), 3), status="ok")
            del gs
            torch.cuda.empty_cache()
        except Exception as e:  # one bad PLY must not stop the rest
            row["status"] = f"fail: {type(e).__name__}: {str(e)[:120]}"
            traceback.print_exc()
            torch.cuda.empty_cache()
        w.writerow(row); fcsv.flush()
        print(f"{row.get('status')}  {lab}  n={row.get('n_gaussians')}  pairs={row.get('num_rendered_mean')}  "
              f"fb={row.get('fwdbwd_ms_mean')}ms", flush=True)
    print("ALL DONE", flush=True)
    with open(os.path.join(args.out, "DONE"), "w") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
