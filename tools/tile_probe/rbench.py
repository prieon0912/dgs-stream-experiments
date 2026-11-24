"""[env: gs] Rasterizer variant benchmark (one process per variant: DGR_MODE is read once per process).

For a PLY and the 21 cameras of frame 0, with an L1 loss against the real images (realistic gradients):
  - forward / forward+backward time (median of reps)
  - Gaussian-tile pairs
  - saves the image and the gradients (xyz, opacity, scale, rot, f_dc) of the first --save_views views to
    <out>/<label>_<mode>.pt so that variants can be compared against mode 0 (compare.py).

  DGR_MODE=3 PYTHONPATH=/workspace/dgr_fast_pkg python rbench.py --ply X.ply --label opacity1m --out /workspace/rbench
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                     # noqa: E402
import torch                           # noqa: E402
import diff_gaussian_rasterization as dgr   # noqa: E402

_LAST = {}
_REAL = dgr._C


class _P:
    def __getattr__(self, n):
        return getattr(_REAL, n)

    def rasterize_gaussians(self, *a):
        r = _REAL.rasterize_gaussians(*a)
        try:
            _LAST["pairs"] = int(r[0])
        except Exception:
            _LAST["pairs"] = -1
        return r


dgr._C = _P()
from common.colmap_io import load_cameras, load_image    # noqa: E402
from common.gs_render import GS, RCam, render             # noqa: E402


def sync():
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--frame_dir", default="/workspace/data/flame_steak/frame000000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--save_views", type=int, default=4)
    a = ap.parse_args()
    mode = int(os.environ.get("DGR_MODE", "0"))
    os.makedirs(a.out, exist_ok=True)
    cams = load_cameras(a.frame_dir, "images_2")
    rcs = [RCam(c) for c in cams]
    gts = [torch.from_numpy(load_image(c.image_path)).permute(2, 0, 1).cuda() for c in cams]
    gs = GS.from_ply(a.ply)
    render(gs, rcs[0]); sync()
    rows, saved = [], []
    for vi, (c, rc, gt) in enumerate(zip(cams, rcs, gts)):
        fw = []
        with torch.no_grad():
            for _ in range(a.reps):
                sync(); t = time.perf_counter(); render(gs, rc); sync(); fw.append((time.perf_counter() - t) * 1e3)
        pairs = _LAST.get("pairs", -1)
        fb = []
        for r in range(a.reps):
            gs.zero_grad(set_to_none=True)
            sync(); t = time.perf_counter()
            img = render(gs, rc)
            loss = (img - gt).abs().mean()
            loss.backward()
            sync(); fb.append((time.perf_counter() - t) * 1e3)
        rows.append(dict(cam=c.name, pairs=pairs, fwd_ms=statistics.median(fw), fwdbwd_ms=statistics.median(fb)))
        if vi < a.save_views:
            gs.zero_grad(set_to_none=True)
            img = render(gs, rc)
            (img - gt).abs().mean().backward()
            saved.append(dict(cam=c.name, image=img.detach().cpu(),
                              grads={k: getattr(gs, k).grad.detach().cpu().clone()
                                     for k in ("xyz", "opacity", "scale", "rot", "f_dc")}))
    torch.save(saved, os.path.join(a.out, f"{a.label}_m{mode}.pt"))
    summ = dict(label=a.label, mode=mode, n=int(gs.xyz.shape[0]),
                pairs=float(np.mean([r["pairs"] for r in rows])),
                fwd_ms=float(np.mean([r["fwd_ms"] for r in rows])),
                fwdbwd_ms=float(np.mean([r["fwdbwd_ms"] for r in rows])))
    summ["bwd_ms"] = summ["fwdbwd_ms"] - summ["fwd_ms"]
    with open(os.path.join(a.out, "rbench.jsonl"), "a") as f:
        f.write(json.dumps(summ) + "\n")
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
