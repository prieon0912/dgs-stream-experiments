"""[env: gs, GPU, cwd = $REPO_GS] Phase 1 identity check: fused-ssim vs 3DGStream utils.loss_utils.ssim.

  cd $REPO_GS && python $EXP/tools/ssim_check.py --scene $SCENE [--out $EXP/results/ssim_check.json]

20 image pairs from flame_steak (images_2, the training resolution), img1 = "render" (gets the gradient),
img2 = GT, both (3,H,W) exactly as train_frames calls ssim(image, gt_image):
  10 pairs: cam_v frame t+1 vs frame t  (real scene motion; t, v spread over the sequence / train views)
  10 pairs: GT vs GT + gaussian noise (sigma 0.02..0.11) blurred 3x3, clamped  (render-like error, SSIM ~0.6-0.95)
For each pair, both functions are evaluated on the SAME tensors, and the fused one through the exact wrapper the
hook installs (train_frames_select.make_fused_ssim):
  value:     |ssim_fused - ssim_upstream|                        <= 1e-4
  gradient:  ||g_fused - g_upstream||_2 / ||g_upstream||_2       <= 1e-3   (g = d ssim / d img1)
Also reports the median time per forward+backward of each (info). Exit code 1 if any pair fails.
"""
import argparse
import glob
import json
import os
import statistics
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
REPO_GS = os.environ.get("REPO_GS", os.getcwd())
for p in (REPO_GS, str(EXP / "stream")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402
import torch.nn.functional as F                      # noqa: E402
from PIL import Image                                # noqa: E402


def load(scene, frame, cam, images):
    hits = sorted(glob.glob(os.path.join(scene, f"frame{frame:06d}", images, cam + ".*")))
    if not hits:
        raise SystemExit(f"no image {cam} in frame {frame}")
    a = np.asarray(Image.open(hits[0]).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).contiguous().cuda()


def pairs(scene, images, n_frames):
    cams = sorted({os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(os.path.join(scene, "frame000000", images, "cam*.*"))})
    train = [c for c in cams if c != "cam00"] or cams
    out = []
    for i in range(10):
        t = int(i * (n_frames - 2) / 9)
        cam = train[i % len(train)]
        out.append((f"motion {cam} f{t + 1}-f{t}", load(scene, t + 1, cam, images), load(scene, t, cam, images)))
    g = torch.Generator(device="cuda").manual_seed(0)
    k = torch.ones(3, 1, 3, 3, device="cuda") / 9
    for i in range(10):
        t = int(i * (n_frames - 1) / 9)
        cam = train[(i + 3) % len(train)]
        gt = load(scene, t, cam, images)
        sigma = 0.02 + 0.01 * i
        noisy = gt + sigma * torch.randn(gt.shape, device="cuda", generator=g)
        noisy = F.conv2d(noisy[None], k, padding=1, groups=3)[0].clamp(0, 1)
        out.append((f"noise{sigma:.2f} {cam} f{t}", noisy, gt))
    return out


def value_and_grad(fn, img1, img2):
    x = img1.detach().clone().requires_grad_(True)
    v = fn(x, img2)
    v.backward()
    return float(v), x.grad.detach()


def timed(fn, img1, img2, reps=20):
    ts = []
    for _ in range(reps):
        x = img1.detach().clone().requires_grad_(True)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(x, img2).backward()
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=os.environ.get("SCENE", "/workspace/data/flame_steak"))
    ap.add_argument("--images", default="images_2")
    ap.add_argument("--n_frames", type=int, default=300)
    ap.add_argument("--value_tol", type=float, default=1e-4)
    ap.add_argument("--grad_rel_tol", type=float, default=1e-3)
    ap.add_argument("--out", default=str(EXP / "results" / "ssim_check.json"))
    a = ap.parse_args()

    from utils.loss_utils import ssim as ssim_upstream      # 3DGStream
    from fused_ssim import fused_ssim
    import train_frames_select as tfs
    ssim_fused = tfs.make_fused_ssim(fused_ssim)             # the exact function the hook installs

    rows = []
    for name, img1, img2 in pairs(a.scene, a.images, a.n_frames):
        vu, gu = value_and_grad(ssim_upstream, img1, img2)
        vf, gf = value_and_grad(ssim_fused, img1, img2)
        dv = abs(vf - vu)
        rel = float((gf - gu).norm() / gu.norm().clamp_min(1e-30))
        rows.append(dict(pair=name, shape=list(img1.shape), ssim_upstream=vu, ssim_fused=vf, value_diff=dv,
                         grad_rel_diff=rel, grad_max_abs_diff=float((gf - gu).abs().max()),
                         ok=dv <= a.value_tol and rel <= a.grad_rel_tol))
        print(f"{'ok  ' if rows[-1]['ok'] else 'FAIL'} {name:32s} ssim={vu:.6f} |dv|={dv:.2e} grad_rel={rel:.2e}")
    _, img1, img2 = pairs(a.scene, a.images, a.n_frames)[0]
    timing = dict(upstream_ms=timed(ssim_upstream, img1, img2), fused_ms=timed(ssim_fused, img1, img2))
    res = dict(pass_all=all(r["ok"] for r in rows), value_tol=a.value_tol, grad_rel_tol=a.grad_rel_tol,
               max_value_diff=max(r["value_diff"] for r in rows),
               max_grad_rel_diff=max(r["grad_rel_diff"] for r in rows),
               timing_fwd_bwd_median=timing, torch=torch.__version__, gpu=torch.cuda.get_device_name(0), pairs=rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "pairs"}, indent=1))
    sys.exit(0 if res["pass_all"] else 1)


if __name__ == "__main__":
    main()
