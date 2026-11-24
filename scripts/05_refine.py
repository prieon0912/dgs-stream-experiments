"""[env: gs] Short 3DGS refinement of an init PLY on frame 0 train views (cam00 excluded).

Self-contained graphdeco-style loop (no Scene/train.py patching):
  loss = 0.8 L1 + 0.2 (1-SSIM), Adam with graphdeco per-param LRs,
  NO densification, NO opacity reset, position lr = 1.6e-4 * cameras_extent (constant for short runs).
SH: all bands of the PLY active from iter 0 (DA3 already provides band 1); --sh_warmup reproduces
graphdeco's "start at degree 0, +1 every 1000 iters".
"""
import argparse
import json
import os
import random
import shutil
import time

import numpy as np
import torch

from common import timing
from common.colmap_io import load_cameras, load_image, nerfpp_radius
from common.timing import gpu_exclusive
from common.gs_render import GS, RCam, render
from common.metrics import psnr, ssim_t, to_t
from common.ply_io import save_ply
from common.diagnostics import sources_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", required=True, help="output PLY")
    ap.add_argument("--frame_dir", required=True)
    ap.add_argument("--images", default="images_2")
    ap.add_argument("--test_cam", default="cam00")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--opacity_lr", type=float, default=0.05)   # 3DGS original (2023) value; newer graphdeco uses 0.025
    ap.add_argument("--sh_warmup", action="store_true")
    ap.add_argument("--white_bg", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.iters < 1:
        ap.error("--iters must be positive")
    if os.path.exists(args.out):
        raise FileExistsError(f"Output already exists: {args.out}")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    bg = (1.0, 1.0, 1.0) if args.white_bg else (0.0, 0.0, 0.0)

    cams = load_cameras(args.frame_dir, args.images)
    test = [c for c in cams if c.name == args.test_cam][0]
    train = [c for c in cams if c.name != args.test_cam]
    extent = nerfpp_radius(train)
    rtrain = [RCam(c) for c in train]
    gts = [to_t(load_image(c.image_path)) for c in train]
    rtest, gt_test = RCam(test), to_t(load_image(test.image_path))

    gs = GS.from_ply(args.ply)
    opt = torch.optim.Adam([
        dict(params=[gs.xyz], lr=1.6e-4 * extent),
        dict(params=[gs.f_dc], lr=2.5e-3),
        dict(params=[gs.f_rest], lr=2.5e-3 / 20),
        dict(params=[gs.opacity], lr=args.opacity_lr),
        dict(params=[gs.scale], lr=5e-3),
        dict(params=[gs.rot], lr=1e-3),
    ], lr=0.0, eps=1e-15)

    log = []
    with torch.no_grad():
        p0 = psnr(render(gs, rtest, bg).clamp(0, 1), gt_test)
    print(f"n={gs.xyz.shape[0]:,} extent={extent:.3f}  iter 0 test PSNR {p0:.2f}")
    log.append(dict(it=0, test_psnr=p0))
    exclusive = gpu_exclusive()
    torch.cuda.synchronize(); t0 = time.time(); eval_sec = 0.0
    order = []
    for it in range(1, args.iters + 1):
        if not order:
            order = list(range(len(rtrain))); random.shuffle(order)
        i = order.pop()
        deg = min(gs.sh_degree, it // 1000) if args.sh_warmup else gs.sh_degree
        img = render(gs, rtrain[i], bg, sh_degree=deg)
        loss = 0.8 * (img - gts[i]).abs().mean() + 0.2 * (1 - ssim_t(img, gts[i]))
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        if it % 250 == 0 or it == args.iters:
            torch.cuda.synchronize(); te = time.time()
            with torch.no_grad():
                p = psnr(render(gs, rtest, bg, sh_degree=deg).clamp(0, 1), gt_test)
            print(f"iter {it}  loss {loss.item():.4f}  test PSNR {p:.2f}", flush=True)
            log.append(dict(it=it, loss=loss.item(), test_psnr=p))
            torch.cuda.synchronize(); eval_sec += time.time() - te
    torch.cuda.synchronize()
    dt = time.time() - t0 - eval_sec          # pure optimization time (test evals excluded)
    save_ply(gs.to_dict(), args.out)
    timing.write(args.out, f"refine_{args.iters}", dt, src_ply=args.ply, exclusive=exclusive,
                 measurement_required=True,
                 init_metadata=dict(timing.load(args.ply).get("init_metadata", {}), refine_iters=args.iters,
                                    refinement_seed=args.seed, refinement_cameras=[camera.name for camera in train],
                                    n_final_gaussians=int(gs.xyz.shape[0])),
                 notes=[f"refine: {args.iters} iters, no densification (time excludes image loading)"])
    if sources_path(args.ply).is_file():
        shutil.copyfile(sources_path(args.ply), sources_path(args.out))
    json.dump(dict(args=vars(args), train_time_sec=dt, extent=extent, log=log,
                   peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9),
              open(os.path.splitext(args.out)[0] + "_refine.json", "w"), indent=1)
    print(f"done in {dt:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
