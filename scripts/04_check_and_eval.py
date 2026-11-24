"""[env: gs] Static check of one or more init PLYs on frame 0.

For every --ply name=path:
  * stats: count, xyz 1/99 percentiles, fraction inside the NTC AABB (hard-coded in 3DGStream's warm-up)
  * cam00 render -> PSNR/SSIM/LPIPS vs GT   (+ mean PSNR over train views with --train)
  * PNG panel: GT | baseline | this | 50% blend(baseline, this)   <- eyeball alignment here

Usage:
  python scripts/04_check_and_eval.py --frame_dir $FRAME0 --out $RUNS/static \
      --ply baseline=$BASE_INIT_PLY --ply da3_raw=$RUNS/inits/da3_raw.ply --train
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from common import timing
from common.aabb import load_aabb, inside_mask
from common.colmap_io import load_cameras, load_image
from common.gs_render import GS, RCam, render
from common.metrics import all_metrics, psnr, to_t
from common.ply_io import load_ply, stats
from common.result_io import write_json, write_csv


def to_png(t):
    return np.rint(t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame_dir", required=True)
    ap.add_argument("--images", default="images_2")
    ap.add_argument("--test_cam", default="cam00")
    ap.add_argument("--ply", action="append", required=True, help="name=path (first one is the reference for the blend)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", action="store_true", help="also mean PSNR over train views (fit vs. generalization)")
    ap.add_argument("--white_bg", action="store_true")
    ap.add_argument("--aabb", default=os.environ.get("AABB_CONFIG"), help="Shared AABB JSON")
    args = ap.parse_args()
    aabb = load_aabb(args.aabb)
    os.makedirs(args.out, exist_ok=True)
    bg = (1.0, 1.0, 1.0) if args.white_bg else (0.0, 0.0, 0.0)

    cams = load_cameras(args.frame_dir, args.images)
    test = [c for c in cams if c.name == args.test_cam][0]
    train = [c for c in cams if c.name != args.test_cam]
    gt = to_t(load_image(test.image_path))

    results, ref_img = {}, None
    for item in args.ply:
        name, path = item.split("=", 1)
        d = load_ply(path)
        st = stats(d)
        inside = inside_mask(d["xyz"], aabb)
        st["aabb_inside_frac"] = float(inside.mean())
        st["aabb_inside_count"] = int(inside.sum())
        st["aabb_outside_count"] = int(len(inside) - inside.sum())
        st["aabb"] = aabb
        st["quantile_ok"] = bool(st["n"] * 3 <= 2 ** 24)
        st["size_mb"] = os.path.getsize(path) / 1e6
        tj = timing.load(path)
        st["init_time_sec"], st["init_steps"] = tj.get("init_time_sec"), tj.get("steps")
        gs = GS(d)
        with torch.no_grad():
            torch.cuda.synchronize(); t0 = time.time()
            img = render(gs, RCam(test), bg)
            torch.cuda.synchronize()
            render_ms = 1000 * (time.time() - t0)
            clipped = img.clamp(0, 1)
            m = all_metrics(torch.round(clipped * 255) / 255, gt)
            m["psnr_png"] = m["psnr"]
            m["psnr"] = psnr(clipped, gt)
            m["psnr_source"] = "float_render"
            m["ssim_lpips_source"] = "8bit_render"
            m["render_ms"] = render_ms
            if args.train:
                m["train_psnr_mean"] = float(np.mean([psnr(render(gs, RCam(c), bg).clamp(0, 1),
                                                           to_t(load_image(c.image_path))) for c in train]))
        results[name] = dict(path=path, **st, **m)
        if ref_img is None:
            ref_img = img
        panel = np.concatenate([to_png(gt), to_png(ref_img), to_png(img), to_png(0.5 * ref_img + 0.5 * img)], 1)
        Image.fromarray(panel).save(os.path.join(args.out, f"{name}_cam00_panel.png"))
        Image.fromarray(to_png(img)).save(os.path.join(args.out, f"{name}_cam00.png"))
        print(f"[{name}] n={st['n']:,}  AABB inside {100 * st['aabb_inside_frac']:.1f}%  "
              f"PSNR {m['psnr']:.2f} SSIM {m['ssim']:.4f} LPIPS {m['lpips']:.4f}"
              + (f"  trainPSNR {m['train_psnr_mean']:.2f}" if args.train else ""))
        print(f"   NTC candidates: {st['aabb_inside_count']:,} inside, {st['aabb_outside_count']:,} outside; "
              "outside points remain static (a high inside fraction does not imply all points move)")
        if not st["quantile_ok"]:
            print("   WARN: too many gaussians for torch.quantile in 3DGStream -> 07_subsample.py --topk")
        del gs; torch.cuda.empty_cache()

    path = os.path.join(args.out, "static.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    old.update(results)
    write_json(path, old)
    write_csv(os.path.join(args.out, "static.csv"), [dict(name=name, **result) for name, result in old.items()])
    print("saved", path)


if __name__ == "__main__":
    main()
