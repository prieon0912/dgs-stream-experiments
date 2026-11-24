"""[env: gs] NTC warm-up for a given init PLY. Script version of 3DGStream/scripts/cache_warmup.ipynb
(same AABB, net config, loss, 3000 iters, Adam 1e-4). Run once per init so every init gets the same
treatment (the shipped ntc/flame_steak_ntc_params_F_4.pth was warmed on the baseline init only).

NaN loss (3DGStream issue #16) -> automatic retry with a new seed and lower lr.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

REPO_GS = os.environ.get("REPO_GS", "/workspace/3DGStream")
sys.path.insert(0, REPO_GS)
import tinycudann as tcnn  # noqa: E402
from ntc import NeuralTransformationCache  # noqa: E402

from common.ply_io import load_ply  # noqa: E402
from common.aabb import load_aabb, file_sha256, point_stats  # noqa: E402
from common.timing import gpu_exclusive, gpu_name  # noqa: E402


def quaternion_loss(q1, q2):
    cos = torch.clamp(F.cosine_similarity(q1, q2, dim=1), -1 + 1e-7, 1 - 1e-7)
    return 1 - torch.pow(cos, 2).mean()


def warmup(xyz, conf, iters, lr, seed, aabb_min, aabb_max):
    torch.manual_seed(seed)
    ntc = tcnn.NetworkWithInputEncoding(n_input_dims=3, n_output_dims=8, encoding_config=conf["encoding"],
                                        network_config=conf["network"]).to("cuda")
    optim = torch.optim.Adam(ntc.parameters(), lr=lr)
    nx = (xyz - aabb_min.to(xyz.device)) / (aabb_max.to(xyz.device) - aabb_min.to(xyz.device))
    mask = ((nx >= 0) & (nx <= 1)).all(1)
    inp = nx[mask]
    if not len(inp):
        raise ValueError("AABB contains no Gaussian centers; inspect coordinates/bounds before warm-up")
    noisy = inp + 0.01 * torch.rand_like(inp)
    dxyz_gt = torch.tensor([0., 0., 0.]).cuda()
    drot_gt = torch.tensor([1., 0., 0., 0.]).cuda()
    dummy_gt = torch.tensor([1.]).cuda()
    loss = None
    for it in range(iters):
        out = ntc(torch.cat([noisy, inp, torch.rand_like(inp)], 0))
        loss = (out[:, :3] - dxyz_gt).abs().mean() + quaternion_loss(out[:, 3:7], drot_gt) + \
               (out[:, 7:8] - dummy_gt).abs().mean()
        if not torch.isfinite(loss):
            return None, float("nan"), float(mask.float().mean())
        if it % 500 == 0:
            print(f"  it {it} loss {loss.item():.6f}", flush=True)
        loss.backward(); optim.step(); optim.zero_grad(set_to_none=True)
    return ntc, float(loss.item()), float(mask.float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", default=os.path.join(REPO_GS, "configs/cache/cache_F_4.json"))
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--aabb", default=os.environ.get("AABB_CONFIG"), help="Same AABB JSON as static evaluation")
    args = ap.parse_args()
    if args.iters < 1:
        ap.error("--iters must be positive")
    if os.path.exists(args.out):
        raise FileExistsError(f"NTC already exists: {args.out}; use a new output/run name")
    aabb = load_aabb(args.aabb)
    aabb_min = torch.tensor(aabb["min"], dtype=torch.float32)
    aabb_max = torch.tensor(aabb["max"], dtype=torch.float32)
    with open(args.conf) as f:
        try:
            conf = json.load(f)
        except json.JSONDecodeError:
            import commentjson; f.seek(0); conf = commentjson.load(f)
    points = load_ply(args.ply)["xyz"]
    stats = point_stats(points, aabb)
    if stats["n_nonfinite"] or not stats["n_inside"]:
        raise ValueError("Non-finite points or empty AABB; inspect input PLY and AABB")
    exclusive = gpu_exclusive()
    xyz = torch.tensor(points, dtype=torch.float, device="cuda")
    torch.cuda.synchronize(); t0 = time.time()
    for attempt, (lr, seed) in enumerate([(1e-4, 0), (1e-4, 1), (5e-5, 2), (2e-5, 3)]):
        ntc, final, frac = warmup(xyz, conf, args.iters, lr, seed, aabb_min, aabb_max)
        print(f"attempt {attempt}: lr={lr} seed={seed} final loss={final} inside AABB={100 * frac:.1f}%")
        if ntc is not None:
            break
    else:
        sys.exit("NTC warm-up NaN in all attempts")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(NeuralTransformationCache(ntc, aabb_min, aabb_max).state_dict(), args.out)
    torch.cuda.synchronize()
    sec = time.time() - t0
    print(f"warm-up time {sec:.1f}s (attempts: {attempt + 1})")
    json.dump(dict(ply=args.ply, final_loss=final, inside_aabb=frac, lr=lr, seed=seed,
                   warmup_sec=sec, attempts=attempt + 1, aabb=aabb,
                   aabb_inside_count=stats["n_inside"], n_total=stats["n_total"],
                   gpu=gpu_name(), gpu_exclusive=exclusive,
                   init_sha256=file_sha256(args.ply), ntc_config_sha256=file_sha256(args.conf)),
              open(os.path.splitext(args.out)[0] + "_warmup.json", "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
