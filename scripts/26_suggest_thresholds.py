"""[env: any] Choose the mode-B motion threshold from a short mode-A pilot (motion dumped per frame).

  python scripts/26_suggest_thresholds.py --dump $RUNS/stream/prune_s0_pilot_A/motion \
      [--target-active 0.35 --window 10 --warmup 10 --recheck 30] [--write-env jobs/prune_update_s0/phase2.env.suggested]

Replays mode B offline on the pilot's per-gaussian |d_xyz| (what A updated = everything inside the AABB):
for each candidate tau, frames that B would freeze are hidden from the history exactly as in the real
run, and the mean fraction of inside gaussians still updated on non-full frames is reported.
The suggested tau is the one whose active fraction is closest to --target-active (default 0.35 following
DecoGS, which updates about 35% of the Gaussians; 0.5 is the conservative end). The same tau is then used for all four pruning methods.
"""
import argparse
import glob
import json
import os
import re

import numpy as np


def load_dump(d, key="disp"):
    files = sorted(glob.glob(os.path.join(d, f"frame*_{key}.npy")),
                   key=lambda p: int(re.search(r"frame(\d+)", p).group(1)))
    if not files:
        raise SystemExit(f"no frame*_{key}.npy in {d}")
    ks = [int(re.search(r"frame(\d+)", p).group(1)) for p in files]
    return ks, np.stack([np.load(p).astype(np.float32) for p in files])


def full_frame(k, warmup, recheck):
    return k <= warmup or (recheck > 0 and k % recheck == 0)


def replay(ks, disp, tau, window, warmup, recheck):
    n = disp.shape[1]
    inside = ~np.isnan(disp).all(0)
    buf = np.full((window, n), np.nan, np.float32)
    last = np.full(n, np.inf, np.float32)
    ptr, fracs = 0, []
    for k, d in zip(ks, disp):
        if full_frame(k, warmup, recheck):
            active = inside.copy()
        else:
            seen = ~np.isnan(buf)
            score = np.where(seen.any(0), np.where(seen, buf, -1.0).max(0), last)
            active = (score > tau) & inside
            fracs.append(active.sum() / max(inside.sum(), 1))
        obs = active & ~np.isnan(d)
        buf[ptr] = np.where(obs, d, np.nan)
        last = np.where(obs, d, last)
        ptr = (ptr + 1) % window
    return float(np.mean(fracs)) if fracs else float("nan"), len(fracs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--target-active", type=float, default=0.35)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--recheck", type=int, default=30)
    ap.add_argument("--write-env")
    a = ap.parse_args()
    ks, disp = load_dump(a.dump)
    obs = disp[~np.isnan(disp)]
    qs = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    cands = sorted(set(float(np.quantile(obs, q)) for q in qs))
    table = []
    for tau in cands:
        frac, nf = replay(ks, disp, tau, a.window, a.warmup, a.recheck)
        table.append(dict(tau_xyz=tau, active_frac=frac, n_frozen_frames=nf))
        print(f"tau_xyz={tau:.6g}  mean active fraction on non-full frames {frac:.3f}  ({nf} frames)")
    valid = [t for t in table if np.isfinite(t["active_frac"])]
    if not valid:
        raise SystemExit("pilot too short: no non-full frames (need more than --warmup frames)")
    best = min(valid, key=lambda t: abs(t["active_frac"] - a.target_active))
    print(f"suggest B_TAU_XYZ={best['tau_xyz']:.6g} (active {best['active_frac']:.3f}, target {a.target_active})")
    summary = dict(frames=ks, observed_quantiles={q: float(np.quantile(obs, q)) for q in qs},
                   window=a.window, warmup=a.warmup, recheck=a.recheck, target_active=a.target_active,
                   candidates=table, suggestion=best)
    with open(os.path.join(a.dump, "threshold_suggestion.json"), "w") as f:
        json.dump(summary, f, indent=1)
    if a.write_env:
        with open(a.write_env, "a") as f:
            f.write(f"# from 26_suggest_thresholds.py (target active {a.target_active})\n")
            f.write(f"B_TAU_XYZ={best['tau_xyz']:.6g}\n")
        print("appended to", a.write_env)


if __name__ == "__main__":
    main()
