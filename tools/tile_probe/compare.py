"""Compare rasterizer variants against mode 0 from rbench outputs.
  python compare.py --dir /workspace/rbench --labels opacity1m redundancy600k --modes 1 2 3
Writes <dir>/rbench_compare.csv: per label/mode: speed-up (fwd, bwd, fwd+bwd), pairs ratio,
image PSNR vs mode 0 and max abs diff, relative L2 error of each gradient vs mode 0.
"""
import argparse
import csv
import json
import math
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--modes", nargs="+", type=int, default=[1, 2, 3])
    a = ap.parse_args()
    summ = {}
    for line in open(os.path.join(a.dir, "rbench.jsonl")):
        s = json.loads(line)
        summ[(s["label"], s["mode"])] = s          # last run wins
    rows = []
    for lab in a.labels:
        ref = torch.load(os.path.join(a.dir, f"{lab}_m0.pt"))
        s0 = summ[(lab, 0)]
        for m in a.modes:
            p = os.path.join(a.dir, f"{lab}_m{m}.pt")
            if not os.path.isfile(p) or (lab, m) not in summ:
                continue
            cur = torch.load(p)
            s = summ[(lab, m)]
            mse = max(float(((x["image"] - y["image"]) ** 2).mean()) for x, y in zip(ref, cur))
            mad = max(float((x["image"] - y["image"]).abs().max()) for x, y in zip(ref, cur))
            row = dict(label=lab, mode=m, n=s["n"], pairs_ratio=s["pairs"] / s0["pairs"],
                       fwd_speedup=s0["fwd_ms"] / s["fwd_ms"], bwd_speedup=s0["bwd_ms"] / s["bwd_ms"],
                       fwdbwd_speedup=s0["fwdbwd_ms"] / s["fwdbwd_ms"],
                       fwd_ms=s["fwd_ms"], bwd_ms=s["bwd_ms"], ref_fwd_ms=s0["fwd_ms"], ref_bwd_ms=s0["bwd_ms"],
                       image_psnr_vs_ref=(99.0 if mse == 0 else 10 * math.log10(1.0 / mse)), image_maxabs=mad)
            for k in ("xyz", "opacity", "scale", "rot", "f_dc"):
                num = sum(float((x["grads"][k] - y["grads"][k]).norm()) for x, y in zip(ref, cur))
                den = sum(float(x["grads"][k].norm()) for x in ref) + 1e-30
                row[f"grad_relerr_{k}"] = num / den
            rows.append(row)
    if rows:
        with open(os.path.join(a.dir, "rbench_compare.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))


if __name__ == "__main__":
    main()
