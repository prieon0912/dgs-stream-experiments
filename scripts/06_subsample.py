"""[env: any] Keep the top-K gaussians by opacity.
  --topk 500000          memory / speed cap (state it in the results)
  --match_ref base.ply   K = #gaussians of the reference (init #4: "count hypothesis")
"""
import argparse
import json
import os
import time

import numpy as np

from common import timing
from common.ply_io import load_ply, save_ply, stats, subset
from common.diagnostics import sources_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--topk", type=int)
    g.add_argument("--match_ref")
    args = ap.parse_args()
    if args.topk is not None and args.topk < 1:
        ap.error("--topk must be positive")
    d = load_ply(args.ply)
    t0 = time.time()
    k = args.topk if args.topk else len(load_ply(args.match_ref)["xyz"])
    if k >= len(d["xyz"]):
        print(f"K={k:,} >= N={len(d['xyz']):,}: nothing removed")
        idx = np.arange(len(d["xyz"]))
    else:
        idx = np.argsort(-d["opacity"])[:k]
    out = subset(d, np.sort(idx))
    save_ply(out, args.out)
    timing.write(args.out, "subsample", time.time() - t0, src_ply=args.ply,
                 init_metadata=dict(timing.load(args.ply).get("init_metadata", {}),
                                    max_gaussians=int(k), n_final_gaussians=int(len(out["xyz"]))),
                 notes=[f"top-{k} by opacity from {os.path.basename(args.ply)}"])
    if sources_path(args.ply).is_file():
        with np.load(sources_path(args.ply), allow_pickle=False) as source:
            np.savez_compressed(sources_path(args.out), view_ids=source["view_ids"][np.sort(idx)],
                                camera_names=source["camera_names"])
    st = stats(out)
    json.dump(dict(src=args.ply, k=int(k), n_src=int(len(d["xyz"])), stats=st),
              open(os.path.splitext(args.out)[0] + "_subsample.json", "w"), indent=1)
    print(f"{len(d['xyz']):,} -> {st['n']:,}  saved {args.out}")


if __name__ == "__main__":
    main()
