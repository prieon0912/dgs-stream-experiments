"""[env: any] Phase-2 table: pruning method x update mode (quality, speed, NTC input size).

  python scripts/25_summarize_phase2.py --runs prune_s0_opacity_600000_A ... --tag prune_s0 --out $RESULTS

Reads <run>/run_summary.json (09_eval_stream.py) and <run>/selection.jsonl (train_frames_select.py).
Speed columns (main metric = train_sec_per_frame):
  train_sec_per_frame     mean of stage-1 + stage-2 time per frame (excludes data/model loading, so storage
                          hiccups do not affect it); selection (A/B/C) runs before stage 1 and is reported
                          separately as select_sec_mean, so add it when comparing modes
  wall_sec_per_frame      upstream "Frame ... finished in" time (includes selection and bookkeeping)
  algo_sec_per_frame      wall minus io_sec (mode-B resume-state save, not part of the algorithm)
  stage1_sec_per_frame    NTC training only; selection runs before it (counted in t_prep)
"""
import argparse
import csv
import json
import os
import re
from pathlib import Path


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None


def selection_stats(path):
    rows = []
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    last = {}
    for r in rows:                               # a resumed frame may appear twice: keep the last line
        last[r["k"]] = r
    rows = [last[k] for k in sorted(last)]
    if not rows:
        return {}
    sel_frac = [r["n_selected"] / r["n_inside"] for r in rows if r.get("n_inside")]
    partial = [r for r in rows if not r.get("full_update")]
    return dict(sel_frames=len(rows), n_base=rows[-1].get("n_base"), n_inside_mean=mean(r.get("n_inside") for r in rows),
                n_selected_mean=mean(r.get("n_selected") for r in rows), selected_frac_mean=mean(sel_frac),
                selected_frac_partial_frames=mean(r["n_selected"] / r["n_inside"] for r in partial if r.get("n_inside")),
                full_update_frames=sum(bool(r.get("full_update")) for r in rows),
                fallback_frames=sum(bool(r.get("fallback")) for r in rows),
                select_sec_mean=mean(r.get("select_sec") for r in rows),
                bookkeeping_sec_mean=mean(r.get("bookkeeping_sec") for r in rows),
                io_sec_mean=mean(r.get("io_sec") for r in rows),
                disp_q50_mean=mean(r.get("disp_q50") for r in rows),
                mask_frac_mean=mean(r.get("mask_frac_mean") for r in rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--runs_dir", default=os.path.join(os.environ.get("RUNS", "/workspace/runs"), "stream"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for name in a.runs:
        m = re.fullmatch(rf"{re.escape(a.tag)}_(.+)_(\d+)_([ABC])", name)
        method, n, mode = (m.group(1), int(m.group(2)), m.group(3)) if m else (None, None, None)
        run = Path(a.runs_dir) / name
        s = json.loads((run / "run_summary.json").read_text()) if (run / "run_summary.json").is_file() else {}
        t = s.get("time", {})
        sel = selection_stats(run / "selection.jsonl")
        wall = t.get("stream_wall_sec_per_frame")
        row = dict(run=name, method=method, n=n, mode=mode,
                   complete=(s.get("completion") or {}).get("complete"),
                   psnr=(s.get("stream") or {}).get("psnr"), ssim=(s.get("stream") or {}).get("ssim"),
                   lpips=(s.get("stream") or {}).get("lpips"), drift_psnr=s.get("drift_psnr"),
                   train_sec_per_frame=t.get("stream_train_sec_per_frame"),
                   train_plus_select_sec_per_frame=(t["stream_train_sec_per_frame"] + sel["select_sec_mean"]
                                                    if t.get("stream_train_sec_per_frame") is not None
                                                    and sel.get("select_sec_mean") is not None else None),
                   wall_sec_per_frame=wall,
                   algo_sec_per_frame=(wall - sel["io_sec_mean"]) if wall is not None and sel.get("io_sec_mean") is not None else wall,
                   stage1_sec_per_frame=t.get("stage1_sec_per_frame"), stage2_sec_per_frame=t.get("stage2_sec_per_frame"),
                   prep_sec_per_frame=t.get("stream_prep_sec_per_frame"), ntc_warmup_sec=t.get("ntc_warmup_sec"),
                   storage_mb_per_frame=(s.get("storage") or {}).get("per_frame_payload_mb"), **sel)
        rows.append(row)
    os.makedirs(a.out, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(a.out, "phase2.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    json.dump(rows, open(os.path.join(a.out, "phase2.json"), "w"), indent=1)
    f2 = lambda x, p=2: "-" if x is None else f"{x:.{p}f}"
    for r in rows:
        print(f"{r['run']:40s} PSNR {f2(r['psnr'])}  train {f2(r['train_sec_per_frame'])}s (+sel {f2(r.get('select_sec_mean'), 3)}s)  "
              f"NTC input {f2(r.get('selected_frac_mean'), 3)} of inside")
    print("wrote", os.path.join(a.out, "phase2.csv"))


if __name__ == "__main__":
    main()
