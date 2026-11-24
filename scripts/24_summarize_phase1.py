"""[env: any] Phase-1 table (pruning method x the counts you tried, frame 0 only) and the suggested N per method.

  python scripts/24_summarize_phase1.py --root $RUNS/prune_update --tag prune_s0 --out $RUNS/prune_update/results \
      [--tolerance-db 0.3] [--write-env jobs/prune_update_s0/phase2.env.suggested]

Reference: opacity top-1,000,000 (= the current pipeline), after the same 1k refine.
A condition passes tolerance t when BOTH
    cam00 PSNR      >= reference cam00 PSNR      - t      (held-out view)
    train-view PSNR >= reference train-view PSNR - t      (mean of the 20 training views; less noisy than one view)
Pass/fail is reported for t = 0.1 / 0.3 / 0.5 dB (literature: ~0.1-0.3 dB is treated as "quality kept",
~0.5 dB as a visible loss). The suggested N per method = smallest TESTED count passing --tolerance-db.
Only counts you actually ran are considered: the trend table shows whether to try larger or smaller N next.
"""
import argparse
import csv
import json
import os
import re
from pathlib import Path

from common.selection import METHODS

REF = ("opacity", 1_000_000)
TOLERANCES = (0.1, 0.3, 0.5)


def load(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def rows_for(root, tag):
    rows = []
    for d in sorted(Path(root, "init").glob(f"{tag}_*")):
        m = re.fullmatch(rf"{re.escape(tag)}_(.+)_(\d+)", d.name)
        if not m or m.group(1) not in METHODS:
            continue
        method, n = m.group(1), int(m.group(2))
        sel = load(d / "selected_select.json") or {}
        ref = load(d / "refined_refine.json") or {}
        static = load(d / "static" / "static.json") or {}
        fin, pre = static.get("final", {}), static.get("pre", {})
        rows.append(dict(method=method, n_target=n, n_final=fin.get("n"),
                         select_sec=sel.get("total_sec"), score_sec=sel.get("score_sec"),
                         voxel_size=(sel.get("method_info") or {}).get("voxel_size"),
                         refine_sec=ref.get("train_time_sec"),
                         pre_psnr=pre.get("psnr"), psnr=fin.get("psnr"), ssim=fin.get("ssim"), lpips=fin.get("lpips"),
                         train_psnr=fin.get("train_psnr_mean"),
                         inside_count=fin.get("aabb_inside_count"), inside_frac=fin.get("aabb_inside_frac"),
                         render_ms=fin.get("render_ms"), complete=bool(fin) and fin.get("psnr") is not None,
                         dir=str(d)))
    return rows


def passes(r, ref, tol):
    if ref is None or r.get("psnr") is None or r.get("train_psnr") is None:
        return None
    return bool(r["psnr"] >= ref["psnr"] - tol and r["train_psnr"] >= ref["train_psnr"] - tol)


def evaluate(rows, tol):
    ref = next((r for r in rows if (r["method"], r["n_target"]) == REF and r["complete"]
                and r.get("train_psnr") is not None), None)
    for r in rows:
        if ref is not None and r["complete"]:
            r["d_psnr"] = r["psnr"] - ref["psnr"]
            r["d_train_psnr"] = None if r.get("train_psnr") is None else r["train_psnr"] - ref["train_psnr"]
        for t in TOLERANCES:
            r[f"pass_{t}dB"] = passes(r, ref, t)
    sug = {}
    for method in METHODS:
        mine = sorted([r for r in rows if r["method"] == method and r["complete"]], key=lambda r: r["n_target"])
        if not mine:
            continue
        if ref is None:
            sug[method] = dict(n=None, reason="reference opacity/1M missing")
            continue
        ok = [r for r in mine if passes(r, ref, tol)]
        if ok:
            smallest = ok[0]["n_target"]
            below = [r["n_target"] for r in mine if r["n_target"] < smallest]
            hint = "try a smaller N" if not below else "a smaller tested N failed -> boundary is between them"
            sug[method] = dict(n=smallest, reason=f"smallest tested N within {tol} dB on cam00 and train views; {hint}")
        else:
            sug[method] = dict(n=None, reason=f"no tested N within {tol} dB -> try a larger N")
    return ref, sug


def fmt(x, p=2, signed=False):
    if x is None:
        return "-"
    return f"{x:+.{p}f}" if signed else f"{x:.{p}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tolerance-db", type=float, default=0.3)
    ap.add_argument("--write-env")
    a = ap.parse_args()
    rows = rows_for(a.root, a.tag)
    ref, sug = evaluate(rows, a.tolerance_db)
    order = lambda r: (METHODS.index(r["method"]), -r["n_target"])
    rows.sort(key=order)
    os.makedirs(a.out, exist_ok=True)
    cols = ["method", "n_target", "n_final", "psnr", "d_psnr", "train_psnr", "d_train_psnr",
            *[f"pass_{t}dB" for t in TOLERANCES], "ssim", "lpips", "pre_psnr", "inside_count", "inside_frac",
            "select_sec", "score_sec", "refine_sec", "voxel_size", "render_ms", "complete", "dir"]
    with open(os.path.join(a.out, "phase1.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    json.dump(dict(reference=REF, reference_psnr=ref and ref["psnr"], reference_train_psnr=ref and ref["train_psnr"],
                   tolerance_db=a.tolerance_db, tolerances_reported=TOLERANCES, suggestion=sug, rows=rows),
              open(os.path.join(a.out, "phase1.json"), "w"), indent=1)

    if ref:
        print(f"reference opacity/1M: cam00 {ref['psnr']:.2f} dB, train views {ref['train_psnr']:.2f} dB")
    else:
        print("reference opacity/1M not finished yet: deltas and pass/fail are empty")
    print(f"{'method':15s} {'N':>9s} {'cam00':>6s} {'d':>6s} {'train':>6s} {'d':>6s}  0.1/0.3/0.5  {'inside':>9s} {'sel s':>6s}")
    mark = lambda v: "-" if v is None else ("o" if v else "x")
    for r in rows:
        print(f"{r['method']:15s} {r['n_target']:>9,} {fmt(r['psnr']):>6s} {fmt(r.get('d_psnr'), signed=True):>6s} "
              f"{fmt(r.get('train_psnr')):>6s} {fmt(r.get('d_train_psnr'), signed=True):>6s}  "
              f"  {'/'.join(mark(r[f'pass_{t}dB']) for t in TOLERANCES):9s}  {r['inside_count'] or 0:>9,} "
              f"{fmt(r['select_sec'], 1):>6s}")
    for m, s in sug.items():
        print(f"suggest {m}: N={s['n'] if s['n'] is None else format(s['n'], ',')}  ({s['reason']})")
    if a.write_env:
        keep = []                                   # lines written by 26_suggest_thresholds.py survive a rerun
        if os.path.isfile(a.write_env):
            keep = [l for l in open(a.write_env).read().splitlines()
                    if l.strip() and not l.startswith(("N_", "# N_")) and "24_summarize_phase1" not in l]
        with open(a.write_env, "w") as f:
            f.write("# suggested by 24_summarize_phase1.py -- review, then copy to phase2.env\n")
            for m in METHODS:
                if m in sug:
                    f.write(f"N_{m}={sug[m]['n']}\n" if sug[m]["n"] else f"# N_{m}= (none passed: {sug[m]['reason']})\n")
            for l in keep:
                f.write(l + "\n")
        print("wrote", a.write_env)


if __name__ == "__main__":
    main()
