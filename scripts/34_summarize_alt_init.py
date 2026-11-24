"""[env: any] Third-experiment table: alternative first-frame initializers vs DA3 (+ pruning reference).

  python scripts/34_summarize_alt_init.py --root $ALT_RUNS/init --tag alt_s0 \
      --ref $RUNS/prune_update/init/prune_s0_opacity_1000000 --out $ALT_RUNS/results

Reference row: DA3 all views, opacity top-1,000,000, + 1k refine (experiment 2, same pod, same evaluation).
Same rule as experiment 2: a condition passes tolerance t when BOTH
    cam00 PSNR      >= reference cam00 PSNR      - t      (held-out validation view)
    train-view PSNR >= reference train-view PSNR - t
reported for t = 0.1 / 0.3 / 0.5 dB. Single runs (seed 0): differences of a few 0.1 dB are not
established as real effects by this table alone.

Time = init_time_sec of the final PLY (sum of timed steps; weight loading and COLMAP excluded;
3DGS/Taming wall clock includes start-up, see notes in each *_timing.json).
"""
import argparse
import csv
import json
import re
from pathlib import Path

TOLERANCES = (0.1, 0.3, 0.5)
ORDER = {"da3_opacity": 0, "da3ref": 1, "3dgs15k": 2, "taming": 3, "edgs": 4, "zipsplat": 5}


def load(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def row_from(d, method, param, final="final.ply"):
    static = load(d / "static" / "static.json") or {}
    fin, pre = static.get("final", {}), static.get("pre", {})
    timing = load(d / (Path(final).stem + "_timing.json")) or {}
    extra = load(d / "init_edgs.json") or load(d / "init_zipsplat.json") or {}
    return dict(method=method, param=param, n_final=fin.get("n"),
                psnr=fin.get("psnr"), train_psnr=fin.get("train_psnr_mean"),
                ssim=fin.get("ssim"), lpips=fin.get("lpips"), pre_psnr=pre.get("psnr"),
                inside_count=fin.get("aabb_inside_count"), inside_frac=fin.get("aabb_inside_frac"),
                init_time_sec=timing.get("init_time_sec"), steps=json.dumps(timing.get("steps", {})),
                gpu_exclusive=timing.get("gpu_exclusive"),
                alignment_residual_rel=extra.get("center_residual_rel"),
                rotation_error_deg=extra.get("rotation_error_deg_median"),
                n_raw=extra.get("n_raw") or extra.get("n_gaussians"),
                excluded=(d / "excluded.json").is_file(),
                complete=bool(fin) and fin.get("psnr") is not None, dir=str(d))


def rows_for(root, tag):
    rows = []
    for d in sorted(Path(root).glob(f"{tag}_*")):
        m = re.fullmatch(rf"{re.escape(tag)}_(da3ref|3dgs15k|taming|edgs|zipsplat)_(.+)", d.name)
        if m:
            rows.append(row_from(d, m.group(1), m.group(2)))
    return rows


def passes(r, ref, tol):
    if ref is None or r.get("psnr") is None or r.get("train_psnr") is None:
        return None
    return bool(r["psnr"] >= ref["psnr"] - tol and r["train_psnr"] >= ref["train_psnr"] - tol)


def fmt(x, nd=2, signed=False):
    if x is None:
        return "-"
    return (f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--tag", default="alt_s0")
    ap.add_argument("--ref", help="experiment-2 condition dir of DA3 opacity top-1M + 1k")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = rows_for(args.root, args.tag)
    ref = None
    if args.ref and Path(args.ref, "static", "static.json").is_file():
        ref = row_from(Path(args.ref), "da3_opacity", "1000000", final="refined.ply")
        if ref["psnr"] is None or ref["train_psnr"] is None:
            ref = None
        else:
            rows.insert(0, ref)
    for r in rows:
        r["d_psnr"] = None if ref is None or r["psnr"] is None else r["psnr"] - ref["psnr"]
        r["d_train_psnr"] = None if ref is None or r["train_psnr"] is None else r["train_psnr"] - ref["train_psnr"]
        for t in TOLERANCES:
            r[f"pass_{t}dB"] = passes(r, ref, t)
    rows.sort(key=lambda r: (ORDER.get(r["method"], 9), -(r["n_final"] or 0)))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["method", "param", "n_final", "psnr", "d_psnr", "train_psnr", "d_train_psnr",
            *[f"pass_{t}dB" for t in TOLERANCES], "ssim", "lpips", "pre_psnr", "inside_count", "inside_frac",
            "init_time_sec", "gpu_exclusive", "steps", "alignment_residual_rel", "rotation_error_deg", "n_raw",
            "excluded", "complete", "dir"]
    with open(out / "alt_init.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    json.dump(dict(reference=args.ref, reference_psnr=ref and ref["psnr"],
                   reference_train_psnr=ref and ref["train_psnr"], tolerances_db=TOLERANCES, rows=rows),
              open(out / "alt_init.json", "w"), indent=1)

    if ref:
        print(f"reference DA3 opacity/1M + 1k: cam00 {ref['psnr']:.2f} dB, train views {ref['train_psnr']:.2f} dB")
    print(f"{'method':12s} {'param':>9s} {'#G':>9s} {'cam00':>6s} {'d':>6s} {'train':>6s} {'d':>6s} "
          f"{'0.1':>4s} {'0.3':>4s} {'0.5':>4s} {'inside':>9s} {'time s':>7s}")
    for r in [r for r in rows if r.get("excluded")]:
        print(f"{r['method']:12s} {r['param']:>9s}  excluded: fewer visible points than N (see edgs all)")
    for r in [r for r in rows if not r.get("excluded")]:
        ok = lambda v: "-" if v is None else ("yes" if v else "no")  # noqa: E731
        print(f"{r['method']:12s} {r['param']:>9s} {r['n_final'] or 0:>9,} {fmt(r['psnr']):>6s} "
              f"{fmt(r['d_psnr'], signed=True):>6s} {fmt(r['train_psnr']):>6s} {fmt(r['d_train_psnr'], signed=True):>6s} "
              f"{ok(r['pass_0.1dB']):>4s} {ok(r['pass_0.3dB']):>4s} {ok(r['pass_0.5dB']):>4s} "
              f"{r['inside_count'] or 0:>9,} {fmt(r['init_time_sec'], 1):>7s}")
    chk = next((r for r in rows if r["method"] == "da3ref" and r.get("steps")), None)
    if ref and chk:
        a = json.loads(ref["steps"]).get("refine_1000")
        b = json.loads(chk["steps"]).get("refine_1000")
        if a and b:
            print(f"machine check: the same DA3 1M refine took {b:.1f} s here vs {a:.1f} s in experiment 2 "
                  f"(ratio {b / a:.3f}); timings from the two machines differ by about this factor")
    print("single runs (seed 0); cam00 is the validation view used for selection, not an independent test")


if __name__ == "__main__":
    main()
