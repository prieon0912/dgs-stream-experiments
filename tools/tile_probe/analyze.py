"""Join tile_pairs.csv (probe on pod) with camera_count_results.csv and report correlations.
usage: python analyze.py tile_pairs.csv camera_count_results.csv out.csv"""
import csv
import math
import sys

# camera_count_results label -> substring of the probed PLY path
MAP = {
    "3DGS 15k": "inits/camera15_cap1m_s0_baseline_own.ply",
    "DA3 all views": "shared_all_views/limited.ply",
    "DA3+1k all views": "camera15_cap1m_s0_da3_1k/refined.ply",
}


def path_key(label):
    if label in MAP:
        return MAP[label]
    sel = label.replace("+1k", "")          # e.g. random_12, direction_fps_04
    return f"camera15_cap1m_s0_{sel}_da3_1000/refined.ply"


def pearson(x, y):
    n = len(x); mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x)); sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else float("nan")


def rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2
        i = j + 1
    return r


def main(tp, cc, out):
    probe = [r for r in csv.DictReader(open(tp)) if r["status"] == "ok"]
    rows = []
    for r in csv.DictReader(open(cc)):
        if not r["stream_train_s_per_frame"]:
            continue
        key = path_key(r["label"])
        m = [p for p in probe if p["path"].endswith(key)]
        if not m:
            print("no probe for", r["label"], key); continue
        p = m[0]
        rows.append(dict(label=r["label"], n_gaussians=int(p["n_gaussians"]), aabb_inside=int(r["aabb_inside_count"]),
                         tile_pairs=float(p["num_rendered_mean"]), tile_pairs_train=float(p["num_rendered_train_mean"]),
                         visible=float(p["visible_mean"]), pairs_per_visible=float(p["pairs_per_visible"]),
                         mean_radius=float(p["mean_radius"]), fwdbwd_ms=float(p["fwdbwd_ms_mean"]),
                         stage1_s=float(r["stage1_s_per_frame"]), stage2_s=float(r["stage2_s_per_frame"]),
                         train_s=float(r["stream_train_s_per_frame"])))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} conditions")
    for target in ("train_s", "stage1_s", "stage2_s", "fwdbwd_ms"):
        y = [r[target] for r in rows]
        for feat in ("n_gaussians", "aabb_inside", "tile_pairs", "visible", "fwdbwd_ms"):
            if feat == target:
                continue
            x = [r[feat] for r in rows]
            print(f"{target:9s} vs {feat:12s}  pearson {pearson(x, y):+.3f}  spearman {pearson(rank(x), rank(y)):+.3f}")
    # same analysis without 3DGS 15k (all DA3 rows, mostly 1M gaussians)
    da3 = [r for r in rows if not r["label"].startswith("3DGS")]
    print(f"-- DA3 only ({len(da3)})")
    for feat in ("n_gaussians", "aabb_inside", "tile_pairs", "fwdbwd_ms"):
        x = [r[feat] for r in da3]; y = [r["train_s"] for r in da3]
        print(f"train_s vs {feat:12s}  pearson {pearson(x, y):+.3f}  spearman {pearson(rank(x), rank(y)):+.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
