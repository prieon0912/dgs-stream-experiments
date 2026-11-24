"""[env: any] After the pilot: write final.env (B_TAU_XYZ, E_EVERY) for the 16 main runs + a pilot report.

Rule for E_EVERY (fixed before seeing any result):
  among the EB pilot runs (61 frames, same init), keep those whose mean cam00 PSNR over frames 1..60 is
  within 0.1 dB of the best one, then take the one with the smallest mean train time (stage 1 + stage 2)
  per frame. Profiled frames (torch.profiler, pilot only) are excluded from the time.
  If only one EB pilot finished, it is used; if none finished, E_EVERY=10 (noted in the report).
The report also lists every pilot run's PSNR / time / selected fraction, so the 300-frame time of each
main run can be estimated.
"""
import argparse
import csv
import json
import math
import os


def read_rows(run_dir):
    p = os.path.join(run_dir, "per_frame.csv")
    if not os.path.isfile(p):
        return None
    rows = [r for r in csv.DictReader(open(p, encoding="utf-8-sig")) if int(float(r["frame"])) >= 1]
    return rows


def profiled_frames(run_dir):
    p = os.path.join(run_dir, "selection.jsonl")
    out, sel = set(), []
    if os.path.isfile(p):
        for line in open(p):
            r = json.loads(line)
            if r.get("profiled"):
                out.add(int(r["k"]))
            sel.append(r)
    return out, sel


def f(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def stats(run_dir):
    rows = read_rows(run_dir)
    if not rows:
        return None
    prof, sel = profiled_frames(run_dir)
    psnr = [f(r.get("psnr_main")) for r in rows]
    psnr = [v for v in psnr if v is not None]
    t = []
    for r in rows:
        k = int(float(r["frame"]))
        a, b = f(r.get("t_s1")), f(r.get("t_s2"))
        if k not in prof and a is not None and b is not None:
            t.append(a + b)
    fr = [s["n_selected"] / s["n_inside"] for s in sel if s.get("n_inside")]
    pairs = [s["tile_pairs_mean"] for s in sel if s.get("tile_pairs_mean")]
    return dict(frames=len(rows), psnr=sum(psnr) / len(psnr) if psnr else None,
                train_sec_per_frame=sum(t) / len(t) if t else None,
                selected_frac_mean=sum(fr) / len(fr) if fr else None,
                tile_pairs_mean=sum(pairs) / len(pairs) if pairs else None,
                profiled_frames=sorted(prof))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--pilot-prefix", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--candidates", nargs="+", type=int, default=[10, 30])
    ap.add_argument("--pilot-env", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--tol-db", type=float, default=0.1)
    a = ap.parse_args()

    report = dict(rule=__doc__.split("Rule for E_EVERY")[1].split("The report")[0].strip(), runs={})
    for d in sorted(os.listdir(a.store)):
        if d.startswith(a.pilot_prefix):
            report["runs"][d] = stats(os.path.join(a.store, d))
    eb = {}
    for n in a.candidates:
        s = report["runs"].get(f"{a.pilot_prefix}_EB{n}_{a.init}")
        if s and s["psnr"] is not None and s["train_sec_per_frame"] is not None:
            eb[n] = s
    if eb:
        best = max(s["psnr"] for s in eb.values())
        ok = {n: s for n, s in eb.items() if best - s["psnr"] <= a.tol_db}
        choice = min(ok, key=lambda n: ok[n]["train_sec_per_frame"])
        report["decision"] = dict(E_EVERY=choice, best_psnr=best, within_tol=sorted(ok), tol_db=a.tol_db)
    else:
        choice = 10
        report["decision"] = dict(E_EVERY=choice, note="no EB pilot finished -> default 10")
    tau = None
    if os.path.isfile(a.pilot_env):
        for line in open(a.pilot_env):
            if line.startswith("B_TAU_XYZ="):
                tau = line.strip().split("=", 1)[1]
    report["decision"]["B_TAU_XYZ"] = tau
    # rough 300-frame estimate per mode from the pilot (same init); other inits scale by their own count
    est = {}
    for name, s in report["runs"].items():
        if s and s["train_sec_per_frame"]:
            est[name] = round(s["train_sec_per_frame"] * 299 / 3600, 2)
    report["estimate_hours_train_only_300_frames_opacity1m"] = est
    tmp = a.out + ".tmp"
    with open(tmp, "w") as fh:
        if tau:
            fh.write(f"B_TAU_XYZ={tau}\n")
        fh.write(f"E_EVERY={choice}\n")
    os.replace(tmp, a.out)
    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    json.dump(report, open(a.report, "w"), indent=1)
    print(json.dumps(report["decision"]))


if __name__ == "__main__":
    main()
