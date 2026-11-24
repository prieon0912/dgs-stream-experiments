"""[env: any] Per-init M* (update mode) and R* (fast rasterizer) from the 30-frame pilots -> final plan.

Rules (fixed before seeing the pilot-2 results):
  * Compare every run with its own init's mode-A pilot on the COMMON frames 1..30 (EB pilots run 60 frames;
    only their first 30 are used). PSNR = mean cam00 psnr_main; time = mean (t_s1 + t_s2), profiled frames
    (torch.profiler, pilot only) excluded per run.
  * M*: among B, C, EB keep those with PSNR(A) - PSNR(m) <= 0.1 dB and time < time(A); take the fastest.
        If none qualifies, M* = A (then the plan has only A and A+R).
  * R*: DGR_MODE=3 (SnugBox-style rect + warp-reduced backward) passes if PSNR(A) - PSNR(AR) <= 0.1 dB and
        time(AR) < time(A); otherwise there is no R* run.
  Final runs per init (--set 16): A, M*, R* (=A+fast), M*+R* (=M*+fast), duplicates dropped.
  --set 8 (chosen 2026-11-24): only A+R* and M*+R* (if R* passes; otherwise A and M*).
Writes <out>.json (all numbers + decision) and <out>.env (PLAN_<init>="A B AR BR" style).
"""
import argparse
import csv
import json
import math
import os


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def load(run_dir, kmax):
    p = os.path.join(run_dir, "per_frame.csv")
    if not os.path.isfile(p):
        return None
    prof = set()
    s = os.path.join(run_dir, "selection.jsonl")
    if os.path.isfile(s):
        for line in open(s):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("profiled"):
                prof.add(int(r["k"]))
    psnr, t = {}, []
    for r in csv.DictReader(open(p, encoding="utf-8-sig")):
        k = int(float(r["frame"]))
        if k < 1 or k > kmax:
            continue
        v = fnum(r.get("psnr_main"))
        if v is not None:
            psnr[k] = v
        a, b = fnum(r.get("t_s1")), fnum(r.get("t_s2"))
        if k not in prof and a is not None and b is not None:
            t.append(a + b)
    if not psnr or not t:
        return None
    return dict(psnr_by_k=psnr, time=sum(t) / len(t))


def first(store, names):
    for n in names:
        if os.path.isfile(os.path.join(store, n, "per_frame.csv")):
            return n
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--inits", nargs="+", required=True)
    ap.add_argument("--e_every", type=int, required=True)
    ap.add_argument("--kmax", type=int, default=30)
    ap.add_argument("--tol_db", type=float, default=0.1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", choices=["16", "8"], default="8",
                    help="8: per init {A+R*, M*+R*} (fast rasterizer is lossless for images); 16: A, M*, R*, M*+R*")
    a = ap.parse_args()
    P2, PI = "final_s0_p2", "final_s0_pilot"
    report, env, missing = {}, [], []
    for code in a.inits:
        cand = {
            "A": [f"{P2}_{code}_A", f"{PI}_A_{code}"],
            "AR": [f"{P2}_{code}_AR"],
            "B": [f"{P2}_{code}_B", f"{PI}_B_{code}"],
            "C": [f"{P2}_{code}_C", f"{PI}_C_{code}"],
            "EB": [f"{P2}_{code}_EB", f"{PI}_EB{a.e_every}_{code}"],
        }
        runs = {}
        for m, names in cand.items():
            n = first(a.store, names)
            d = load(os.path.join(a.store, n), a.kmax) if n else None
            if d:
                runs[m] = dict(run=n, **d)
        if "A" not in runs:
            missing.append(f"{code}:A")
            continue
        for m in cand:
            if m not in runs:
                missing.append(f"{code}:{m}")
        common = set(runs["A"]["psnr_by_k"])
        for r in runs.values():
            common &= set(r["psnr_by_k"])
        rows = {}
        for m, r in runs.items():
            ps = sum(r["psnr_by_k"][k] for k in common) / len(common)
            rows[m] = dict(run=r["run"], psnr=round(ps, 4), train_sec_per_frame=round(r["time"], 3))
        A = rows["A"]
        for m, r in rows.items():
            r["dpsnr_vs_A"] = round(r["psnr"] - A["psnr"], 4)
            r["speedup_vs_A"] = round(A["train_sec_per_frame"] / r["train_sec_per_frame"], 3)
        ok = [m for m in ("B", "C", "EB") if m in rows and -rows[m]["dpsnr_vs_A"] <= a.tol_db
              and rows[m]["train_sec_per_frame"] < A["train_sec_per_frame"]]
        mstar = min(ok, key=lambda m: rows[m]["train_sec_per_frame"]) if ok else "A"
        r_ok = "AR" in rows and -rows["AR"]["dpsnr_vs_A"] <= a.tol_db and \
            rows["AR"]["train_sec_per_frame"] < A["train_sec_per_frame"]
        plan = ["A"]
        if mstar != "A":
            plan.append(mstar)
        if r_ok:
            plan.append("AR")
            if mstar != "A":
                plan.append(mstar + "R")
        if a.set == "8" and r_ok:
            plan = [v for v in plan if v.endswith("R")]
        report[code] = dict(common_frames=len(common), runs=rows, M_star=mstar, M_qualified=ok,
                            R_star="DGR_MODE=3" if r_ok else None, plan=plan)
        env.append(f'PLAN_{code}="{" ".join(plan)}"')
    out = dict(rules=__doc__, e_every=a.e_every, tol_db=a.tol_db, kmax=a.kmax, missing=missing, inits=report)
    json.dump(out, open(a.out + ".json", "w"), indent=1)
    open(a.out + ".env", "w").write("\n".join(env) + "\n")
    for code, r in report.items():
        print(code, "M*=", r["M_star"], "R*=", r["R_star"], "plan=", r["plan"],
              {m: (v["dpsnr_vs_A"], v["speedup_vs_A"]) for m, v in r["runs"].items()})
    if missing:
        print("MISSING", missing)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
