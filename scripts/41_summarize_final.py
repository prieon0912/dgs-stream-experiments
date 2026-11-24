"""[env: any] Tables of the final experiment from the compact run copies on the shared volume.

  final_runs.csv : one row per main run (init x mode): status, cam00 PSNR/SSIM/LPIPS (stream mean, first/last 30),
                   train s/frame (stage 1, stage 2), wall s/frame, NTC warm-up, selected fraction, adaptive-box
                   fraction, tile pairs, init count, storage
  final_pivot_psnr.csv / final_pivot_time.csv : init x mode
"""
import argparse
import csv
import json
import os


def jl(path):
    out = []
    if os.path.isfile(path):
        for line in open(path):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def mean(v):
    v = [x for x in v if isinstance(x, (int, float))]
    return sum(v) / len(v) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--inits", nargs="+", required=True)
    ap.add_argument("--modes", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for code in a.inits:
        for mode in a.modes:
            name = f"{a.tag}_{code}_{mode}"
            d = os.path.join(a.store, name)
            status = ("done" if os.path.isfile(os.path.join(a.state, name + ".done")) else
                      "failed" if os.path.isfile(os.path.join(a.state, name + ".failed")) else
                      "running" if os.path.isdir(os.path.join(a.state, "claims", name)) else "pending")
            s = {}
            p = os.path.join(d, "run_summary.json")
            if os.path.isfile(p):
                s = json.load(open(p))
            sel = jl(os.path.join(d, "selection.jsonl"))
            t = s.get("time", {})
            st = s.get("stream", {})
            rows.append(dict(
                init=code, mode=mode, run=name, status=status, n_init=s.get("n_init"),
                psnr=st.get("psnr"), psnr_png=st.get("psnr_png"), ssim=st.get("ssim"), lpips=st.get("lpips"),
                first30_psnr=s.get("first30_psnr"), last30_psnr=s.get("last30_psnr"), drift_psnr=s.get("drift_psnr"),
                train_s_per_frame=t.get("stream_train_sec_per_frame"), stage1_s=t.get("stage1_sec_per_frame"),
                stage2_s=t.get("stage2_sec_per_frame"), prep_s=t.get("stream_prep_sec_per_frame"),
                wall_s_per_frame=t.get("stream_wall_sec_per_frame"), ntc_warmup_s=t.get("ntc_warmup_sec"),
                init_s=t.get("init_sec"),
                selected_frac=mean([r["n_selected"] / r["n_inside"] for r in sel if r.get("n_inside")]),
                n_inside_mean=mean([r.get("n_inside") for r in sel]),
                n_selected_mean=mean([r.get("n_selected") for r in sel]),
                box_frac_of_inside=mean([r.get("box_frac_of_inside") for r in sel]),
                fallback_frames=sum(1 for r in sel if r.get("fallback")),
                tile_pairs_mean=mean([r.get("tile_pairs_mean") for r in sel]),
                storage_mb_per_frame=s.get("storage", {}).get("per_frame_payload_mb"),
                peak_gpu_mb=s.get("peak_gpu_mem_mb"), time_reportable=s.get("time_reportable"),
                quality_complete=s.get("quality_data_complete")))
    keys = list(rows[0])
    with open(os.path.join(a.out, "final_runs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    for metric, fn in (("psnr", "final_pivot_psnr.csv"), ("train_s_per_frame", "final_pivot_time.csv")):
        with open(os.path.join(a.out, fn), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["init"] + a.modes)
            for code in a.inits:
                w.writerow([code] + [next((r[metric] for r in rows if r["init"] == code and r["mode"] == m), None)
                                     for m in a.modes])
    print(f"{sum(r['status'] == 'done' for r in rows)}/{len(rows)} done -> {a.out}")


if __name__ == "__main__":
    main()
