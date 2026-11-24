"""[env: gs] Pipeline comparison: table + figures + interpretation draft. Everything is read
automatically from each run's per_frame.csv / run_summary.json (run 09 first).

  python scripts/10_report.py --runs baseline_own:"3DGS 15k" da3_raw:"DA3" da3_1k:"DA3+1k" --out $RUNS/report

The FIRST run is the reference for gaps.  Outputs in --out:
  summary.csv / summary.md   one row per pipeline (time, init quality, streaming quality, storage, memory)
  fig_per_frame.png          PSNR / SSIM / LPIPS / train-time per frame (frame 0 = init, marked)
  fig_time_quality.png       init time (log) vs streaming mean PSNR
"""
import argparse
import json
import os

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common.result_io import tri_all

# categorical slots in fixed order (validated: CVD/normal-vision pass; contrast WARN -> direct labels + table)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
STYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fmt(x, nd=2):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help='run or run:"label"')
    ap.add_argument("--runs_dir", default=os.path.join(os.environ.get("RUNS", "/workspace/runs"), "stream"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-unfinished", action="store_true",
                    help="Leave out runs without a complete stream (e.g. skipped after a failed DA3 alignment)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs = []
    for i, item in enumerate(args.runs):
        name, label = (item.split(":", 1) + [item])[:2]
        R = os.path.join(args.runs_dir, name)
        if args.skip_unfinished:
            try:
                ready = json.load(open(os.path.join(R, "run_summary.json")))
                ready = ready.get("completion", {}).get("complete") and ready.get("quality_data_complete") is True
            except (OSError, ValueError):
                ready = False
            if not ready:
                print(f"SKIPPED in figures: {name}")
                continue
        df = pd.read_csv(os.path.join(R, "per_frame.csv")).sort_values("frame")
        s = json.load(open(os.path.join(R, "run_summary.json")))
        if not s.get("completion", {}).get("complete") or s.get("quality_data_complete") is not True:
            raise ValueError(f"Incomplete run cannot enter formal report: {name}")
        timed = json.load(open(os.path.join(R, "init_timing.json")))
        meta = timed.get("init_metadata", {})
        method = "3dgs" if meta.get("model") == "3dgs" else meta.get("selection", {}).get("method", "all_views")
        colors = {"random": "#2a78d6", "position_fps": "#eb6834", "direction_fps": "#1baf7a",
                  "all_views": "#715aa1", "3dgs": "#333333"}
        count = meta.get("n_views")
        style = {4: "-", 8: "--", 12: "-.", 18: ":"}.get(count, "--" if meta.get("refine_iters") else "-")
        runs.append(dict(name=name, label=label, df=df, s=s, c=colors.get(method, COLORS[i % len(COLORS)]),
                         m=MARKERS[i % len(MARKERS)], ls=style))

    # ---------------- table
    rows = []
    for r in runs:
        s, t, f0 = r["s"], r["s"]["time"], r["df"].iloc[0]
        rows.append({
            "pipeline": r["label"], "run": r["name"],
            "init_time_s": t.get("init_sec"), "init_n": s.get("n_init"), "init_mb": s["storage"]["init_mb"],
            "init_psnr": f0.psnr_main, "init_ssim": f0.ssim, "init_lpips": f0.lpips,
            "ntc_warmup_s": t.get("ntc_warmup_sec"),
            "stream_train_s_per_frame": t.get("stream_train_sec_per_frame"),
            "stream_prep_s_per_frame": t.get("stream_prep_sec_per_frame"),
            "stream_wall_s_total": t.get("stream_wall_sec_total"),
            "end_to_end_s": t.get("end_to_end_sec"),
            "psnr_1toN": s["stream"]["psnr"], "ssim_1toN": s["stream"]["ssim"], "lpips_1toN": s["stream"]["lpips"],
            "psnr_0toN": s["all"]["psnr"], "ssim_0toN": s["all"]["ssim"], "lpips_0toN": s["all"]["lpips"],
            "psnr_first30": s["first30_psnr"], "psnr_last30": s["last30_psnr"], "psnr_drift": s["drift_psnr"],
            "storage_paper_avg_mb": s["storage"]["paper_avg_mb"], "per_frame_payload_mb": s["storage"]["per_frame_payload_mb"],
            "peak_gpu_mem_mb": s.get("peak_gpu_mem_mb"), "frames": s["all"]["frames"],
            "time_reportable": s.get("time_reportable") is True,
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(args.out, "summary.csv"), index=False, encoding="utf-8-sig")

    # ---------------- figure 1: per-frame curves (one measure per panel, shared x)
    fig, axs = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    panels = [("psnr_main", "PSNR [dB] ↑", axs[0, 0]), ("ssim", "SSIM ↑", axs[0, 1]),
              ("lpips", "LPIPS ↓", axs[1, 0]), ("train", "3DGStream train time / frame [s] (S1+S2)", axs[1, 1])]
    for key, title, ax in panels:
        style_axes(ax)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        for r in runs:
            df = r["df"]
            if key == "train":
                if "t_s1" not in df:
                    continue
                d = df[df.frame >= 1]
                x, y = d.frame, d.t_s1 + d.t_s2
            else:
                d = df.dropna(subset=[key])
                x, y = d.frame, d[key]
            sparse = key == "lpips" and len(d) < 0.5 * len(df)   # LPIPS every k frames -> show points
            ax.plot(x, y, color=r["c"], lw=2 if not sparse else 1.2, ls=r["ls"],
                    marker=r["m"] if sparse else None, ms=4, label=r["label"])
            if key != "train":
                f0 = df.iloc[0]
                ax.plot([0], [f0[key]], marker=r["m"], ms=8, color=r["c"], mec="white", mew=2, zorder=5)
            if len(y) and key == "psnr_main" and len(runs) <= 6:
                ax.annotate(r["label"], (x.iloc[-1], y.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                            fontsize=8, color=INK, va="center")
    for ax in axs[1]:
        ax.set_xlabel("frame (0 = init)", fontsize=9, color=MUTED)
    axs[0, 0].margins(x=0.12)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, loc="lower center", ncol=min(3, len(runs)))
    fig.suptitle("flame_steak · cam00 · per-frame quality and time by init pipeline (big marker = frame 0 / init)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, min(.25, .04 * ((len(runs) + 2) // 3)), 1, 0.95))
    fig.savefig(os.path.join(args.out, "fig_per_frame.png"), dpi=160)
    plt.close(fig)

    # ---------------- figure 2: init time vs streaming quality
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    style_axes(ax); ax.grid(axis="x", color=GRID, lw=0.8)
    skipped = []
    for r, row in zip(runs, rows):
        if row["init_time_s"] is None or (isinstance(row["init_time_s"], float) and np.isnan(row["init_time_s"])):
            skipped.append(r["label"]); continue
        ax.scatter(row["init_time_s"], row["psnr_1toN"], s=70, color=r["c"], marker=r["m"],
                   edgecolor="white", linewidth=2, zorder=3, label=r["label"])
        if len(runs) > 6:
            continue
        ax.annotate(f"{r['label']}\n{row['init_time_s']:.0f}s · {row['psnr_1toN']:.2f}dB",
                    (row["init_time_s"], row["psnr_1toN"]), xytext=(8, -2), textcoords="offset points",
                    fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.margins(x=0.35, y=0.2)
    ax.set_xlabel("init time [s, log] (poses given, weights pre-loaded)", fontsize=9, color=MUTED)
    ax.set_ylabel("streaming mean PSNR, frames 1..N-1 [dB]", fontsize=9, color=MUTED)
    ax.set_title("Init time vs streaming quality", fontsize=10, color=INK, loc="left")
    if skipped:
        ax.text(0.01, 0.01, "no init time: " + ", ".join(skipped), transform=ax.transAxes, fontsize=7, color=MUTED)
    if len(runs) > 6:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_time_quality.png"), dpi=160)
    plt.close(fig)

    # ---------------- markdown table + interpretation draft
    cols = [("pipeline", "pipeline", 0), ("init_time_s", "init time [s]", 1), ("init_n", "#G init", 0),
            ("init_psnr", "init PSNR", 2), ("init_ssim", "init SSIM", 4), ("init_lpips", "init LPIPS", 4),
            ("ntc_warmup_s", "NTC warm-up [s]", 1), ("stream_train_s_per_frame", "train s/frame", 2),
            ("stream_wall_s_total", "stream wall [s]", 0), ("psnr_1toN", "PSNR 1..N", 2),
            ("ssim_1toN", "SSIM 1..N", 4), ("lpips_1toN", "LPIPS 1..N", 4), ("psnr_0toN", "PSNR 0..N", 2),
            ("psnr_drift", "drift (last30-first30)", 2), ("storage_paper_avg_mb", "MB/frame", 2),
            ("peak_gpu_mem_mb", "peak GPU [MB]", 0), ("time_reportable", "time OK", 0)]
    md = ["| " + " | ".join(h for _, h, _ in cols) + " |", "|" + "---|" * len(cols)]
    for row in rows:
        md.append("| " + " | ".join(str(row[k]) if nd == 0 and not isinstance(row[k], float)
                                     else (fmt(row[k], nd) if nd else fmt(row[k], 0)) for k, _, nd in cols) + " |")
    base = rows[0]
    lines = ["# Init pipeline comparison (flame_steak, cam00)", "", *md, "",
             "Notes: init time excludes COLMAP/poses (shared by all) and model weight loading. "
             "PSNR = float PSNR (3DGStream's own value for frames >= 1); SSIM/LPIPS from 8-bit images for all frames. "
             "'time OK' = False means timing/exclusivity was not fully verified or a timed step used a shared GPU.", "",
             "## Interpretation draft (edit!)", ""]
    for row in rows[1:]:
        g0 = row["init_psnr"] - base["init_psnr"]
        gs = row["psnr_1toN"] - base["psnr_1toN"]
        gl = (row["psnr_last30"] - base["psnr_last30"]
              if row["psnr_last30"] is not None and base["psnr_last30"] is not None else None)
        tr = (f"{row['init_time_s'] / base['init_time_s']:.2f}x the init time"
              if row["init_time_s"] and base["init_time_s"] else "init time n/a")
        lines.append(f"- **{row['pipeline']}** vs {base['pipeline']}: {tr}; init PSNR {g0:+.2f} dB, streaming "
                     f"{gs:+.2f} dB, last-30 {fmt(gl)} dB. #G {row['init_n']:,} vs {base['init_n']:,}; "
                     f"train {fmt(row['stream_train_s_per_frame'])} vs {fmt(base['stream_train_s_per_frame'])} s/frame.")
    open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))
    print("saved", args.out)


if __name__ == "__main__":
    main()
