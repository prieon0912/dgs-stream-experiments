"""Evaluate finished streams; audit missing frames first, keep metric sources explicit."""
import argparse
import os
from pathlib import Path

from common.result_io import (read_json, write_json, write_csv, finite_number, strict_mean,
                              strict_sum, tri_all)
from common.run_audit import audit_run, parse_log, ply_count, frame_paths
from common.aabb import file_sha256


def megabytes(path):
    path = Path(path)
    return path.stat().st_size / 1e6 if path.is_file() else None


def segment_exclusivity(path):
    path = Path(path)
    if not path.is_file():
        return None
    import json
    flags = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line).get("gpu_exclusive_at_start")
        except (ValueError, TypeError):
            flags.append(None)
            continue
        flags.append(True if value is True or value == "True" else False if value is False or value == "False" else None)
    return tri_all(flags)


def peak_gpu_mb(path):
    path = Path(path)
    if not path.is_file():
        return None
    values = [line.split(",")[0].strip() for line in path.read_text(encoding="utf-8").splitlines()]
    values = [float(value) for value in values if finite_number(value)]
    return max(values) if values else None


def summarize(rows, run, audit, lpips_every=1):
    rows = sorted(rows, key=lambda row: row["frame"])
    stream = [row for row in rows if row["frame"] >= 1]
    initial = next((row for row in rows if row["frame"] == 0), {})
    summary = dict(completion=audit, lpips_every=lpips_every)
    initial_path = Path(run) / "init/point_cloud/iteration_15000/point_cloud.ply"
    summary["source_init_sha256"] = file_sha256(initial_path) if initial_path.is_file() else None
    for tag, subset in (("stream", stream), ("all", rows)):
        sampled = [row.get("lpips") for row in subset if finite_number(row.get("lpips"))]
        summary[tag] = dict(frames=len(subset), psnr=strict_mean(row.get("psnr_main") for row in subset),
                            psnr_png=strict_mean(row.get("psnr_png") for row in subset),
                            ssim=strict_mean(row.get("ssim") for row in subset),
                            lpips=strict_mean(sampled), lpips_sampled_frames=len(sampled))
    windows_valid = len(stream) >= 60
    summary["first30_psnr"] = strict_mean(row.get("psnr_main") for row in stream[:30]) if windows_valid else None
    summary["last30_psnr"] = strict_mean(row.get("psnr_main") for row in stream[-30:]) if windows_valid else None
    summary["drift_psnr"] = (summary["last30_psnr"] - summary["first30_psnr"]
                             if finite_number(summary["last30_psnr"]) and finite_number(summary["first30_psnr"]) else None)
    summary["drift_windows_nonoverlapping"] = windows_valid
    run = Path(run)
    init = read_json(run / "init_timing.json")
    warm = read_json(run / "ntc_warmup.json")
    train_times = [strict_sum([row.get("t_s1"), row.get("t_s2")]) for row in stream]
    summary["time"] = dict(init_sec=init.get("init_time_sec"), init_steps=init.get("steps"),
                            init_gpu_exclusive=init.get("gpu_exclusive"),
                            ntc_warmup_sec=warm.get("warmup_sec"), ntc_gpu_exclusive=warm.get("gpu_exclusive"),
                            stream_train_sec_total=strict_sum(train_times),
                            stream_train_sec_per_frame=strict_mean(train_times),
                            stream_prep_sec_per_frame=strict_mean(row.get("t_prep") for row in stream),
                            stage1_sec_per_frame=strict_mean(row.get("t_s1") for row in stream),
                            stage2_sec_per_frame=strict_mean(row.get("t_s2") for row in stream),
                            stream_wall_sec_total=strict_sum(row.get("t_frame") for row in stream),
                            stream_wall_sec_per_frame=strict_mean(row.get("t_frame") for row in stream))
    measured = strict_sum([summary["time"]["init_sec"], summary["time"]["ntc_warmup_sec"],
                           summary["time"]["stream_wall_sec_total"]])
    summary["time"].update(measured_pipeline_sec=measured, end_to_end_sec=measured,
                            scope="Recorded init + NTC warm-up + streaming wall only; excludes calibration, weight loading, evaluation and diagnostics")
    payloads = [row.get("storage_mb") for row in stream]
    storage_sum = strict_sum([initial.get("storage_mb"), *payloads])
    summary["storage"] = dict(init_mb=initial.get("storage_mb"), per_frame_payload_mb=strict_mean(payloads),
                               paper_avg_mb=storage_sum / len(rows) if storage_sum is not None and rows else None)
    summary["n_init"] = initial.get("n_points")
    summary["peak_gpu_mem_mb"] = peak_gpu_mb(run / "gpu.csv")
    summary["peak_gpu_scope"] = "Device memory sampled every 2 s during streaming; excludes NTC warm-up and can miss transient peaks"
    summary["gpu_exclusive_all_segments"] = segment_exclusivity(run / "segments.jsonl")
    summary["time_reportable"] = (audit["complete"] and
                                   tri_all([init.get("gpu_exclusive"), warm.get("gpu_exclusive"),
                                            summary["gpu_exclusive_all_segments"]]) is True and
                                   all(finite_number(summary["time"].get(key)) for key in
                                       ("init_sec", "ntc_warmup_sec", "stream_train_sec_total", "stream_wall_sec_total")))
    quality_ok = (len(rows) == audit["expected_frames"] and
                  all(all(finite_number(row.get(key)) for key in ("psnr_main", "psnr_png", "ssim")) for row in rows) and
                  all(finite_number(row.get("lpips")) for row in rows if row["frame"] == 0 or row["frame"] % lpips_every == 0))
    summary["quality_data_complete"] = quality_ok
    return summary


def frame0_row(run, scene, test_cam):
    import torch
    from common.colmap_io import load_cameras, load_image
    from common.gs_render import GS, RCam, render
    from common.metrics import all_metrics, psnr, to_t
    ply = Path(run) / "init/point_cloud/iteration_15000/point_cloud.ply"
    cam = load_cameras(str(Path(scene) / "frame000000"), "images_2", only=(test_cam,))[0]
    gt = to_t(load_image(cam.image_path))
    gs = GS.from_ply(ply)
    with torch.no_grad():
        image = render(gs, RCam(cam)).clamp(0, 1)
        float_psnr = psnr(image, gt)
        metrics = all_metrics(torch.round(image * 255) / 255, gt)
    count = int(gs.xyz.shape[0])
    del gs
    torch.cuda.empty_cache()
    return dict(frame=0, psnr_main=float_psnr, psnr_source="float_render", psnr_png=metrics["psnr"],
                ssim=metrics["ssim"], lpips=metrics["lpips"], n_points=count, storage_mb=megabytes(ply))


def evaluate_run(run, scene, test_cam, expected_frames, lpips_every, allow_incomplete=False):
    audit = audit_run(run, expected_frames, verify_images=True)
    write_json(Path(run) / "completion.json", audit)
    if not audit["complete"] and not allow_incomplete:
        return None
    import torch
    from common.colmap_io import load_image
    from common.metrics import all_metrics, to_t
    rows = [frame0_row(run, scene, test_cam)]
    config = read_json(Path(run) / "cfg_run.json")
    logs = parse_log(Path(run) / "train.log")
    for frame in range(1, expected_frames):
        paths = frame_paths(run, frame, config)
        if not paths["png"].is_file():
            continue
        gt = to_t(load_image(str(Path(scene) / f"frame{frame:06d}" / "images_2" / (test_cam + ".png"))))
        metrics = all_metrics(to_t(load_image(str(paths["png"]))), gt, do_lpips=frame % lpips_every == 0)
        log = logs.get(f"frame{frame:06d}", {})
        # Never fall back to a PNG PSNR under the float-PSNR column.
        final_psnr = log.get(f"psnr_it{paths['final_iteration']}")
        count = ply_count(paths["full"])
        if count is None:
            text = paths["folder"] / "n_points.txt"
            value = text.read_text().strip() if text.is_file() else None
            count = int(float(value)) if finite_number(value) else None
        rows.append({**log, "frame": frame, "psnr_main": final_psnr,
                     "psnr_source": "upstream_float_log" if final_psnr is not None else "missing",
                     "psnr_png": metrics["psnr"], "ssim": metrics["ssim"], "lpips": metrics.get("lpips"),
                     "n_points": count, "n_added": ply_count(paths["added"]),
                     "storage_mb": strict_sum([megabytes(paths["ntc"]), megabytes(paths["added"])])})
    write_csv(Path(run) / "per_frame.csv", rows)
    summary = summarize(rows, run, audit, lpips_every)
    write_json(Path(run) / "run_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--runs_dir", default=os.path.join(os.environ.get("RUNS", "/workspace/runs"), "stream"))
    parser.add_argument("--scene", default=os.environ.get("SCENE", "/workspace/data/flame_steak"))
    parser.add_argument("--test_cam", default="cam00")
    parser.add_argument("--expected-frames", type=int, default=int(os.environ.get("N_FRAMES", "300")))
    parser.add_argument("--lpips_every", type=int, default=1)
    parser.add_argument("--allow-incomplete", action="store_true", help="Diagnostic summaries only; not eligible for formal comparison")
    args = parser.parse_args()
    if args.lpips_every < 1 or args.expected_frames < 2:
        parser.error("Invalid LPIPS interval or frame count")
    failed = []
    for name in args.runs:
        run = Path(args.runs_dir) / name
        try:
            summary = evaluate_run(run, args.scene, args.test_cam, args.expected_frames, args.lpips_every,
                                   args.allow_incomplete)
            if summary is None:
                failed.append(name)
                print(f"INCOMPLETE: {name}; see completion.json")
            else:
                print(f"{name}: complete={summary['completion']['complete']} "
                      f"quality_complete={summary['quality_data_complete']} time_reportable={summary['time_reportable']}")
                if not summary["completion"]["complete"] or not summary["quality_data_complete"]:
                    failed.append(name)
        except Exception as error:
            failed.append(name)
            write_json(run / "evaluation_error.json", dict(error=type(error).__name__, message=str(error)))
            print(f"FAILED: {name}: {error}")
    if failed:
        raise SystemExit(f"Evaluation incomplete/failed: {', '.join(failed)}")
