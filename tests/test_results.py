"""Synthetic CPU tests of complete, incomplete, missing-value and mixed-condition results."""
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.aabb import load_aabb, aabb_id, file_sha256
from common.collect_results import collect
from common.diagnostics import occupancy, sources_path
from common.evaluate_stream import summarize, segment_exclusivity, megabytes
from common.ply_io import save_ply
from common.result_io import read_json, write_json, write_csv, strict_sum, tri_all
from common.run_audit import audit_run, parse_log, frame_paths, valid_ply


def ply(path, count=2):
    save_ply(dict(xyz=np.zeros((count, 3), dtype=np.float32), f_dc=np.zeros((count, 3)),
                  f_rest=np.zeros((count, 9)), opacity=np.zeros(count), scale=np.zeros((count, 3)),
                  rot=np.tile([1., 0., 0., 0.], (count, 1))), path)


def fixture(root, name="synthetic_a", gpu="SYNTHETIC_GPU", complete=True):
    run = root / "stream" / name
    folder = root / "initializations" / name
    folder.mkdir(parents=True)
    final = folder / "final.ply"
    ply(final)
    initial = run / "init/point_cloud/iteration_15000/point_cloud.ply"
    ply(initial)
    box = load_aabb()
    filters = dict(border=.03125, depth_prune=1., min_opacity=.005, sh1="keep")
    names = ["cam01", "cam04", "cam05", "cam12"]
    metadata = dict(n_raw_gaussians=10, n_converted_gaussians=3, n_final_gaussians=2,
                    camera_names=names, n_views=4, process_res=504, alignment_pass=True,
                    alignment_residual_rel=.001, max_gaussians=0, refine_iters=1000, filter_policy=filters)
    timing = dict(init_time_sec=.6, gpu=gpu, gpu_exclusive=True,
                  steps=dict(da3_infer=.2, convert=.1, refine_1000=.3), init_metadata=metadata)
    quality = dict(psnr=32., psnr_png=31.9, ssim=.95, lpips=.1)
    record = dict(run=name, counts=dict(n_total=2, n_inside=1, inside_fraction=.5),
                  timing=timing, init_metadata=metadata, quality_final=quality,
                  quality_pre_refine=dict(psnr=25., ssim=.8, lpips=.2), occupancy=dict(voxel_size=.2))
    write_json(folder / "initialization.json", record)
    write_json(folder / "final_timing.json", timing)
    write_json(run / "init_timing.json", timing)
    write_json(run / "ntc_warmup.json", dict(warmup_sec=1., gpu_exclusive=True))
    write_json(run / "run_contract.json", dict(aabb=box, aabb_id=aabb_id(box), init_sha256=file_sha256(final)))
    write_json(run / "cfg_run.json", dict(frame_end=4, iterations=150, iterations_s2=100))
    environment = dict(experiment_code_sha256="synthetic-code", packages={"torch": "SYNTHETIC"}, gpu=gpu)
    write_json(run / "environment_gs.json", environment)
    (run / "segments.jsonl").write_text(json.dumps(dict(gpu_exclusive_at_start=True)) + "\n")
    logs, rows = [], [dict(frame=0, psnr_main=32., psnr_png=31.9, ssim=.95, lpips=.1,
                           psnr_source="float_render", n_points=2, storage_mb=megabytes(initial))]
    for frame in range(1, 4):
        paths = frame_paths(run, frame)
        paths["folder"].mkdir(parents=True)
        if complete or frame != 2:
            Image.new("RGB", (8, 8)).save(paths["png"])
        paths["ntc"].write_bytes(b"synthetic checkpoint placeholder; not real weights")
        ply(paths["added"], 0)
        logs += [f"Optimizing /output/frame{frame:06d}",
                 "[ITER 150] Evaluating test: L1 0.1 PSNR 30.0",
                 "[ITER 250] Evaluating test: L1 0.1 PSNR 32.0",
                 "Preparation: 1.0", "Stage 1: 2.0", "Stage 2: 3.0",
                 f"Frame frame{frame:06d} finished in 6.0 seconds."]
        rows.append(dict(frame=frame, psnr_main=32., psnr_png=31.9, ssim=.95, lpips=.1,
                         psnr_source="upstream_float_log", n_points=2, n_added=0,
                         t_prep=1., t_s1=2., t_s2=3., t_frame=6.,
                         storage_mb=strict_sum([megabytes(paths["ntc"]), megabytes(paths["added"])])))
    (run / "train.log").write_text("\n".join(logs) + "\n")
    write_csv(run / "per_frame.csv", rows)
    write_json(run / "run_summary.json", summarize(rows, run, audit_run(run, 4, True)))
    condition = dict(run=name, label=name, kind="da3", group="camera_selection", method="random", n_views=4,
                     camera_names=names, seed=0, refine_iters=1000, max_gaussians=0, final_ply=str(final),
                     initialization_record=str(folder / "initialization.json"),
                     environment_gs=str(folder / "environment_gs.json"))
    return condition, run, rows


def manifest(root, conditions):
    path = root / "manifest.json"
    write_json(path, dict(expected_frames=4, stream_root=str(root / "stream"), aabb=load_aabb(),
                         max_gaussians=0, process_res=504,
                         filter_policy=dict(border=.03125, depth_prune=1., min_opacity=.005, sh1="keep"),
                         conditions=conditions))
    return path


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class ResultTests(unittest.TestCase):
    def test_complete_export_and_before_after_quality(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            condition, run, rows = fixture(root)
            output = root / "results"
            exported = collect(manifest(root, [condition]), output)
            self.assertTrue(exported[0]["comparison_eligible"])
            self.assertTrue(exported[0]["time_reportable"])
            self.assertEqual(exported[0]["refine_psnr_gain"], 7.)
            self.assertEqual(exported[0]["raw_gaussians"], 10)
            self.assertEqual(len(read_csv(output / "per_frame.csv")), 4)
            self.assertEqual(len(read_csv(output / "summary_valid.csv")), 1)
            self.assertTrue((output / "summary.csv").read_bytes().startswith(b"\xef\xbb\xbf"))
            self.assertEqual(exported[0]["last30_minus_first30_psnr"], None)
            self.assertNotIn("NaN", (output / "collection.json").read_text())

    def test_missing_frame_is_not_a_valid_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            condition, run, _ = fixture(root, complete=False)
            result = collect(manifest(root, [condition]), root / "results")[0]
            self.assertFalse(result["complete"])
            self.assertFalse(result["comparison_eligible"])
            self.assertEqual(result["completed_stream_frames"], 2)
            self.assertEqual(len(read_csv(root / "results/summary_valid.csv")), 0)

    def test_unmeasured_exclusivity_cannot_be_time_ok(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            condition, run, rows = fixture(root)
            warm = read_json(run / "ntc_warmup.json")
            warm["gpu_exclusive"] = None
            write_json(run / "ntc_warmup.json", warm)
            write_json(run / "run_summary.json", summarize(rows, run, audit_run(run, 4, True)))
            result = collect(manifest(root, [condition]), root / "results")[0]
            self.assertTrue(result["comparison_eligible"])
            self.assertFalse(result["time_reportable"])
            row = read_csv(root / "results/summary_valid.csv")[0]
            self.assertEqual(row["init_time_s"], "")
            self.assertEqual(row["stream_wall_total_s"], "")

    def test_missing_payload_and_truncated_ply_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            condition, run, _ = fixture(root)
            paths = frame_paths(run, 2)
            paths["ntc"].unlink()
            self.assertFalse(audit_run(run, 4)["complete"])
            self.assertIsNone(strict_sum([megabytes(paths["ntc"]), megabytes(paths["added"])]))
            ply(paths["added"], 2)
            data = paths["added"].read_bytes()
            paths["added"].write_bytes(data[:-1])
            self.assertFalse(valid_ply(paths["added"]))

    def test_psnr_log_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            condition, run, rows = fixture(root)
            text = (run / "train.log").read_text().replace("[ITER 250] Evaluating test: L1 0.1 PSNR 32.0", "missing")
            (run / "train.log").write_text(text)
            audit = audit_run(run, 4)
            self.assertFalse(audit["complete"])
            rows[1]["psnr_main"] = None
            result = summarize(rows, run, audit)
            self.assertIsNone(result["stream"]["psnr"])
            self.assertFalse(result["quality_data_complete"])

    def test_mixed_gpu_and_changed_policy_are_flagged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, _, _ = fixture(root, "first", "SYNTHETIC_GPU_A")
            second, _, _ = fixture(root, "second", "SYNTHETIC_GPU_B")
            result = collect(manifest(root, [first, second]), root / "results")
            self.assertTrue(all(not row["comparison_eligible"] for row in result))
            self.assertIn("mixed_gpu_models", result[0]["issues"])
            data = read_json(first["initialization_record"])
            data["init_metadata"]["max_gaussians"] = 1000000
            write_json(first["initialization_record"], data)
            result = collect(manifest(root, [first]), root / "results")
            self.assertIn("gaussian_limit_policy_unverified_or_mismatch", result[0]["issues"])

    def test_not_started_condition_has_blank_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            condition = dict(run="not_started", label="미실행", kind="da3", group="camera_selection", method="random",
                             n_views=4, camera_names=["cam01", "cam02", "cam03", "cam04"], seed=0, refine_iters=1000,
                             max_gaussians=0, final_ply=str(root / "missing.ply"), initialization_record=str(root / "missing.json"),
                             environment_gs=str(root / "environment.json"))
            result = collect(manifest(root, [condition]), root / "results")[0]
            self.assertEqual(result["status"], "not_started")
            row = read_csv(root / "results/summary.csv")[0]
            self.assertEqual(row["stream_psnr"], "")
            self.assertEqual(row["raw_gaussians"], "")

    def test_alignment_failed_condition_is_reported_with_residual(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            init_dir = root / "initializations" / "align_fail"
            init_dir.mkdir(parents=True)
            write_json(init_dir / "converted_alignment.json",
                       dict(alignment_pass=False, residual_median_rel=0.1183, alignment_threshold=0.02))
            write_json(root / "stream" / "align_fail" / "execution_status.json",
                       dict(stage="initialization", state="alignment_failed", exit_code=1))
            condition = dict(run="align_fail", label="random_04+1k", kind="da3", group="camera_selection", method="random",
                             n_views=4, camera_names=["cam01", "cam04", "cam05", "cam12"], seed=0, refine_iters=1000,
                             max_gaussians=1000000, final_ply=str(init_dir / "refined.ply"),
                             pre_ply=str(init_dir / "limited.ply"),
                             initialization_record=str(init_dir / "initialization.json"),
                             environment_gs=str(root / "environment.json"))
            result = collect(manifest(root, [condition]), root / "results")[0]
            self.assertEqual(result["status"], "alignment_failed")
            self.assertFalse(result["comparison_eligible"])
            row = read_csv(root / "results/summary.csv")[0]
            self.assertEqual(row["status"], "alignment_failed")
            self.assertAlmostEqual(float(row["alignment_residual_rel"]), 0.1183)
            self.assertEqual(row["stream_psnr"], "")

    def test_retried_partial_frame_clears_old_final_psnr(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.log"
            path.write_text("Optimizing /out/frame000001\n[ITER 250] Evaluating test: L1 0.1 PSNR 32\n"
                            "Optimizing /out/frame000001\n[ITER 150] Evaluating test: L1 0.1 PSNR 20\n")
            row = parse_log(path)["frame000001"]
            self.assertNotIn("psnr_it250", row)

    def test_stale_summary_cannot_hide_truncated_frame_table(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            condition, run, rows = fixture(root)
            write_csv(run / "per_frame.csv", rows[:-1])
            result = collect(manifest(root, [condition]), root / "results")[0]
            self.assertTrue(result["complete"])
            self.assertFalse(result["comparison_eligible"])
            self.assertIn("metrics_incomplete", result["issues"])

    def test_invalid_result_json_is_reported_in_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            condition, run, _ = fixture(root)
            (run / "run_summary.json").write_text('{"interrupted":')
            result = collect(manifest(root, [condition]), root / "results")[0]
            self.assertFalse(result["comparison_eligible"])
            self.assertIn("summary_json_invalid", result["issues"])


class DiagnosticTests(unittest.TestCase):
    def test_overlap_counts_view_origins_without_calling_them_redundant(self):
        box = dict(min=[0, 0, 0], max=[3, 3, 3])
        points = np.array([[.1, .1, .1], [.2, .1, .1], [2.1, .1, .1]])
        result = occupancy(points, box, voxel_size=1, source_views=np.array([0, 1, 1]))
        self.assertEqual(result["occupied_voxels"], 2)
        self.assertEqual(result["multi_view_voxels"], 1)
        self.assertAlmostEqual(result["fraction_points_in_multi_view_voxels"], 2 / 3)
        self.assertIn("not", result["diagnostic_scope"])

    def test_gaussian_subsample_keeps_matching_view_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, output = root / "source.ply", root / "selected.ply"
            from common.ply_io import load_ply
            ply(path, 3)
            data = load_ply(path)
            data["opacity"] = np.array([-2., 4., 1.])
            save_ply(data, path)
            np.savez_compressed(sources_path(path), view_ids=np.array([0, 1, 2], dtype=np.int16),
                                camera_names=np.array(["cam01", "cam02", "cam03"]))
            completed = subprocess.run([sys.executable, str(ROOT / "scripts/06_subsample.py"), "--ply", str(path),
                                        "--out", str(output), "--topk", "1"], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            with np.load(sources_path(output), allow_pickle=False) as selected:
                self.assertEqual(selected["view_ids"].tolist(), [1])

    def test_stage_resume_verifies_outputs_and_preserves_partial(self):
        spec = importlib.util.spec_from_file_location("stage", ROOT / "scripts/18_run_stage.py")
        stage = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stage)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, marker = root / "output.txt", root / "stage.json"
            output.write_text("partial")
            command = [sys.executable, "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('complete')", str(output)]
            stage.run_stage(marker, "signature", [], [str(output)], command)
            self.assertEqual(output.read_text(), "complete")
            self.assertEqual(len(list(root.glob("output.txt.incomplete_*"))), 1)
            stage.run_stage(marker, "signature", [], [str(output)], command)
            output.write_text("changed")
            with self.assertRaises(ValueError):
                stage.run_stage(marker, "signature", [], [str(output)], command)


if __name__ == "__main__":
    unittest.main(verbosity=2)
