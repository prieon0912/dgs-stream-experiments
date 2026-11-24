"""CPU regression tests for selection geometry, held-out views, AABB and job generation."""
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.camera_selection import make_plan, selection_order, load_geometry, read_views_file
from common.aabb import (load_aabb, save_aabb, aabb_id, inside_mask, point_stats,
                         assert_same_bounds, bind_run, load_xyz)

spec = importlib.util.spec_from_file_location("make_jobs", ROOT / "scripts/14_make_camera_jobs.py")
jobs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jobs)


def cameras():
    result = []
    for index in range(21):
        theta = 2 * math.pi * index / 21
        center = [4 * math.cos(theta), 3 * math.sin(theta), 1 + .2 * math.cos(2 * theta)]
        result.append(dict(name=f"cam{index:02d}", center=center, forward=(-np.asarray(center)).tolist()))
    return result


def write_colmap_fixture(root):
    sparse = root / "sparse/0"
    sparse.mkdir(parents=True)
    images = root / "images_2"
    images.mkdir()
    with (sparse / "cameras.bin").open("wb") as handle:
        handle.write(struct.pack("<QiiQQdddd", 1, 1, 1, 8, 8, 6., 6., 4., 4.))
    with (sparse / "images.bin").open("wb") as handle:
        handle.write(struct.pack("<Q", 21))
        for index, camera in enumerate(cameras()):
            # Identity camera rotation: +Z direction; center = -translation.
            handle.write(struct.pack("<idddddddi", index + 1, 1., 0., 0., 0.,
                                     *(-np.asarray(camera["center"])), 1))
            handle.write((camera["name"] + ".png").encode() + b"\0")
            handle.write(struct.pack("<Q", 0))
            Image.new("RGB", (8, 8)).save(images / (camera["name"] + ".png"))


class SelectionTests(unittest.TestCase):
    def test_exact_counts_exclusion_and_nested_sets(self):
        plan = make_plan(cameras())
        self.assertEqual(len(plan["conditions"]), 12)
        self.assertEqual(plan, make_plan(list(reversed(cameras()))))
        for method in ("random", "position_fps", "direction_fps"):
            previous = set()
            for row in [row for row in plan["conditions"] if row["method"] == method]:
                names = row["camera_names"]
                self.assertEqual(len(names), row["n_views"])
                self.assertEqual(len(names), len(set(names)))
                self.assertNotIn("cam00", names)
                self.assertEqual(names[0], "cam01")
                self.assertTrue(previous.issubset(names))
                previous = set(names)

    def test_seed_and_invalid_requests(self):
        self.assertNotEqual(selection_order(cameras(), "random", 0), selection_order(cameras(), "random", 1))
        self.assertEqual(selection_order(cameras(), "position_fps", 0), selection_order(cameras(), "position_fps", 1))
        with self.assertRaises(ValueError):
            make_plan(cameras(), counts=[21])
        with self.assertRaises(ValueError):
            make_plan(cameras(), anchor="cam00")
        with self.assertRaises(ValueError):
            make_plan(cameras(), counts=[4, 4])

    def test_fps_extreme_separation(self):
        geometry = [dict(name="a", center=[0, 0, 0], forward=[0, 0, 1]),
                    dict(name="b", center=[1, 0, 0], forward=[1, 0, 0]),
                    dict(name="c", center=[10, 0, 0], forward=[0, 0, -1])]
        self.assertEqual(selection_order(geometry, "position_fps")[:2], ["a", "c"])
        self.assertEqual(selection_order(geometry, "direction_fps")[:2], ["a", "c"])

    def test_colmap_geometry_and_cpu_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "frame000000"
            write_colmap_fixture(frame)
            actual = load_geometry(frame_dir=frame)
            np.testing.assert_allclose(actual[3]["center"], cameras()[3]["center"])
            np.testing.assert_allclose(actual[3]["forward"], [0, 0, 1])
            views = Path(directory) / "views.json"
            views.write_text(json.dumps(dict(camera_names=["cam01", "cam05", "cam09", "cam13"],
                                            n_views=4, anchor="cam01", excluded_cameras=["cam00"])))
            command = [sys.executable, str(ROOT / "scripts/02_run_da3.py"), "--frame_dir", str(frame),
                       "--views-file", str(views), "--out", str(Path(directory) / "unused.pt"), "--dry-run"]
            good = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(good.returncode, 0, good.stderr)
            self.assertFalse((Path(directory) / "unused.pt").exists())
            views.write_text(json.dumps(["cam00", "cam01"]))
            bad = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("Held-out", bad.stderr)
            views.write_text(json.dumps(["cam01", "cam99"]))
            missing = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("not found", missing.stderr)


class AABBTests(unittest.TestCase):
    def test_center_inclusion_and_nonfinite_diagnostics(self):
        box = dict(min=[0, 0, 0], max=[1, 1, 1])
        points = np.array([[0, 0, 0], [1, 1, 1], [.5, .5, .5], [1.01, 0, 0], [np.nan, 0, 0]])
        self.assertEqual(inside_mask(points, box).tolist(), [True, True, True, False, False])
        stats = point_stats(points, box)
        self.assertEqual(stats["n_inside"], 3)
        self.assertEqual(stats["n_nonfinite"], 1)
        self.assertEqual(stats["inside_fraction"], .6)

    def test_manual_box_cli_and_count_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            box, points, report = root / "manual.json", root / "points.json", root / "counts.json"
            points.write_text(json.dumps([[0, 0, 0], [1, 2, 3], [2, 2, 3]]))
            script = [sys.executable, str(ROOT / "scripts/13_aabb.py")]
            manual = subprocess.run(script + ["manual", "--min", "0", "0", "0", "--max", "1", "2", "3",
                                              "--out", str(box)], capture_output=True, text=True)
            self.assertEqual(manual.returncode, 0, manual.stderr)
            count = subprocess.run(script + ["stats", "--aabb", str(box), "--points", str(points),
                                             "--out", str(report)], capture_output=True, text=True)
            self.assertEqual(count.returncode, 0, count.stderr)
            result = json.loads(report.read_text())
            self.assertEqual(result["point_sets"][0]["n_inside"], 2)
            self.assertEqual(result["point_sets"][0]["n_total"], 3)

    def test_invalid_boxes_and_float_bounds_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aabb.json"
            first = save_aabb(path, dict(min=[-.123456789, 0, 0], max=[1, 2, 3]))
            self.assertEqual(aabb_id(first), aabb_id(load_aabb(path)))
            assert_same_bounds(first, dict(min=np.asarray(first["min"]).tolist(), max=first["max"]))
            with self.assertRaises(ValueError):
                assert_same_bounds(first, dict(min=[0, 0, 0], max=[1, 2, 3]))
            with self.assertRaises(ValueError):
                save_aabb(Path(directory) / "bad.json", dict(min=[0, 0, 0], max=[0, 1, 1]))

    def test_run_resume_rejects_changed_box_and_init(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ply, conf, run = root / "init.ply", root / "conf.json", root / "run"
            ply.write_bytes(b"original init")
            conf.write_text("{}")
            box = load_aabb()
            bind_run(run, box, ply, conf, "warmup")
            bind_run(run, box, ply, conf, "warmup")
            with self.assertRaises(ValueError):
                bind_run(run, dict(min=[-1, -1, -1], max=[1, 1, 1]), ply, conf, "warmup")
            ply.write_bytes(b"changed init")
            with self.assertRaises(ValueError):
                bind_run(run, box, ply, conf, "warmup")
            legacy = root / "legacy"
            legacy.mkdir()
            (legacy / "ntc.pth").write_bytes(b"old")
            with self.assertRaises(ValueError):
                bind_run(legacy, box, ply, conf, "warmup")

    def test_ply_and_json_point_loading(self):
        from common.ply_io import save_ply
        with tempfile.TemporaryDirectory() as directory:
            points = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
            path = Path(directory) / "points.ply"
            save_ply(dict(xyz=points, f_dc=np.zeros((2, 3)), f_rest=np.zeros((2, 9)),
                          opacity=np.zeros(2), scale=np.zeros((2, 3)), rot=np.ones((2, 4))), path)
            np.testing.assert_array_equal(load_xyz(path), points)
            path = Path(directory) / "points.json"
            path.write_text(json.dumps(dict(xyz=points.tolist())))
            np.testing.assert_array_equal(load_xyz(path), points)


class JobTests(unittest.TestCase):
    def test_generated_jobs_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            plan, box = make_plan(cameras()), load_aabb()
            jobs.generate_jobs(plan, box, out)
            self.assertEqual(len((out / "queue.txt").read_text().splitlines()), 15)
            initialize = (out / "run_initializations.sh").read_text()
            self.assertEqual(initialize.count("--views-file"), 13)
            self.assertEqual(initialize.count("--iters 1000"), 13)
            self.assertNotIn("06_subsample.py", initialize)
            self.assertIn('export AABB_CONFIG="$JOB_DIR/aabb.json"', initialize)
            self.assertEqual(len(list((out / "views").glob("*.json"))), 13)
            self.assertNotIn("\\workspace", initialize)
            self.assertIn("15_collect_results.py", (out / "evaluate.sh").read_text())
            jobs.generate_jobs(plan, box, out)
            with self.assertRaises(FileExistsError):
                jobs.generate_jobs(plan, box, out, tag="different")
            for views in (out / "views").glob("*.json"):
                names, _ = read_views_file(views)
                self.assertNotIn("cam00", names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
