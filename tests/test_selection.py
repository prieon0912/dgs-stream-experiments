import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import selection as S  # noqa: E402
from common.ply_io import load_ply, save_ply  # noqa: E402


def cloud(n, seed=0):
    rng = np.random.default_rng(seed)
    return dict(xyz=rng.normal(size=(n, 3)).astype(np.float32) * 3,
                f_dc=rng.normal(size=(n, 3)).astype(np.float32),
                f_rest=rng.normal(size=(n, 9)).astype(np.float32),
                opacity=rng.normal(size=n).astype(np.float32),
                scale=rng.normal(size=(n, 3)).astype(np.float32) - 3,
                rot=rng.normal(size=(n, 4)).astype(np.float32))


class SelectionTests(unittest.TestCase):
    def test_every_method_returns_exactly_n_unique_sorted(self):
        d = cloud(5000)
        score = np.random.default_rng(1).random(5000)
        for m in S.NUMPY_METHODS:
            for n in (1, 777, 4999, 5000, 9000):
                idx, _ = S.select(m, n, xyz=d["xyz"], opacity_logit=d["opacity"], score=score)
                self.assertEqual(len(idx), min(n, 5000), (m, n))
                self.assertEqual(len(np.unique(idx)), len(idx))
                self.assertTrue(np.all(np.diff(idx) > 0))

    def test_opacity_topk_matches_argsort(self):
        d = cloud(1000)
        idx = S.opacity_topk(d["opacity"], 100)
        np.testing.assert_array_equal(idx, np.sort(np.argsort(-d["opacity"])[:100]))

    def test_voxel_keeps_max_opacity_per_voxel_and_spreads_points(self):
        # dense cluster of 900 high-opacity points + 100 sparse low-opacity points
        rng = np.random.default_rng(0)
        dense = rng.normal(scale=0.01, size=(900, 3))
        sparse = rng.uniform(-10, 10, size=(100, 3))
        xyz = np.concatenate([dense, sparse])
        op = np.concatenate([np.full(900, 5.0), np.full(100, -5.0)])
        op_idx = S.opacity_topk(op, 100)
        vox_idx, info = S.voxel_select(xyz, op, 100, return_info=True)
        self.assertTrue(np.all(op_idx < 900))                    # opacity keeps only the cluster
        self.assertGreater((vox_idx >= 900).sum(), 50)           # voxel keeps the sparse coverage
        self.assertGreaterEqual(info["occupied_voxels"], 100)
        # the representative of each voxel is its max-opacity member
        rep = S.voxel_representatives(xyz, op, info["voxel_size"])
        self.assertTrue(set(vox_idx.tolist()) <= set(rep.tolist()))

    def test_contrib_rank_and_sample(self):
        score = np.zeros(1000)
        score[:50] = 10.0
        score[50:100] = 1e-6
        self.assertEqual(set(S.contrib_rank(score, 50).tolist()), set(range(50)))
        s = S.contrib_sample(score, 50, seed=0)
        self.assertGreater((s < 50).sum(), 45)                     # sampling favours high score
        np.testing.assert_array_equal(s, S.contrib_sample(score, 50, seed=0))   # deterministic
        self.assertFalse(np.array_equal(s, S.contrib_sample(np.ones(1000), 50, seed=0)))
        # more than the positive-score count: zeros fill the rest
        self.assertEqual(len(S.contrib_sample(score, 300, seed=0)), 300)

    def test_contrib_sample_is_proportional(self):
        score = np.array([1.0] * 500 + [4.0] * 500)
        hits = np.zeros(1000)
        for seed in range(200):
            hits[S.contrib_sample(score, 10, seed=seed)] += 1
        ratio = hits[500:].sum() / hits[:500].sum()
        self.assertGreater(ratio, 3.0)
        self.assertLess(ratio, 5.3)

    def test_volume_weight(self):
        w = S.volume_weight(np.log(np.array([[1, 1, 1], [2, 2, 2], [0.5, 0.5, 0.5]] * 10)), 0.1)
        self.assertTrue(np.all((w > 0) & (w <= 1)))
        np.testing.assert_array_equal(S.volume_weight(np.zeros((3, 3)), 0), np.ones(3))

    def test_cli_opacity_and_voxel_write_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src.ply")
            save_ply(cloud(3000), src)
            np.savez_compressed(os.path.join(tmp, "src_sources.npz"), view_ids=np.arange(3000) % 20,
                                camera_names=np.array([f"cam{i:02d}" for i in range(1, 21)]))
            env = dict(os.environ, PYTHONPATH=str(ROOT))
            for m in ("opacity", "voxel"):
                out = os.path.join(tmp, f"{m}.ply")
                r = subprocess.run([sys.executable, str(ROOT / "scripts/23_select_gaussians.py"), "--ply", src,
                                    "--out", out, "--method", m, "--n", "1000"], env=env, capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertEqual(len(load_ply(out)["xyz"]), 1000)
                rec = json.load(open(os.path.join(tmp, f"{m}_select.json")))
                self.assertEqual(rec["n_out"], 1000)
                idx = np.load(os.path.join(tmp, f"{m}_indices.npy"))
                with np.load(os.path.join(tmp, f"{m}_sources.npz")) as z:
                    np.testing.assert_array_equal(z["view_ids"], idx % 20)
                np.testing.assert_allclose(load_ply(out)["xyz"], load_ply(src)["xyz"][idx])
                self.assertTrue(os.path.isfile(os.path.join(tmp, f"{m}_timing.json")))
            # contrib without frame_dir is refused
            r = subprocess.run([sys.executable, str(ROOT / "scripts/23_select_gaussians.py"), "--ply", src,
                                "--out", os.path.join(tmp, "c.ply"), "--method", "contrib_rank", "--n", "10"],
                               env=env, capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)


class Phase1SummaryTests(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import importlib
        self.mod = importlib.import_module("24_summarize_phase1")

    @staticmethod
    def row(method, n, psnr, train):
        return dict(method=method, n_target=n, psnr=psnr, train_psnr=train, complete=True)

    def test_both_views_must_pass_and_smallest_tested_n_wins(self):
        r = self.row
        rows = [r("opacity", 1_000_000, 30.0, 32.0), r("opacity", 600_000, 29.5, 31.9),
                r("voxel", 1_000_000, 30.1, 32.1), r("voxel", 600_000, 29.9, 31.6),   # train -0.4 -> fails 0.3
                r("voxel", 800_000, 29.8, 31.8),
                r("contrib_rank", 1_000_000, 29.0, 31.0)]
        ref, sug = self.mod.evaluate(rows, 0.3)
        self.assertEqual(ref["psnr"], 30.0)
        self.assertEqual(sug["opacity"]["n"], 1_000_000)
        self.assertEqual(sug["voxel"]["n"], 800_000)
        self.assertIn("between", sug["voxel"]["reason"])
        self.assertIsNone(sug["contrib_rank"]["n"])
        self.assertIn("larger", sug["contrib_rank"]["reason"])
        v600 = rows[3]
        self.assertEqual((v600["pass_0.1dB"], v600["pass_0.3dB"], v600["pass_0.5dB"]), (False, False, True))

    def test_only_one_count_tested_suggests_trying_smaller(self):
        rows = [self.row(m, 1_000_000, 30.0, 32.0) for m in S.METHODS]
        _, sug = self.mod.evaluate(rows, 0.3)
        self.assertTrue(all(s["n"] == 1_000_000 and "smaller" in s["reason"] for s in sug.values()))

    def test_missing_reference(self):
        _, sug = self.mod.evaluate([self.row("voxel", 600_000, 30.0, 32.0)], 0.3)
        self.assertIsNone(sug["voxel"]["n"])


if __name__ == "__main__":
    unittest.main()
