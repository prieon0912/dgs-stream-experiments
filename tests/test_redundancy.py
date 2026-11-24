"""CPU tests for common/redundancy.py (official redundancy score port + reprojection duplicates)."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import redundancy as rd  # noqa: E402


def perspective_full(fov_deg=60.0, znear=0.01, zfar=100.0):
    """3DGS full_proj_transform for an identity camera (row-vector convention)."""
    t = math.tan(math.radians(fov_deg) / 2)
    P = torch.zeros(4, 4)
    P[0, 0] = 1 / t
    P[1, 1] = 1 / t
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P.T.contiguous()                    # world_view = I


IDENT_Q = torch.tensor([1.0, 0.0, 0.0, 0.0])


class RotationTests(unittest.TestCase):
    def test_quat_to_rot_z90(self):
        q = torch.tensor([[math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]])
        R = rd.quat_to_rot(q)[0]
        np.testing.assert_allclose(R @ torch.tensor([1.0, 0, 0]), [0, 1, 0], atol=1e-6)


class PixelSizeTests(unittest.TestCase):
    def test_pixel_size_matches_pinhole_and_invisible_stays_10000(self):
        full = perspective_full(60.0)
        xyz = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 4.0], [0.0, 0.0, -1.0]])
        W, H = 1000, 800
        ps = rd.min_projected_pixel_size(xyz, [(full, W, H)])
        expect = 2 * 2.0 * math.tan(math.radians(30)) / W     # one pixel at depth 2
        self.assertAlmostEqual(float(ps[0]), expect, delta=expect * 0.02)
        self.assertAlmostEqual(float(ps[1]) / float(ps[0]), 2.0, delta=0.05)
        self.assertEqual(float(ps[2]), 10000.0)


class RedundancyTests(unittest.TestCase):
    def make_cloud(self):
        rng = np.random.default_rng(0)
        cluster = rng.normal(scale=0.001, size=(12, 3)) + [0, 0, 2]       # 12 overlapping gaussians
        sparse = np.stack([np.linspace(-1, 1, 12), np.full(12, 0.5), np.full(12, 2.0)], 1)   # well separated
        xyz = torch.tensor(np.concatenate([cluster, sparse]), dtype=torch.float32)
        scale = torch.full((24, 3), 0.01)
        rot = IDENT_Q.repeat(24, 1)
        return xyz, scale, rot

    def test_cluster_scores_high_isolated_score_one(self):
        xyz, scale, rot = self.make_cloud()
        ps = torch.full((24,), 1e-4)
        nbr = torch.from_numpy(rd.knn_indices(xyz.numpy().astype(np.float64), k=5))
        red = rd.redundancy_score(xyz, scale, rot, ps, nbr)
        self.assertTrue(torch.all(red[:12] == 6))          # 1 + all 5 neighbours cover it
        self.assertTrue(torch.all(red[12:] == 1))

    def test_min_over_containing_spheres(self):
        # a: crowded (covered by 3), b sits inside a's neighbourhood but only b's own sphere is sparse.
        xyz = torch.tensor([[0, 0, 0], [0.001, 0, 0], [0.002, 0, 0], [0.003, 0, 0], [0.5, 0, 0]], dtype=torch.float32)
        scale = torch.tensor([[0.01] * 3] * 4 + [[0.0001] * 3])
        rot = IDENT_Q.repeat(5, 1)
        ps = torch.full((5,), 1e-5)
        nbr = torch.from_numpy(rd.knn_indices(xyz.numpy().astype(np.float64), k=4))
        red = rd.redundancy_score(xyz, scale, rot, ps, nbr)
        self.assertEqual(int(red[4]), 1)                    # isolated, tiny
        self.assertTrue(torch.all(red[:4] == 5) or torch.all(red[:4] >= 4))

    def test_select_reaches_exact_n_and_prunes_cluster_first(self):
        xyz, scale, rot = self.make_cloud()
        op = torch.linspace(-1, 1, 24)                      # sparse points have HIGHER opacity
        op[:12] = op[:12] + 3                               # ...except: give the cluster high opacity too
        full = perspective_full(60.0)
        # with only 5 neighbours mean+1*std (=6.05) exceeds the cluster score (6); lambda 0.5 -> threshold 4.8
        kept, info = rd.redundancy_select(xyz, scale, rot, op, [(full, 1000, 800)], 18, k=5, lam=0.5,
                                          log=lambda *_: None)
        self.assertEqual(len(kept), 18)
        self.assertEqual(len(torch.unique(kept)), 18)
        removed = set(range(24)) - set(kept.tolist())
        self.assertTrue(removed <= set(range(12)), removed)  # only cluster members removed
        self.assertGreater(info["removed_by_redundancy"], 0)

    def test_cached_order_equals_iterative_for_every_n(self):
        rng = np.random.default_rng(3)
        cl = [rng.normal(scale=0.002, size=(15, 3)) + c for c in ([0, 0, 2], [0.3, 0.1, 2.5], [-0.2, 0.2, 1.8])]
        xyz = torch.tensor(np.concatenate(cl + [rng.uniform(-1, 1, (25, 3)) + [0, 0, 3]]), dtype=torch.float32)
        n = xyz.shape[0]
        scale = torch.full((n, 3), 0.01)
        rot = IDENT_Q.repeat(n, 1)
        op = torch.tensor(rng.normal(size=n), dtype=torch.float32)
        cams = [(perspective_full(), 1000, 800)]
        order, n_red, _ = rd.redundancy_removal_order(xyz, scale, rot, op, cams, k=8, lam=0.5, log=lambda *_: None)
        self.assertEqual(sorted(order.tolist()), list(range(n)))
        for target in (60, 50, 40, 30, 10):
            a, _ = rd.redundancy_select(xyz, scale, rot, op, cams, target, k=8, lam=0.5, log=lambda *_: None)
            b, info = rd.select_from_order(order, n, target, n_red)
            self.assertEqual(a.tolist(), b.tolist(), target)
            self.assertEqual(info["removed_by_redundancy"] + info["removed_by_opacity_fallback"], n - target)

    def test_once_order_removes_candidates_first(self):
        xyz, scale, rot = self.make_cloud()
        op = torch.linspace(1, -1, 24)                       # cluster has HIGH opacity, sparse low
        order, n_cand, rounds = rd.redundancy_once_order(xyz, scale, rot, op, [(perspective_full(), 1000, 800)],
                                                         k=5, lam=0.5)
        self.assertEqual(n_cand, 12)
        self.assertEqual(set(order[:12].tolist()), set(range(12)))   # candidates first despite high opacity
        kept, info = rd.select_from_order(order, 24, 18, n_cand)
        self.assertEqual(info["removed_by_redundancy"], 6)

    def test_fallback_when_no_candidates(self):
        xyz = torch.tensor([[x, 0, 2.0] for x in np.linspace(-1, 1, 10)], dtype=torch.float32)
        scale = torch.full((10, 3), 1e-4)
        rot = IDENT_Q.repeat(10, 1)
        op = torch.arange(10, dtype=torch.float32)
        kept, info = rd.redundancy_select(xyz, scale, rot, op, [(perspective_full(), 1000, 800)], 6, k=3,
                                          log=lambda *_: None)
        self.assertEqual(kept.tolist(), [4, 5, 6, 7, 8, 9])
        self.assertEqual(info["removed_by_opacity_fallback"], 4)


class ReprojectionTests(unittest.TestCase):
    def setUp(self):
        self.full = perspective_full(60.0)
        self.wv = torch.eye(4)

    def views(self, order=(0, 1)):
        return [dict(view_id=i, name=f"v{i}", full_proj=self.full, world_view=self.wv, W=200, H=200) for i in order]

    def test_same_surface_is_duplicate_front_and_back_are_not(self):
        g = np.linspace(-0.5, 0.5, 5)
        plane0 = np.array([[x, y, 2.0] for x in g for y in g])                # view 0: surface at z=2
        plane1 = plane0 * [1, 1, 1.005]                                       # view 1: same surface, 0.5% deeper
        front = np.array([[0.0, 0.0, 1.0]])                                   # view 1: thin object in front
        back = np.array([[0.25, 0.25, 3.0]])                                  # view 1: behind the surface
        xyz = torch.tensor(np.concatenate([plane0, plane1, front, back]), dtype=torch.float32)
        vid = torch.tensor([0] * 25 + [1] * 25 + [1, 1])
        dup, stats = rd.reprojection_duplicates(xyz, vid, self.views(), rel_tol=0.02, cell_px=3,
                                                log=lambda *_: None)
        self.assertFalse(dup[:25].any())                    # first view is the reference
        self.assertTrue(dup[25:50].all())
        self.assertFalse(dup[50])                           # in front: kept
        self.assertFalse(dup[51])                           # behind: kept
        self.assertEqual(stats[1]["duplicates"], 25)

    def test_tolerance_and_exact_count(self):
        plane0 = np.array([[0.0, 0.0, 2.0], [0.2, 0.0, 2.0]])
        plane1 = np.array([[0.0, 0.0, 2.2], [0.2, 0.0, 2.01]])                 # 10% (not dup) and 0.5% (dup)
        xyz = torch.tensor(np.concatenate([plane0, plane1]), dtype=torch.float32)
        vid = torch.tensor([0, 0, 1, 1])
        op = torch.tensor([0.0, 1.0, 2.0, 3.0])
        kept, info = rd.reprojection_select(xyz, vid, op, self.views(), 3, rel_tol=0.02, log=lambda *_: None)
        self.assertEqual(info["n_duplicates"], 1)
        self.assertEqual(kept.tolist(), [0, 1, 2])
        kept, info = rd.reprojection_select(xyz, vid, op, self.views(), 4, log=lambda *_: None)
        self.assertEqual(info["added_back_by_opacity"], 1)
        kept, info = rd.reprojection_select(xyz, vid, op, self.views(), 2, log=lambda *_: None)
        self.assertEqual(kept.tolist(), [1, 2])             # 3 unique -> keep the 2 highest opacity


if __name__ == "__main__":
    unittest.main()
