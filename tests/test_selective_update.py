"""CPU tests of the selective-update logic, including the runtime hooks on fake 3DGStream modules."""
import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stream"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import selective_update as su  # noqa: E402

# camera: identity view, clip = (x, y, z, z) -> ndc = (x/z, y/z)
PROJ = torch.tensor([[1., 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]])
VIEW = torch.eye(4)


class MaskTests(unittest.TestCase):
    def test_change_mask_threshold_and_dilation(self):
        prev = torch.zeros(3, 20, 20)
        cur = prev.clone()
        cur[1, 10, 10] = 30 / 255
        cur[0, 2, 2] = 20 / 255                               # below tau=25
        m = su.change_mask(cur, prev, tau=25, radius=0)
        self.assertEqual(int(m.sum()), 1)
        self.assertTrue(m[10, 10])
        m2 = su.change_mask(cur, prev, tau=25, radius=2)
        self.assertEqual(int(m2.sum()), 25)
        u8 = su.change_mask((cur * 255).round().to(torch.uint8), torch.zeros(3, 20, 20, dtype=torch.uint8), 25, 0)
        self.assertTrue(torch.equal(u8, m))

    def test_projection_pixel_convention_and_depth(self):
        H = W = 10
        mask = torch.zeros(H, W, dtype=torch.bool)
        mask[:, 9] = True                                     # right-most column
        xyz = torch.tensor([[0.95, 0.0, 1.0],                  # ndc 0.95 -> px ((1.95)*10-1)/2 = 9.25 -> 9
                            [-0.95, 0.0, 1.0],                 # left column
                            [0.95, 0.0, -1.0],                 # behind camera
                            [3.0, 0.0, 1.0]])                  # outside image
        hit = su.project_hits(xyz, PROJ, VIEW, mask)
        self.assertEqual(hit.tolist(), [True, False, False, False])

    def test_kappa_views(self):
        xyz = torch.tensor([[0.9, 0.0, 1.0], [-0.9, 0.0, 1.0]])
        prev = torch.zeros(3, 10, 10)
        cur = prev.clone()
        cur[:, :, 7:] = 1.0
        both = [(cur, prev, PROJ, VIEW), (cur, prev, PROJ, VIEW)]
        one = [(cur, prev, PROJ, VIEW), (prev, prev, PROJ, VIEW)]
        self.assertEqual(su.select_by_change(xyz, both, kappa=2, radius=0)[0].tolist(), [True, False])
        self.assertEqual(su.select_by_change(xyz, one, kappa=2, radius=0)[0].tolist(), [False, False])
        self.assertEqual(su.select_by_change(xyz, one, kappa=1, radius=0)[0].tolist(), [True, False])


class MotionTests(unittest.TestCase):
    def test_full_frames(self):
        self.assertEqual([k for k in range(1, 70) if su.b_is_full_frame(k, 10, 30)],
                         list(range(1, 11)) + [30, 60])

    def test_history_freezes_and_remembers(self):
        h = su.MotionHistory(4, window=2, device="cpu")
        all_obs = torch.ones(4, dtype=torch.bool)
        h.push(torch.tensor([0.0, 0.2, 0.0, 0.0]), torch.zeros(4), all_obs)
        h.push(torch.tensor([0.0, 0.0, 0.3, 0.0]), torch.zeros(4), all_obs)
        self.assertEqual(su.select_by_motion(h, 0.1, float("inf")).tolist(), [False, True, True, False])
        # frozen gaussians are not observed: their score stays at the last observed value
        obs = torch.tensor([False, True, True, False])
        h.push(torch.tensor([0.0, 0.0, 0.0, 0.0]), torch.zeros(4), obs)
        h.push(torch.tensor([0.0, 0.0, 0.0, 0.0]), torch.zeros(4), obs)
        self.assertEqual(su.select_by_motion(h, 0.1, float("inf")).tolist(), [False, False, False, False])
        h2 = su.MotionHistory.from_state(h.state_dict(), "cpu")
        self.assertTrue(torch.equal(h2.score()[0], h.score()[0]))

    def test_never_observed_is_active_and_rotation_threshold(self):
        h = su.MotionHistory(2, window=3, device="cpu")
        self.assertEqual(su.select_by_motion(h, 0.1, 1.0).tolist(), [True, True])
        q = torch.tensor([[1.0, 0, 0, 0], [np.cos(np.deg2rad(1.0)), np.sin(np.deg2rad(1.0)), 0, 0]])
        ang = su.quat_angle_deg(q)
        self.assertAlmostEqual(float(ang[1]), 2.0, places=3)
        h.push(torch.zeros(2), ang, torch.ones(2, dtype=torch.bool))
        self.assertEqual(su.select_by_motion(h, 0.1, 1.0).tolist(), [False, True])

    def test_offline_replay_matches_online_history(self):
        mod = importlib.import_module("26_suggest_thresholds")
        rng = np.random.default_rng(0)
        n, frames = 500, 25
        moving = rng.random(n) < 0.3
        disp = np.where(moving, rng.uniform(0.05, 0.2, (frames, n)), rng.uniform(0, 0.02, (frames, n))).astype(np.float32)
        disp[:, :20] = np.nan                                   # outside the AABB
        ks = list(range(1, frames + 1))
        tau, window, warmup, recheck = 0.03, 4, 5, 12
        frac, _ = mod.replay(ks, disp, tau, window, warmup, recheck)
        inside = torch.tensor(~np.isnan(disp).all(0))
        h = su.MotionHistory(n, window, "cpu")
        online = []
        for k, d in zip(ks, disp):
            if su.b_is_full_frame(k, warmup, recheck):
                active = inside.clone()
            else:
                active = su.select_by_motion(h, tau, float("inf")) & inside
                online.append(float(active.sum() / inside.sum()))
            dd = torch.nan_to_num(torch.tensor(d), nan=0.0)
            h.push(dd, torch.zeros(n), active)
        self.assertAlmostEqual(frac, float(np.mean(online)), places=3)


# ------------------------------------------------------------------------- fake 3DGStream modules
def install_fake_upstream(move_fn):
    ntc = types.ModuleType("ntc")

    class NeuralTransformationCache(torch.nn.Module):
        def __init__(self, model, lo, hi):
            super().__init__()
            self.model, self.lo, self.hi = model, lo, hi

        def get_contracted_xyz(self, xyz):
            return (xyz - self.lo) / (self.hi - self.lo)

        def forward(self, xyz):                     # replaced by the hook
            raise AssertionError("upstream forward must be patched")

    ntc.NeuralTransformationCache = NeuralTransformationCache

    scene = types.ModuleType("scene")

    class Scene:
        def __init__(self, args, gaussians):
            self.cams = args.cams

        def getTrainCameras(self):
            return self.cams

    class GaussianModel:
        def __init__(self, xyz):
            self._xyz = xyz
            self._d_xyz = self._d_rot = None

        def training_one_frame_setup(self, args):
            self.ntc = NeuralTransformationCache(move_fn, torch.tensor(-1.0), torch.tensor(1.0))

        def query_ntc(self):
            self.mask, self._d_xyz, self._d_rot = self.ntc(self._xyz)

        def update_by_ntc(self):
            self._xyz = self._xyz + self._d_xyz.float() * 0       # keep positions fixed for the test
            self._d_xyz = self._d_rot = None

    scene.Scene, scene.GaussianModel = Scene, GaussianModel

    tf = types.ModuleType("train_frames")
    tf.state = {}

    def train_one_frame(lp, op, pp, args):
        g = tf.state.setdefault("g", GaussianModel(args.xyz))
        Scene(args, g)
        g.training_one_frame_setup(args)
        g.query_ntc()
        tf.state.setdefault("d_xyz", []).append(g._d_xyz.clone())
        g.update_by_ntc()
        return {}

    tf.train_one_frame = train_one_frame
    sys.modules.update(ntc=ntc, scene=scene, train_frames=tf)
    return tf


def moving_model(cx):
    """resi (8 dims): points with x > 0 move by 0.1 in x, the others do not move."""
    out = torch.zeros(cx.shape[0], 8, dtype=torch.half)
    out[:, 3] = 1.0
    out[:, 0] = torch.where(cx[:, 0] > 0.5, 0.1, 0.0).half()
    return out


class HookTests(unittest.TestCase):
    def setUp(self):
        self.saved = {k: sys.modules.get(k) for k in ("ntc", "scene", "train_frames")}
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        for k, v in self.saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        self.tmp.cleanup()

    def run_frames(self, argv, frames, cams=()):
        tf = install_fake_upstream(moving_model)
        tfs = importlib.import_module("train_frames_select")
        tfs = importlib.reload(tfs)
        log = os.path.join(self.tmp.name, "sel.jsonl")
        cfg, rest = tfs.own_args(argv + ["--selection_log", log])
        self.assertEqual(rest, [])
        tfs.SELECTOR = tfs.Selector(cfg, self.tmp.name, "images_2")
        tfs.install_hooks(tf)
        xyz = torch.tensor([[x, 0.0, 1.0] for x in np.linspace(-0.9, 0.9, 10)] + [[5.0, 0.0, 1.0]])  # last: outside AABB
        args = types.SimpleNamespace(xyz=xyz, cams=list(cams))
        prev = None
        for k in frames:
            args.source_path = f"/x/frame{k:06d}"
            if cams and prev is not None:
                tfs.SELECTOR.prev_images = prev
            tf.train_one_frame(None, None, None, args)
            if cams:
                prev = {c.image_name: (c.prev_img * 255).to(torch.uint8) for c in cams}
        recs = [json.loads(l) for l in open(log)]
        return recs, tf.state["d_xyz"], xyz

    def test_mode_a_updates_everything_inside(self):
        recs, dx, xyz = self.run_frames(["--update_mode", "A"], [1, 2, 3])
        self.assertTrue(all(r["n_selected"] == r["n_inside"] == 10 for r in recs))
        moving = xyz[:, 0] > 0
        moving[-1] = False
        self.assertTrue(torch.all(dx[0][moving, 0] > 0.09))
        self.assertEqual(float(dx[0][-1].abs().sum()), 0.0)      # outside AABB untouched
        self.assertIn("disp_q50", recs[0])

    def test_mode_b_freezes_static_gaussians_after_warmup(self):
        recs, dx, xyz = self.run_frames(["--update_mode", "B", "--b_tau_xyz", "0.05", "--b_window", "2",
                                         "--b_warmup", "2", "--b_recheck", "5", "--min_selected", "1"],
                                        [1, 2, 3, 4, 5, 6])
        sel = [r["n_selected"] for r in recs]
        self.assertEqual(sel, [10, 10, 5, 5, 10, 5])
        self.assertEqual([r["full_update"] for r in recs], [True, True, False, False, True, False])
        self.assertEqual(float(dx[2][xyz[:, 0] < 0].abs().sum()), 0.0)

    def test_mode_b_fallback_when_too_few_selected(self):
        recs, _, _ = self.run_frames(["--update_mode", "B", "--b_tau_xyz", "1.0", "--b_window", "2",
                                      "--b_warmup", "1", "--b_recheck", "0", "--min_selected", "3"], [1, 2])
        self.assertTrue(recs[1]["fallback"])
        self.assertEqual(recs[1]["n_selected"], 10)

    def test_mode_c_selects_gaussians_on_changed_pixels(self):
        H = W = 20
        prev = torch.zeros(3, H, W)
        cur = prev.clone()
        cur[:, :, 15:] = 1.0                                     # right quarter of the image changed

        class Cam:
            def __init__(self, name):
                self.image_name, self.original_image, self.prev_img = name, cur, prev
                self.full_proj_transform, self.world_view_transform = PROJ, VIEW

        cams = [Cam("cam01"), Cam("cam02")]
        tf_prev = {c.image_name: (prev * 255).to(torch.uint8) for c in cams}
        tf = install_fake_upstream(moving_model)
        tfs = importlib.reload(importlib.import_module("train_frames_select"))
        log = os.path.join(self.tmp.name, "c.jsonl")
        cfg, _ = tfs.own_args(["--update_mode", "C", "--c_radius", "0", "--min_selected", "1", "--selection_log", log])
        tfs.SELECTOR = tfs.Selector(cfg, self.tmp.name, "images_2")
        tfs.SELECTOR.prev_images = tf_prev
        tfs.install_hooks(tf)
        xyz = torch.tensor([[x, 0.0, 1.0] for x in np.linspace(-0.9, 0.9, 10)])
        args = types.SimpleNamespace(xyz=xyz, cams=cams, source_path="/x/frame000001")
        tf.train_one_frame(None, None, None, args)
        rec = json.loads(open(log).readline())
        # ndc x -> px = ((x+1)*20-1)/2 ; px >= 15  <=>  x >= 0.55
        expected = int((xyz[:, 0] >= 0.55).sum())
        self.assertEqual(rec["n_selected"], expected)
        self.assertGreater(rec["mask_frac_mean"], 0.2)
        # the cache for the next frame holds this frame's images
        self.assertTrue(torch.equal(tfs.SELECTOR.prev_images["cam01"], (cur * 255).to(torch.uint8)))


class BoxTests(unittest.TestCase):
    def test_box_frames(self):
        self.assertEqual([k for k in range(1, 32) if su.e_is_box_frame(k, 10)], [1, 11, 21, 31])
        self.assertEqual([k for k in range(1, 62) if su.e_is_box_frame(k, 30)], [1, 31, 61])

    def test_box_from_points_percentile_and_margin(self):
        p = torch.stack([torch.linspace(0, 1, 101), torch.zeros(101), torch.ones(101)], 1)
        lo, hi = su.box_from_points(p, 0.0, 1.0, 0.1)
        self.assertAlmostEqual(float(lo[0]), -0.1, places=3)
        self.assertAlmostEqual(float(hi[0]), 1.1, places=3)
        self.assertTrue(bool(su.in_box(torch.tensor([[0.5, 0.0, 1.0]]), lo, hi)[0]))
        self.assertFalse(bool(su.in_box(torch.tensor([[1.5, 0.0, 1.0]]), lo, hi)[0]))

    def test_accumulator_or_and_reset(self):
        acc = su.ChangeAccumulator(25.0, 0)
        a, b = torch.zeros(3, 4, 4), torch.zeros(3, 4, 4)
        b1 = b.clone(); b1[:, 0, 0] = 1
        b2 = b.clone(); b2[:, 3, 3] = 1
        acc.add("c", b1, a); acc.frame_done()
        acc.add("c", b2, a); acc.frame_done()
        self.assertEqual(int(acc.acc["c"].sum()), 2)
        self.assertEqual(acc.pairs, 2)
        acc.reset()
        self.assertEqual(acc.acc, {})


class BoxHookTests(HookTests):
    def _run(self, mode_args, frames):
        H = W = 20
        prev = torch.zeros(3, H, W)
        cur = prev.clone()
        cur[:, :, 15:] = 1.0                                     # changed pixels <=> x >= 0.55

        class Cam:
            def __init__(self, name):
                self.image_name, self.original_image, self.prev_img = name, cur, prev
                self.full_proj_transform, self.world_view_transform = PROJ, VIEW

        cams = [Cam("cam01"), Cam("cam02")]
        tf = install_fake_upstream(moving_model)
        tfs = importlib.reload(importlib.import_module("train_frames_select"))
        log = os.path.join(self.tmp.name, "e.jsonl")
        cfg, _ = tfs.own_args(mode_args + ["--c_radius", "0", "--min_selected", "1", "--e_min_points", "1",
                                           "--e_q", "0.0", "--e_margin", "0.0", "--selection_log", log])
        tfs.SELECTOR = tfs.Selector(cfg, self.tmp.name, "images_2")
        tfs.install_hooks(tf)
        xyz = torch.tensor([[x, 0.0, 1.0] for x in np.linspace(-0.9, 0.9, 10)] + [[5.0, 0.0, 1.0]])
        args = types.SimpleNamespace(xyz=xyz, cams=cams)
        for k in frames:
            tfs.SELECTOR.prev_images = {c.image_name: (prev * 255).to(torch.uint8) for c in cams}
            args.source_path = f"/x/frame{k:06d}"
            tf.train_one_frame(None, None, None, args)
        return [json.loads(l) for l in open(log)], tf.state["d_xyz"], xyz

    def test_mode_e_updates_only_inside_adaptive_box(self):
        recs, dx, xyz = self._run(["--update_mode", "E", "--e_every", "2"], [1, 2, 3])
        expected = int((xyz[:-1, 0] >= 0.55).sum())
        self.assertEqual([r["n_selected"] for r in recs], [expected] * 3)
        self.assertEqual([bool(r.get("box_recomputed")) for r in recs], [True, False, True])
        self.assertEqual(recs[0]["box_pairs"], 1)
        self.assertEqual(recs[2]["box_pairs"], 2)
        self.assertEqual(float(dx[0][xyz[:, 0] < 0.5].abs().sum()), 0.0)

    def test_mode_eb_freezes_inside_box(self):
        recs, _, xyz = self._run(["--update_mode", "EB", "--e_every", "10", "--b_tau_xyz", "0.05",
                                  "--b_window", "2", "--b_warmup", "1", "--b_recheck", "0"], [1, 2, 3])
        in_box = int((xyz[:-1, 0] >= 0.55).sum())
        self.assertEqual(recs[0]["n_selected"], in_box)          # B full frame = whole box
        self.assertTrue(recs[0]["full_update"])
        self.assertLessEqual(recs[1]["n_selected"], in_box)      # moving ones (x > 0) inside the box
        self.assertEqual(recs[1]["n_region"], in_box)

    def test_eb_requires_tau(self):
        tfs = importlib.import_module("train_frames_select")
        with self.assertRaises(SystemExit):
            tfs.own_args(["--update_mode", "EB", "--selection_log", "x"])


if __name__ == "__main__":
    unittest.main()
