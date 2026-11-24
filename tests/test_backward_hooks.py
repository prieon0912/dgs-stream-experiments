"""CPU tests of the backward-speed hooks in stream/train_frames_select.py (fake 3DGStream modules).

Phase 1: --fused_ssim. (Phase 3 adds the rasterizer / grad-skip hooks here.)"""
import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stream"))
sys.path.insert(0, str(ROOT))

MODS = ("ntc", "scene", "train_frames", "arguments", "fused_ssim")


def upstream_ssim(img1, img2, window_size=11, size_average=True):     # stand-in for utils.loss_utils.ssim
    return 1.0 - (img1 - img2).abs().mean()


def install_fakes(with_fused=True):
    ntc = types.ModuleType("ntc")
    ntc.NeuralTransformationCache = type("NeuralTransformationCache", (), {"forward": None})
    scene = types.ModuleType("scene")
    scene.Scene = type("Scene", (), {"__init__": lambda self, *a, **k: None})
    scene.GaussianModel = type("GaussianModel", (), {"training_one_frame_setup": lambda self, a: None,
                                                      "update_by_ntc": lambda self: None})
    tf = types.ModuleType("train_frames")
    tf.ssim = upstream_ssim
    tf.calls = []

    def train_frames(lp, op, pp, args):
        tf.calls.append("train_frames")
        tf.ssim_at_train = tf.ssim
    tf.train_frames = train_frames
    tf.train_one_frame = lambda *a: None

    arguments = types.ModuleType("arguments")

    class ModelParams:
        def __init__(self, parser):
            parser.add_argument("-o", "--output_path", default="")
            parser.add_argument("-m", "--model_path", default="")
            parser.add_argument("-v", "--video_path", default="")
            parser.add_argument("--images", default="images")
    arguments.ModelParams = ModelParams
    arguments.OptimizationParams = arguments.PipelineParams = lambda parser: None
    mods = dict(ntc=ntc, scene=scene, train_frames=tf, arguments=arguments)
    if with_fused:
        fs = types.ModuleType("fused_ssim")
        fs.seen = []

        def fused_ssim(img1, img2, padding="same", train=True):
            fs.seen.append((tuple(img1.shape), tuple(img2.shape), padding, train))
            return upstream_ssim(img1, img2)
        fs.fused_ssim = fused_ssim
        mods["fused_ssim"] = fs
    else:
        mods["fused_ssim"] = None          # sys.modules[name] = None -> ImportError on import
    sys.modules.update(mods)
    return tf


class FusedSsimHookTests(unittest.TestCase):
    def setUp(self):
        self.saved = {k: sys.modules.get(k, "absent") for k in MODS}
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        for k, v in self.saved.items():
            if v == "absent":
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        self.tmp.cleanup()

    def _main(self, extra, with_fused=True):
        tf = install_fakes(with_fused)
        tfs = importlib.reload(importlib.import_module("train_frames_select"))
        tfs._install_raster_counter = lambda: None
        out = Path(self.tmp.name) / "o"
        old = sys.argv
        sys.argv = ["x", "--selection_log", str(Path(self.tmp.name) / "s.jsonl"), "-o", str(out), "-m", "m",
                    "-v", "v"] + extra
        try:
            tfs.main()
        finally:
            sys.argv = old
        return tf, tfs, json.loads((out / "cfg_args.json").read_text())

    def test_default_off_leaves_upstream_ssim(self):
        tf, tfs, cfg = self._main([])
        self.assertEqual(tf.calls, ["train_frames"])
        self.assertIs(tf.ssim_at_train, upstream_ssim)
        self.assertEqual(cfg["selective_update"]["fused_ssim"], "off")
        sel = tfs.Selector(tfs.own_args(["--selection_log", str(Path(self.tmp.name) / "a.jsonl")])[0], ".", "i")
        sel.begin_frame("frame000001")
        self.assertNotIn("fused_ssim", sel.rec)                      # per-frame record unchanged when off

    def test_on_replaces_ssim_and_adds_batch_dim(self):
        tf, tfs, cfg = self._main(["--fused_ssim", "on"])
        self.assertIsNot(tf.ssim_at_train, upstream_ssim)
        self.assertTrue(getattr(tf.ssim_at_train, "fused", False))
        a, b = torch.rand(3, 6, 8), torch.rand(3, 6, 8)
        v = tf.ssim_at_train(a, b)                                  # how train_frames calls it: (C,H,W)
        self.assertAlmostEqual(float(v), float(upstream_ssim(a, b)), places=6)
        self.assertEqual(sys.modules["fused_ssim"].seen[-1], ((1, 3, 6, 8), (1, 3, 6, 8), "same", True))
        tf.ssim_at_train(a[None], b[None])                          # already batched: unchanged
        self.assertEqual(sys.modules["fused_ssim"].seen[-1][0], (1, 3, 6, 8))
        with self.assertRaises(ValueError):
            tf.ssim_at_train(a, b, window_size=7)
        sel = tfs.Selector(tfs.own_args(["--fused_ssim", "on", "--selection_log",
                                         str(Path(self.tmp.name) / "b.jsonl")])[0], ".", "i")
        sel.begin_frame("frame000001")
        self.assertTrue(sel.rec["fused_ssim"])

    def test_on_without_package_is_a_clear_error(self):
        with self.assertRaises(SystemExit) as cm:
            self._main(["--fused_ssim", "on"], with_fused=False)
        self.assertIn("06_install_fused_ssim.sh", str(cm.exception))

    def test_gradient_flows_to_img1(self):
        install_fakes(True)
        tfs = importlib.reload(importlib.import_module("train_frames_select"))
        ssim = tfs.make_fused_ssim(sys.modules["fused_ssim"].fused_ssim)
        a = torch.rand(3, 4, 4, requires_grad=True)
        ssim(a, torch.rand(3, 4, 4)).backward()
        self.assertEqual(tuple(a.grad.shape), (3, 4, 4))

    def test_bad_choice_rejected(self):
        tfs = importlib.import_module("train_frames_select")
        with self.assertRaises(SystemExit):
            tfs.own_args(["--fused_ssim", "yes", "--selection_log", "x"])


if __name__ == "__main__":
    unittest.main()
