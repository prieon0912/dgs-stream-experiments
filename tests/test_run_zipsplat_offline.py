"""Runs alt_init/run_zipsplat.py end to end on CPU with a fake `zipsplat` package.

Checks the parts that can break without a GPU: Sim3 from predicted camera centres, the NaN filter,
the opacity cap (--max_gaussians), the failure json, and the output PLY/timing files.
"""
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_alt_init import _fake_frame  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAKE = textwrap.dedent('''
    import json, os, numpy as np, torch
    from plyfile import PlyData, PlyElement
    torch.cuda.synchronize = lambda *a, **k: None
    torch.cuda.reset_peak_memory_stats = lambda *a, **k: None
    torch.cuda.max_memory_allocated = lambda *a, **k: 0
    CFG = json.load(open(os.environ["FAKE_ZIP_CFG"]))

    def load_image(p):
        from PIL import Image
        return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)

    class _Pose:
        def __init__(self, R, t): self.R, self.t = torch.tensor(R), torch.tensor(t)

    class _G:
        def __init__(self, xyz, op): self.xyz, self.op = xyz, op
        def save_ply(self, path):
            n = len(self.xyz)
            names = ["x","y","z","nx","ny","nz","f_dc_0","f_dc_1","f_dc_2"] + [f"f_rest_{i}" for i in range(9)] + \\
                    ["opacity","scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3"]
            data = np.zeros((n, len(names)), np.float32)
            data[:, :3] = self.xyz; data[:, 18] = self.op; data[:, 19:22] = -3; data[:, 22] = 1
            el = np.empty(n, dtype=[(k, "f4") for k in names]); el[:] = list(map(tuple, data))
            PlyData([PlyElement.describe(el, "vertex")]).write(path)

    class _Head: gaussians_per_token = 32
    class _M: gaussian_head = _Head()

    class ZipSplat:
        input_sizes = [(504, 378)]
        def __init__(self, weights): self.model = _M()
        def cuda(self): return self
        def eval(self): return self
        def __call__(self, images, compression=1.0, ba=True):
            return dict(gaussians=_G(np.array(CFG["xyz"], np.float32), np.array(CFG["op"], np.float32)),
                        poses=_Pose(CFG["R_pred"][:len(images)], CFG["c_pred"][:len(images)]))
''')


def _setup(tmp_path, n_cams=5, s=0.5, nan=0, n=40):
    from common.colmap_io import load_cameras
    frame = _fake_frame(tmp_path, n_cams=n_cams)
    cams = load_cameras(str(frame), exclude=("cam00",))
    rng = np.random.default_rng(0)
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    t = rng.normal(size=3)
    c_gt = np.stack([c.center for c in cams])
    R_gt = np.stack([c.w2c[:3, :3].T for c in cams])
    c_pred = (c_gt - t) @ R / s                    # world = s R pred + t
    R_pred = np.einsum("ij,njk->nik", R.T, R_gt)
    xyz_world = rng.normal(size=(n, 3))
    xyz_pred = (xyz_world - t) @ R / s
    op = np.linspace(-3, 3, n)
    if nan:
        xyz_pred[:nan] = np.nan
    pkg = tmp_path / "fakepkg" / "zipsplat"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(FAKE)
    cfg = tmp_path / "cfg.json"
    json.dump(dict(xyz=xyz_pred.tolist(), op=op.tolist(), c_pred=c_pred.tolist(), R_pred=R_pred.tolist()),
              open(cfg, "w"))
    env = dict(os.environ, PYTHONPATH=f"{tmp_path / 'fakepkg'}:{ROOT}", FAKE_ZIP_CFG=str(cfg))
    return frame, env, xyz_world, op


def _run(frame, out, env, *extra):
    return subprocess.run([sys.executable, os.path.join(ROOT, "alt_init", "run_zipsplat.py"), "--frame_dir", str(frame),
                           "--out", str(out), *extra], capture_output=True, text=True, env=env)


def test_aligned_and_capped(tmp_path):
    from common.ply_io import load_ply
    frame, env, xyz_world, op = _setup(tmp_path, nan=2, n=400)
    out = tmp_path / "o" / "init.ply"
    out.parent.mkdir()
    r = _run(frame, out, env, "--max_gaussians", "200")
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.load(open(tmp_path / "o" / "init_zipsplat.json"))
    assert rep["n_raw"] == 400 and rep["n_nonfinite_dropped"] == 2 and rep["n_capped_by_opacity"] == 198
    assert rep["center_residual_rel"] < 1e-6 and rep["rotation_error_deg_median"] < 1e-3 and rep["alignment_pass"]
    d = load_ply(str(out))
    assert len(d["xyz"]) == 200
    top = np.argsort(-op)[:200]                      # most opaque 200 (the NaN rows have the lowest opacity)
    got = d["xyz"][np.argsort(-d["opacity"], kind="stable")]
    assert np.allclose(got, xyz_world[top], atol=1e-4)
    assert d["f_rest"].shape[1] == 9
    t = json.load(open(tmp_path / "o" / "init_timing.json"))
    assert "zipsplat_infer_ba" in t["steps"] and "align_convert" in t["steps"]


def test_too_many_nans_fails_with_json(tmp_path):
    frame, env, _, _ = _setup(tmp_path, nan=10)
    out = tmp_path / "init.ply"
    r = _run(frame, out, env)
    assert r.returncode == 7 and not out.exists()
    assert "NaN" in json.load(open(tmp_path / "init_zipsplat_failed.json"))["reason"]


def test_view_count_guard(tmp_path):
    frame, env, _, _ = _setup(tmp_path)
    r = _run(frame, tmp_path / "init.ply", env, "--max_views", "3")
    assert r.returncode == 5
