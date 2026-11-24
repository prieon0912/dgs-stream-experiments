"""CPU tests for the third experiment's helpers (no GPU, no external repos needed)."""
import json
import os
import struct
import subprocess
import sys

import numpy as np
from PIL import Image

from alt_init.colmap_subset import make_train_scene
from alt_init.gaussian_ops import apply_sim3, random_subset
from common.colmap_io import load_cameras, read_images_bin
from common.geom import rotmat_to_quat_wxyz, umeyama
from common.ply_io import load_ply, save_ply

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _rand_rot(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _fake_frame(tmp, n_cams=5):
    sp = tmp / "frame" / "sparse" / "0"
    img = tmp / "frame" / "images_2"
    sp.mkdir(parents=True)
    img.mkdir(parents=True)
    with open(sp / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", n_cams))
        for i in range(n_cams):
            f.write(struct.pack("<iiQQ", i + 1, 1, 40, 30))           # PINHOLE
            f.write(struct.pack("<4d", 50.0, 50.0, 20.0, 15.0))
    rng = np.random.default_rng(0)
    with open(sp / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", n_cams))
        for i in range(n_cams):
            q = rotmat_to_quat_wxyz(_rand_rot(rng))
            f.write(struct.pack("<idddddddi", i + 1, *q, *rng.normal(size=3), i + 1))
            f.write(f"cam{i:02d}.png".encode() + b"\x00")
            f.write(struct.pack("<Q", 2))
            f.write(struct.pack("<ddq", 1.0, 2.0, 7) * 2)
            Image.fromarray(np.zeros((30, 40, 3), np.uint8)).save(img / f"cam{i:02d}.png")
    with open(sp / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", 0))
    return tmp / "frame"


def test_train_scene_drops_only_cam00(tmp_path):
    frame = _fake_frame(tmp_path)
    out = tmp_path / "scene"
    info = make_train_scene(str(frame), str(out))
    assert info["dropped"] == ["cam00"] and info["kept"] == [f"cam{i:02d}" for i in range(1, 5)]
    a = {c.name: c for c in load_cameras(str(frame), exclude=("cam00",))}
    b = {c.name: c for c in load_cameras(str(out))}
    assert sorted(a) == sorted(b)
    for k in a:
        assert np.allclose(a[k].w2c, b[k].w2c) and np.allclose(a[k].K, b[k].K)
    # binary layout readable again (no 2D points written)
    assert len(read_images_bin(str(out / "sparse" / "0" / "images.bin"))) == 4
    assert not (out / "images_2" / "cam00.png").exists()
    assert os.path.islink(out / "images_2" / "cam01.png")
    # the source scene is untouched
    assert len(read_images_bin(str(frame / "sparse" / "0" / "images.bin"))) == 5


def test_make_train_scene_script(tmp_path):
    frame = _fake_frame(tmp_path)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "30_make_train_scene.py"),
                        "--frame_dir", str(frame), "--out", str(tmp_path / "s")],
                       capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=ROOT))
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["n_kept"] == 4


def _cov(rot, logscale):
    q = rot / np.linalg.norm(rot, axis=1, keepdims=True)
    out = []
    for qi, si in zip(q, np.exp(logscale)):
        w, x, y, z = qi
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        out.append(R @ np.diag(si ** 2) @ R.T)
    return np.stack(out)


def _sh1_color(f_rest, d):
    # graphdeco band 1: color += C1 * (-y a0 + z a1 - x a2), channel-major (3, 3)
    a = f_rest.reshape(-1, 3, 3)
    x, y, z = d
    return 0.4886025119029199 * (-y * a[:, :, 0] + z * a[:, :, 1] - x * a[:, :, 2])


def _gauss(rng, n=50, deg=1):
    k = (deg + 1) ** 2 - 1
    return dict(xyz=rng.normal(size=(n, 3)).astype(np.float32), f_dc=rng.normal(size=(n, 3)).astype(np.float32),
                f_rest=rng.normal(size=(n, 3 * k)).astype(np.float32),
                opacity=rng.normal(size=n).astype(np.float32),
                scale=rng.normal(size=(n, 3)).astype(np.float32) - 3,
                rot=rng.normal(size=(n, 4)).astype(np.float32))


def test_sim3_moves_positions_covariances_and_sh1_consistently():
    rng = np.random.default_rng(1)
    d = _gauss(rng)
    s, R, t = 2.5, _rand_rot(rng), rng.normal(size=3)
    e = apply_sim3(d, s, R, t)
    assert np.allclose(e["xyz"], s * d["xyz"] @ R.T + t, atol=1e-4)
    assert np.allclose(_cov(e["rot"], e["scale"]), s * s * R @ _cov(d["rot"], d["scale"]) @ R.T, atol=1e-4)
    for _ in range(5):
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        assert np.allclose(_sh1_color(e["f_rest"], R @ v), _sh1_color(d["f_rest"], v), atol=1e-4)
    assert np.array_equal(e["f_dc"], d["f_dc"]) and np.array_equal(e["opacity"], d["opacity"])


def test_sim3_zeroes_unrotated_higher_bands():
    rng = np.random.default_rng(2)
    e = apply_sim3(_gauss(rng, deg=3), 1.0, _rand_rot(rng), np.zeros(3))
    rest = e["f_rest"].reshape(-1, 3, 15)
    assert np.all(rest[:, :, 3:] == 0) and np.any(rest[:, :, :3] != 0)


def test_sim3_from_camera_centres_round_trip():
    rng = np.random.default_rng(3)
    c_gt = rng.normal(size=(19, 3)) * 3
    s, R, t = 0.37, _rand_rot(rng), rng.normal(size=3)
    c_pred = (c_gt - t) @ R / s                       # inverse map: pred frame
    s2, R2, t2 = umeyama(c_pred, c_gt)
    assert abs(s2 - s) < 1e-8 and np.allclose(R2, R) and np.allclose(t2, t)


def test_random_subset_exact_and_reproducible(tmp_path):
    rng = np.random.default_rng(4)
    d = _gauss(rng, n=100)
    a, ia = random_subset(d, 30, seed=0)
    b, ib = random_subset(d, 30, seed=0)
    assert len(a["xyz"]) == 30 and np.array_equal(ia, ib) and np.all(np.diff(ia) > 0)
    full, _ = random_subset(d, 500)
    assert len(full["xyz"]) == 100
    save_ply(a, str(tmp_path / "a.ply"))
    assert np.allclose(load_ply(str(tmp_path / "a.ply"))["xyz"], a["xyz"])


def _fake_condition(base, name, psnr, train, n, t, final="final.ply"):
    d = base / name
    (d / "static").mkdir(parents=True)
    json.dump(dict(final=dict(n=n, psnr=psnr, train_psnr_mean=train, ssim=0.96, lpips=0.17,
                              aabb_inside_count=int(0.9 * n), aabb_inside_frac=0.9)),
              open(d / "static" / "static.json", "w"))
    json.dump(dict(init_time_sec=t, steps={"x": t}, gpu_exclusive=True),
              open(d / (os.path.splitext(final)[0] + "_timing.json"), "w"))
    return d


def test_summary_uses_both_views_and_reference(tmp_path):
    ref = _fake_condition(tmp_path / "prune", "prune_s0_opacity_1000000", 34.35, 32.95, 1_000_000, 80,
                          final="refined.ply")
    init = tmp_path / "alt"
    _fake_condition(init, "alt_s0_taming_600000", 34.20, 32.80, 600_000, 200)   # -0.15 both: 0.3 pass
    _fake_condition(init, "alt_s0_edgs_600000", 34.50, 32.40, 600_000, 90)      # train -0.55: all fail
    _fake_condition(init, "alt_s0_zipsplat_1.0", 30.0, 29.0, 150_000, 30)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "34_summarize_alt_init.py"),
                        "--root", str(init), "--ref", str(ref), "--out", str(tmp_path / "res")],
                       capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=ROOT))
    assert r.returncode == 0, r.stderr
    rows = {(x["method"], x["param"]): x for x in json.load(open(tmp_path / "res" / "alt_init.json"))["rows"]}
    t = rows[("taming", "600000")]
    assert t["pass_0.1dB"] is False and t["pass_0.3dB"] is True and t["pass_0.5dB"] is True
    e = rows[("edgs", "600000")]
    assert e["pass_0.5dB"] is False                    # cam00 better, but train views 0.55 dB worse
    assert rows[("da3_opacity", "1000000")]["init_time_sec"] == 80
    assert rows[("zipsplat", "1.0")]["pass_0.5dB"] is False


def test_edgs_short_bookkeeping(tmp_path):
    from alt_init.edgs_short import best, exclude
    for n, vis in ((1_000_000, 750_000), (800_000, 700_000)):
        d = tmp_path / f"alt_s0_edgs_{n}"
        d.mkdir()
        json.dump(dict(n_target=n, n_visible=vis), open(d / "init_ALL_edgs.json", "w"))
        (d / "init_ALL.ply").write_bytes(b"x")
        exclude(d)
        assert json.load(open(d / "excluded.json"))["n_visible"] == vis
    (tmp_path / "alt_s0_edgs_600000").mkdir()
    assert best(tmp_path, "alt_s0").endswith("alt_s0_edgs_1000000")
