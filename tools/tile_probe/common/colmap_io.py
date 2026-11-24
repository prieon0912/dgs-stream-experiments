"""Minimal COLMAP binary reader + camera list in the same convention as graphdeco/3DGStream.

Cameras are sorted by image name, exactly like readColmapSceneInfo, so with --eval the
test camera is index 0 == cam00.
"""
import os
import struct
from dataclasses import dataclass

import numpy as np
from PIL import Image

_MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
           4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12)}


def read_cameras_bin(path):
    cams = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cid, mid, w, h = struct.unpack("<iiQQ", f.read(24))
            name, npar = _MODELS[mid]
            params = np.array(struct.unpack("<" + "d" * npar, f.read(8 * npar)))
            cams[cid] = dict(model=name, width=w, height=h, params=params)
    return cams


def read_images_bin(path):
    imgs = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            props = struct.unpack("<idddddddi", f.read(64))
            iid, qvec, tvec, cid = props[0], np.array(props[1:5]), np.array(props[5:8]), props[8]
            name = b""
            c = f.read(1)
            while c != b"\x00":
                name += c
                c = f.read(1)
            n2d = struct.unpack("<Q", f.read(8))[0]
            f.seek(24 * n2d, 1)
            imgs[iid] = dict(qvec=qvec, tvec=tvec, camera_id=cid, name=name.decode())
    return imgs


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


@dataclass
class Cam:
    name: str            # "cam00"
    image_path: str
    width: int           # actual image size on disk (e.g. images_2)
    height: int
    K: np.ndarray        # 3x3, scaled to (width, height)
    w2c: np.ndarray      # 4x4 OpenCV world->camera (COLMAP convention)

    @property
    def fovx(self):
        return 2 * np.arctan(self.width / (2 * self.K[0, 0]))

    @property
    def fovy(self):
        return 2 * np.arctan(self.height / (2 * self.K[1, 1]))

    @property
    def center(self):
        return -self.w2c[:3, :3].T @ self.w2c[:3, 3]


def load_cameras(frame_dir, images="images_2", exclude=(), only=None):
    sp = os.path.join(frame_dir, "sparse", "0")
    if not os.path.isdir(sp):
        sp = os.path.join(frame_dir, "sparse")
    intr = read_cameras_bin(os.path.join(sp, "cameras.bin"))
    extr = read_images_bin(os.path.join(sp, "images.bin"))
    img_dir = os.path.join(frame_dir, images)
    cams = []
    for im in extr.values():
        stem = os.path.splitext(im["name"])[0]
        if stem in exclude or (only is not None and stem not in only):
            continue
        path = os.path.join(img_dir, im["name"])
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        W, H = Image.open(path).size
        c = intr[im["camera_id"]]
        p = c["params"]
        if c["model"] == "SIMPLE_PINHOLE":
            fx = fy = p[0]; cx, cy = p[1], p[2]
        elif c["model"] == "PINHOLE":
            fx, fy, cx, cy = p[:4]
        else:
            raise ValueError(f"{c['model']}: expected undistorted (PINHOLE) cameras in {sp}")
        sx, sy = W / c["width"], H / c["height"]
        K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1]])
        w2c = np.eye(4)
        w2c[:3, :3] = qvec2rotmat(im["qvec"])
        w2c[:3, 3] = im["tvec"]
        cams.append(Cam(stem, path, W, H, K, w2c))
    cams.sort(key=lambda c: c.name)
    return cams


def nerfpp_radius(cams):
    """graphdeco getNerfppNorm radius (used as cameras_extent / spatial_lr_scale)."""
    centers = np.stack([c.center for c in cams])
    mid = centers.mean(0)
    return float(np.linalg.norm(centers - mid, axis=1).max() * 1.1)


def load_image(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
