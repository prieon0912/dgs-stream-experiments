"""Small numpy operations on graphdeco-style PLY dicts (common.ply_io layout).

  apply_sim3(d, s, R, t)   x' = s R x + t, with rotations, log-scales and SH band 1 rotated too
  random_subset(d, n, seed) exact-count subsample (documented, reproducible)
"""
import numpy as np

from common.geom import quat_mul, rotate_sh1, rotmat_to_quat_wxyz
from common.ply_io import sh_degree_of, subset


def apply_sim3(d, s, R, t):
    s = float(s)
    if not np.isfinite(s) or s <= 0:
        raise ValueError(f"invalid Sim3 scale {s}")
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    out = dict(d)
    out["xyz"] = (s * d["xyz"].astype(np.float64) @ R.T + t).astype(np.float32)
    q = d["rot"].astype(np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    out["rot"] = quat_mul(rotmat_to_quat_wxyz(R), q).astype(np.float32)
    out["scale"] = (d["scale"].astype(np.float64) + np.log(s)).astype(np.float32)
    deg = sh_degree_of(d)
    if deg >= 1:
        n = len(d["xyz"])
        k = (deg + 1) ** 2 - 1
        rest = d["f_rest"].reshape(n, 3, k).astype(np.float64).copy()
        rest[:, :, 0:3] = rotate_sh1(rest[:, :, 0:3], R)
        if deg >= 2:
            # bands 2/3 are not rotated (zeroed instead of left wrong); the 1k refine re-learns them
            rest[:, :, 3:] = 0.0
        out["f_rest"] = rest.reshape(n, 3 * k).astype(np.float32)
    return out


def random_subset(d, n, seed=0):
    total = len(d["xyz"])
    if n >= total:
        return d, np.arange(total)
    idx = np.sort(np.random.default_rng(seed).choice(total, size=n, replace=False))
    return subset(d, idx), idx
