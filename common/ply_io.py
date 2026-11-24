"""graphdeco-compatible 3DGS PLY I/O (numpy only).

Stored (pre-activation) convention, identical to GaussianModel.save_ply/load_ply:
  xyz (N,3) | f_dc (N,3) SH DC | f_rest (N, 3*(K-1)) channel-major | opacity logit (N,)
  scale log (N,3) | rot wxyz unnormalized (N,4)
"""
import numpy as np
from plyfile import PlyData, PlyElement

C0 = 0.28209479177387814


def _sorted(names, prefix):
    return sorted([n for n in names if n.startswith(prefix)], key=lambda x: int(x.split("_")[-1]))


def load_ply(path):
    v = PlyData.read(path).elements[0]
    names = [p.name for p in v.properties]
    g = lambda ks: np.stack([np.asarray(v[k], dtype=np.float32) for k in ks], 1)
    rest = _sorted(names, "f_rest_")
    d = dict(
        xyz=g(["x", "y", "z"]),
        f_dc=g(["f_dc_0", "f_dc_1", "f_dc_2"]),
        f_rest=g(rest) if rest else np.zeros((len(v.data), 0), np.float32),
        opacity=np.asarray(v["opacity"], dtype=np.float32),
        scale=g(_sorted(names, "scale_")),
        rot=g(_sorted(names, "rot")),
    )
    return d


def sh_degree_of(d):
    k = d["f_rest"].shape[1] // 3 + 1
    deg = int(round(np.sqrt(k))) - 1
    assert (deg + 1) ** 2 == k, f"bad f_rest count {d['f_rest'].shape[1]}"
    return deg


def set_sh_degree(d, deg):
    """Pad with zeros / truncate higher bands. Layout is (3, K-1) channel-major."""
    n = len(d["xyz"])
    cur = sh_degree_of(d)
    k_cur, k_new = (cur + 1) ** 2 - 1, (deg + 1) ** 2 - 1
    r = d["f_rest"].reshape(n, 3, k_cur)
    out = np.zeros((n, 3, k_new), np.float32)
    m = min(k_cur, k_new)
    out[:, :, :m] = r[:, :, :m]
    d = dict(d)
    d["f_rest"] = out.reshape(n, 3 * k_new)
    return d


def save_ply(d, path):
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = len(d["xyz"])
    cols = [("x", d["xyz"][:, 0]), ("y", d["xyz"][:, 1]), ("z", d["xyz"][:, 2]),
            ("nx", np.zeros(n)), ("ny", np.zeros(n)), ("nz", np.zeros(n))]
    cols += [(f"f_dc_{i}", d["f_dc"][:, i]) for i in range(3)]
    cols += [(f"f_rest_{i}", d["f_rest"][:, i]) for i in range(d["f_rest"].shape[1])]
    cols += [("opacity", d["opacity"].reshape(-1))]
    cols += [(f"scale_{i}", d["scale"][:, i]) for i in range(3)]
    cols += [(f"rot_{i}", d["rot"][:, i]) for i in range(4)]
    arr = np.empty(n, dtype=[(k, "f4") for k, _ in cols])
    for k, c in cols:
        arr[k] = c
    PlyData([PlyElement.describe(arr, "vertex")]).write(path)


def subset(d, idx):
    return {k: v[idx] for k, v in d.items()}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def rgb2sh(rgb):
    return (rgb - 0.5) / C0


def stats(d):
    op = sigmoid(d["opacity"])
    sc = np.exp(d["scale"]).max(1)
    return dict(n=int(len(d["xyz"])), sh_degree=sh_degree_of(d),
                xyz_p1=np.percentile(d["xyz"], 1, 0).round(2).tolist(),
                xyz_p99=np.percentile(d["xyz"], 99, 0).round(2).tolist(),
                opacity_mean=float(op.mean()), opacity_gt_0_5=float((op > 0.5).mean()),
                max_scale_median=float(np.median(sc)))
