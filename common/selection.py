"""First-frame gaussian selection (pruning) methods, numpy core.

All functions return a sorted int64 index array of exactly ``n`` entries (or all indices when the
input already has <= n gaussians). The GPU part (render-contribution scores) lives in
scripts/23_select_gaussians.py; everything here is CPU-only and unit-tested.

  opacity        top-N by opacity                         (what 06_subsample.py did)
  voxel          voxel grid, keep the max-opacity gaussian per occupied voxel; the voxel size is found
                 by bisection so that #occupied voxels >= N, then trimmed to exactly N by opacity
  contrib_rank   top-N by render-contribution score       (LightGaussian global significance)
  contrib_sample N drawn without replacement, P(i) ~ score (Mini-Splatting importance sampling)
"""
import numpy as np

METHODS = ("opacity", "voxel", "contrib_rank", "contrib_sample", "redundancy", "redundancy_once", "reproj_dedup")
NUMPY_METHODS = METHODS[:4]   # redundancy / redundancy_once / reproj_dedup: common/redundancy.py (torch, need cameras)


def _all(n_total):
    return np.arange(n_total, dtype=np.int64)


def top_n(score, n):
    """Indices of the n largest scores (ties broken by lower index, deterministic)."""
    score = np.asarray(score, dtype=np.float64)
    if n >= len(score):
        return _all(len(score))
    order = np.lexsort((np.arange(len(score)), -score))
    return np.sort(order[:n]).astype(np.int64)


def opacity_topk(opacity_logit, n):
    return top_n(opacity_logit, n)   # sigmoid is monotone: ranking logits == ranking opacities


def voxel_representatives(xyz, score, voxel_size, origin=None):
    """One index per occupied voxel: the gaussian with the highest score in it."""
    xyz = np.asarray(xyz, dtype=np.float64)
    origin = xyz.min(0) if origin is None else np.asarray(origin, dtype=np.float64)
    cell = np.floor((xyz - origin) / voxel_size).astype(np.int64)
    cell -= cell.min(0)
    dims = cell.max(0) + 1
    if np.prod(dims.astype(np.float64)) >= 2.0 ** 62:
        raise ValueError(f"voxel size {voxel_size} too small for this extent")
    key = (cell[:, 0] * dims[1] + cell[:, 1]) * dims[2] + cell[:, 2]
    order = np.lexsort((np.arange(len(key)), -np.asarray(score, dtype=np.float64), key))
    first = np.ones(len(order), dtype=bool)
    first[1:] = key[order][1:] != key[order][:-1]
    return order[first]


def voxel_select(xyz, opacity_logit, n, iters=40, return_info=False):
    """Largest voxel size whose occupied-voxel count is still >= n, then trim to n by opacity.

    Larger voxels => fewer occupied voxels (monotone up to grid-phase effects, which the bisection
    tolerates because it only keeps sizes that satisfy count >= n)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    total = len(xyz)
    if n >= total:
        idx = _all(total)
        info = dict(voxel_size=0.0, occupied_voxels=total, bisection_iters=0, trimmed=0)
        return (idx, info) if return_info else idx
    origin = xyz.min(0)
    extent = float((xyz.max(0) - origin).max())
    lo, hi = 0.0, max(extent, 1e-6) * 2.0           # count(hi) == 1 < n
    best = None
    k = 0
    for k in range(1, iters + 1):
        mid = 0.5 * (lo + hi)
        if mid <= 0:
            break
        rep = voxel_representatives(xyz, opacity_logit, mid, origin)
        if len(rep) >= n:
            lo, best = mid, (mid, rep)
            if len(rep) == n:
                break
        else:
            hi = mid
        if hi - lo < 1e-9 * max(extent, 1.0):
            break
    if best is None:                                  # pathological: duplicates everywhere
        idx = opacity_topk(opacity_logit, n)
        info = dict(voxel_size=None, occupied_voxels=None, bisection_iters=k, trimmed=None,
                    note="no voxel size reached n occupied voxels; fell back to opacity")
        return (idx, info) if return_info else idx
    size, rep = best
    keep = rep[top_n(np.asarray(opacity_logit)[rep], n)]
    idx = np.sort(keep).astype(np.int64)
    info = dict(voxel_size=float(size), occupied_voxels=int(len(rep)), bisection_iters=k,
                trimmed=int(len(rep) - n))
    return (idx, info) if return_info else idx


def volume_weight(scale_log, power):
    """LightGaussian volume term: (V / V_90%)^power, V = prod(exp(scale)), clipped at 1 before power."""
    if power == 0:
        return np.ones(len(scale_log))
    vol = np.exp(np.asarray(scale_log, dtype=np.float64).sum(1))
    ref = np.quantile(vol, 0.9)
    if ref <= 0:
        return np.ones(len(vol))
    return np.power(np.clip(vol / ref, 0.0, 1.0), power)


def contrib_rank(score, n):
    return top_n(score, n)


def contrib_sample(score, n, seed=0):
    """Weighted sampling without replacement, P ~ score (Gumbel-top-k == sequential sampling).
    Gaussians with score 0 are only taken when fewer than n have positive score."""
    score = np.asarray(score, dtype=np.float64)
    total = len(score)
    if n >= total:
        return _all(total)
    rng = np.random.default_rng(seed)
    with np.errstate(divide="ignore"):
        logp = np.log(np.clip(score, 0.0, None))
    key = logp + rng.gumbel(size=total)
    key[~np.isfinite(key)] = -np.inf
    pos = np.isfinite(key)
    if pos.sum() >= n:
        return top_n(np.where(pos, key, -np.inf), n)
    rest = np.flatnonzero(~pos)
    fill = rest[rng.permutation(len(rest))[: n - pos.sum()]]
    return np.sort(np.concatenate([np.flatnonzero(pos), fill])).astype(np.int64)


def select(method, n, xyz=None, opacity_logit=None, score=None, seed=0):
    if method == "opacity":
        return opacity_topk(opacity_logit, n), {}
    if method == "voxel":
        return voxel_select(xyz, opacity_logit, n, return_info=True)
    if method == "contrib_rank":
        return contrib_rank(score, n), {}
    if method == "contrib_sample":
        return contrib_sample(score, n, seed), {"seed": seed}
    raise ValueError(f"method {method} is not a numpy method; expected one of {NUMPY_METHODS}")
