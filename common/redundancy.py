"""Two duplicate-aware pruning methods for the DA3 first-frame gaussians (torch; CPU or CUDA).

1) Spatial redundancy score of Papantonakis et al. 2024, "Reducing the Memory Footprint of 3D Gaussian
   Splatting" (PACMCGIT 7(1)), ported from the official code (github.com/graphdeco-inria/reduced-3dgs:
   scene/__init__.py calculate_redundancy_metric, find_minimum_projected_pixel_size_python;
   submodules/diff-gaussian-rasterization/reduced_3dgs/redundancy_score.cu; gaussian_model.py mercy_points):
     pixel_size_i = min over training cameras (center inside the frustum) of the world size of one pixel
                    at the gaussian's depth                                       (10000 if never visible)
     r_i          = box_size * pixel_size_i * sqrt(3) / 2                  (sphere around gaussian i)
     neighbours   = 30 nearest centers; j "covers" i if i's center lies in j's ellipsoid with every scale
                    axis enlarged by r_i (the official approximate sphere-ellipsoid test)
     red_i        = 1 + #covering neighbours
     final_i      = min of red_k over every sphere k that contains gaussian i (incl. its own)
     candidates   = final > max(mean + lambda * std, minimum)        (official defaults lambda=1, minimum=3)
     official rule: remove the candidates whose opacity is below the candidates' median, repeatedly
   Our variant (target count): repeat {recompute the score on the remaining set; remove the
   lowest-opacity candidates, at most the official median half, never below N}. If a round has no
   candidates the remainder is removed by global opacity (reported as fallback). Not the paper's
   training-time schedule: it is applied once, before the 1k refine, to reach an exact count.

2) Cross-view duplicate removal by reprojection (our variant for pixel-aligned DA3 output; FreeSplat's
   pixel-wise alignment idea, without its learned feature fusion): views are processed in a fixed order;
   for view t the already-kept gaussians of earlier views are z-buffered into t (front-most depth per
   cell, then a 3x3 min); a gaussian generated from view t is a duplicate when the front-most kept
   surface at its pixel has the same depth within a relative tolerance. A kept surface in front of it
   (occluder / thin object) or behind it never counts, so front and back surfaces are not merged.
   One representative (the earlier view's) is kept; attributes are NOT averaged.
   The count is then brought to exactly N by opacity (drop the lowest kept, or add back the
   highest-opacity duplicates), and both numbers are reported.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- shared helpers
def quat_to_rot(q):
    """graphdeco build_rotation: q = (r, x, y, z), normalized here. Returns (N,3,3) R (columns = local axes)."""
    q = F.normalize(q, dim=-1)
    r, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)], -1)
    return R.view(-1, 3, 3)


def project(xyz, full_proj, world_view):
    """Row-vector convention (3DGS cameras): clip = [x,1] @ full_proj, view = [x,1] @ world_view."""
    ph = torch.cat([xyz, torch.ones_like(xyz[:, :1])], 1)
    clip = ph @ full_proj
    ndc = clip[:, :3] / (clip[:, 3:4] + 1e-7)
    view_z = (ph @ world_view)[:, 2]
    return ndc, view_z


# --------------------------------------------------------------------------- 1) redundancy score
def min_projected_pixel_size(xyz, cams):
    """cams: iterable of (full_proj (4,4), width, height). Port of find_minimum_projected_pixel_size_python."""
    size = torch.full((xyz.shape[0],), 10000.0, device=xyz.device, dtype=xyz.dtype)
    ones = torch.ones_like(xyz[:, :1])
    for full, W, H in cams:
        ph = torch.cat([xyz, ones], 1) @ full
        ndc = ph / ph[:, 3:4]
        m = (ndc[:, 0].abs() <= 1) & (ndc[:, 1].abs() <= 1) & (ndc[:, 2] >= 0) & (ndc[:, 2] <= 1)
        if not m.any():
            continue
        inv = torch.linalg.inv(full)
        d = ndc[m, 2]
        a = torch.zeros((int(m.sum()), 4), device=xyz.device, dtype=xyz.dtype)
        b = torch.zeros_like(a)
        a[:, 0 if W > H else 1] = min(2.0 / W, 2.0 / H)
        a[:, 2] = d; a[:, 3] = 1
        b[:, 2] = d; b[:, 3] = 1
        a = a @ inv; a = a / a[:, 3:4]
        b = b @ inv; b = b / b[:, 3:4]
        size[m] = torch.minimum(size[m], (a - b)[:, :3].norm(dim=1))
    return size


def knn_indices(xyz_np, k=30, workers=-1):
    from scipy.spatial import cKDTree
    tree = cKDTree(xyz_np)
    _, idx = tree.query(xyz_np, k=k + 1, workers=workers)
    idx = np.asarray(idx)
    # drop self (normally column 0; with exact duplicates self may sit elsewhere)
    n = len(xyz_np)
    out = np.empty((n, k), dtype=np.int64)
    self_col = idx == np.arange(n)[:, None]
    has_self = self_col.any(1)
    keep = ~self_col
    keep[~has_self, -1] = False               # self not returned: drop the farthest instead
    out[:] = idx[keep].reshape(n, k)
    return out


def redundancy_score(xyz, scale, rot_q, pixel_size, nbr, box_size=1.0, chunk=200_000):
    """Returns final (minimum) redundancy value per gaussian (int64). scale = activated (exp) scales."""
    n, k = nbr.shape
    radius = pixel_size * box_size * math.sqrt(3.0) / 2.0
    R = quat_to_rot(rot_q)
    red = torch.ones(n, dtype=torch.int64, device=xyz.device)
    hit = torch.zeros((n, k), dtype=torch.bool, device=xyz.device)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        j = nbr[s:e]                                            # (c,k)
        d = xyz[s:e, None, :] - xyz[j]                          # (c,k,3)
        local = torch.einsum("ckd,ckde->cke", d, R[j])          # R^T d  (glm: difference * R)
        aug = scale[j] + radius[s:e, None, None]
        inside = ((local / aug) ** 2).sum(-1) < 1
        hit[s:e] = inside
        red[s:e] += inside.sum(1)
    final = torch.full((n,), n, dtype=torch.int64, device=xyz.device)
    final.scatter_reduce_(0, torch.arange(n, device=xyz.device), red, reduce="amin")
    src = red[:, None].expand(n, k)[hit]
    dst = nbr[hit]
    final.scatter_reduce_(0, dst, src, reduce="amin")
    return final


def redundancy_select(xyz, scale, rot_q, opacity_logit, cams, n_target, k=30, box_size=1.0,
                      lam=1.0, minimum=3.0, max_rounds=60, log=print):
    """Iterative official-score pruning to exactly n_target. Returns (sorted kept indices, info)."""
    dev = xyz.device
    alive = torch.arange(xyz.shape[0], device=dev)
    rounds, removed_red, removed_fb = [], 0, 0
    while alive.numel() > n_target and len(rounds) < max_rounds:
        x = xyz[alive]
        ps = min_projected_pixel_size(x, cams)
        nbr = torch.from_numpy(knn_indices(x.detach().cpu().numpy().astype(np.float64), k)).to(dev)
        red = redundancy_score(x, scale[alive], rot_q[alive], ps, nbr, box_size)
        rf = red.double()
        thr = max(float(rf.mean() + lam * rf.std()), minimum)
        cand = (red > thr).nonzero(as_tuple=True)[0]
        excess = alive.numel() - n_target
        info = dict(n=int(alive.numel()), threshold=thr, candidates=int(cand.numel()),
                    never_visible=int((ps >= 10000).sum()))
        if cand.numel() == 0:
            info["removed"] = 0
            rounds.append(info)
            break
        op = opacity_logit[alive][cand]
        below = cand[op < op.median()]                            # official: below the candidates' median
        if below.numel() == 0:
            below = cand[op <= op.median()]
        if below.numel() > excess:                                # do not go below the target
            o = opacity_logit[alive][below]
            below = below[torch.argsort(o)[:excess]]
        keep = torch.ones(alive.numel(), dtype=torch.bool, device=dev)
        keep[below] = False
        alive = alive[keep]
        removed_red += int(below.numel())
        info["removed"] = int(below.numel())
        rounds.append(info)
        log(f"[redundancy] round {len(rounds)}: n={info['n']:,} thr={thr:.2f} cand={info['candidates']:,} "
            f"removed={info['removed']:,}")
        if info["removed"] == 0:
            break
    if alive.numel() > n_target:                                  # fallback: global opacity
        order = torch.argsort(opacity_logit[alive], descending=True)
        removed_fb = int(alive.numel() - n_target)
        alive = alive[order[:n_target]]
    kept = torch.sort(alive).values
    return kept, dict(rounds=rounds, removed_by_redundancy=removed_red, removed_by_opacity_fallback=removed_fb,
                      knn=k, box_size=box_size, lambda_mercy=lam, mercy_minimum=minimum)


def redundancy_removal_order(xyz, scale, rot_q, opacity_logit, cams, k=30, box_size=1.0, lam=1.0,
                             minimum=3.0, max_rounds=200, log=print):
    """Full removal order of the iterative method (same result as redundancy_select for EVERY N):
    round after round, the removed set is appended sorted by ascending opacity (exactly what the
    per-N cap keeps), then — once no candidates are left — all remaining gaussians by ascending opacity.
    Returns (order int64 (N,), n_by_redundancy, rounds)."""
    dev = xyz.device
    alive = torch.arange(xyz.shape[0], device=dev)
    order, rounds = [], []
    while alive.numel() > 1 and len(rounds) < max_rounds:
        x = xyz[alive]
        ps = min_projected_pixel_size(x, cams)
        nbr = torch.from_numpy(knn_indices(x.detach().cpu().numpy().astype(np.float64), min(k, alive.numel() - 1))).to(dev)
        red = redundancy_score(x, scale[alive], rot_q[alive], ps, nbr, box_size)
        rf = red.double()
        thr = max(float(rf.mean() + lam * rf.std()), minimum)
        cand = (red > thr).nonzero(as_tuple=True)[0]
        if cand.numel() == 0:
            break
        op = opacity_logit[alive][cand]
        below = cand[op < op.median()]
        if below.numel() == 0:
            below = cand[op <= op.median()]
        below = below[torch.argsort(opacity_logit[alive][below])]
        order.append(alive[below])
        keep = torch.ones(alive.numel(), dtype=torch.bool, device=dev)
        keep[below] = False
        rounds.append(dict(n=int(alive.numel()), threshold=thr, candidates=int(cand.numel()), removed=int(below.numel())))
        log(f"[redundancy-order] round {len(rounds)}: n={alive.numel():,} thr={thr:.2f} removed={below.numel():,}")
        alive = alive[keep]
    n_red = int(sum(o.numel() for o in order))
    order.append(alive[torch.argsort(opacity_logit[alive])])          # fallback: global opacity
    return torch.cat(order), n_red, rounds


def redundancy_once_order(xyz, scale, rot_q, opacity_logit, cams, k=30, box_size=1.0, lam=1.0, minimum=3.0):
    """One-shot variant: the official score is computed ONCE on the full set; candidates
    (score > max(mean + lam*std, minimum)) are removed first in ascending opacity, then everything else
    in ascending opacity. Cheaper (one kNN), but scores are not re-evaluated after removals."""
    ps = min_projected_pixel_size(xyz, cams)
    nbr = torch.from_numpy(knn_indices(xyz.detach().cpu().numpy().astype(np.float64), k)).to(xyz.device)
    red = redundancy_score(xyz, scale, rot_q, ps, nbr, box_size)
    rf = red.double()
    thr = max(float(rf.mean() + lam * rf.std()), minimum)
    is_cand = red > thr
    cand = is_cand.nonzero(as_tuple=True)[0]
    rest = (~is_cand).nonzero(as_tuple=True)[0]
    order = torch.cat([cand[torch.argsort(opacity_logit[cand])], rest[torch.argsort(opacity_logit[rest])]])
    return order, int(cand.numel()), [dict(n=int(xyz.shape[0]), threshold=thr, candidates=int(cand.numel()))]


def select_from_order(order, n_total, n_target, n_by_redundancy):
    n_remove = max(0, n_total - n_target)
    removed = order[:n_remove]
    keep = torch.ones(n_total, dtype=torch.bool, device=order.device)
    keep[removed] = False
    kept = keep.nonzero(as_tuple=True)[0]
    return kept, dict(removed_by_redundancy=min(n_remove, n_by_redundancy),
                      removed_by_opacity_fallback=max(0, n_remove - n_by_redundancy))


# --------------------------------------------------------------------------- 2) reprojection duplicates
def _zbuffer(cells, depth, n_cells):
    zb = torch.full((n_cells,), float("inf"), device=depth.device, dtype=depth.dtype)
    if cells.numel():
        zb.scatter_reduce_(0, cells, depth, reduce="amin")
    return zb


def reprojection_duplicates(xyz, view_ids, views, rel_tol=0.02, cell_px=3, znear=0.2, log=print):
    """views: list (in processing order) of dict(view_id, full_proj, world_view, W, H).
    Returns (dup mask bool (N,), per-view stats). Gaussians whose source view is not in `views` are kept."""
    dev = xyz.device
    n = xyz.shape[0]
    dup = torch.zeros(n, dtype=torch.bool, device=dev)
    done = torch.zeros(n, dtype=torch.bool, device=dev)          # gaussians of already processed views
    stats = []
    for v in views:
        W, H = v["W"], v["H"]
        gw, gh = math.ceil(W / cell_px), math.ceil(H / cell_px)
        ndc, z = project(xyz, v["full_proj"], v["world_view"])
        px = ((ndc[:, 0] + 1.0) * W - 1.0) * 0.5
        py = ((ndc[:, 1] + 1.0) * H - 1.0) * 0.5
        cx = torch.floor(px / cell_px).long()
        cy = torch.floor(py / cell_px).long()
        valid = (z > znear) & (cx >= 0) & (cx < gw) & (cy >= 0) & (cy < gh)
        cell = cy.clamp(0, gh - 1) * gw + cx.clamp(0, gw - 1)
        glob = done & ~dup & valid
        zb = _zbuffer(cell[glob], z[glob], gw * gh).view(1, 1, gh, gw)
        zb = -F.max_pool2d(-zb, 3, stride=1, padding=1)            # front-most kept surface in 3x3
        zb = zb.view(-1)
        local = (view_ids == v["view_id"]) & valid
        li = local.nonzero(as_tuple=True)[0]
        front = zb[cell[li]]
        match = torch.isfinite(front) & ((z[li] - front).abs() <= rel_tol * front)
        dup[li[match]] = True
        done |= view_ids == v["view_id"]
        s = dict(view_id=int(v["view_id"]), name=v.get("name"), local=int(li.numel()),
                 duplicates=int(match.sum()), global_before=int(glob.sum()))
        stats.append(s)
        log(f"[reproj] view {s['name']}: local {s['local']:,} dup {s['duplicates']:,}")
    return dup, stats


def reprojection_select(xyz, view_ids, opacity_logit, views, n_target, rel_tol=0.02, cell_px=3, log=print):
    dup, stats = reprojection_duplicates(xyz, view_ids, views, rel_tol, cell_px, log=log)
    keep = ~dup
    n_unique = int(keep.sum())
    idx_keep = keep.nonzero(as_tuple=True)[0]
    if n_unique >= n_target:
        order = torch.argsort(opacity_logit[idx_keep], descending=True)
        kept = idx_keep[order[:n_target]]
        added_back = 0
    else:
        idx_dup = dup.nonzero(as_tuple=True)[0]
        order = torch.argsort(opacity_logit[idx_dup], descending=True)
        added_back = n_target - n_unique
        kept = torch.cat([idx_keep, idx_dup[order[:added_back]]])
    kept = torch.sort(kept).values
    return kept, dict(n_duplicates=int(dup.sum()), n_unique=n_unique, added_back_by_opacity=added_back,
                      removed_unique_by_opacity=max(0, n_unique - n_target), rel_tol=rel_tol,
                      cell_px=cell_px, view_order=[s["name"] for s in stats], per_view=stats)
