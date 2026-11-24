"""[env: gs for contrib_*, any for opacity/voxel] First-frame pruning to exactly N gaussians.

  python scripts/23_select_gaussians.py --ply converted.ply --out sel.ply --method voxel --n 600000
  python scripts/23_select_gaussians.py --ply converted.ply --out sel.ply --method contrib_rank --n 600000 \
      --frame_dir $FRAME0 --scores_cache scores.npz

Drop-in replacement for 06_subsample.py (same PLY / _timing.json / _sources.npz outputs) plus
<out>_select.json with method details and <out>_indices.npy.

redundancy    Papantonakis et al. 2024 spatial redundancy score (official code ported), iterated to N
reproj_dedup  cross-view duplicate removal by reprojection + relative depth (z-buffer), then opacity to N
              (both: see common/redundancy.py; need --frame_dir; reproj_dedup needs <ply>_sources.npz)

Render-contribution score (methods contrib_rank / contrib_sample):
  score_i = sum over the 20 train views and all pixels of alpha_i * T_i     (blending weight)
  computed with the installed 3DGStream rasterizer: render with colors_precomp = 1 and take
  d(sum of image)/d(color_i) = sum_pixels alpha_i T_i  (exact; no custom CUDA kernel needed).
  Optionally multiplied by LightGaussian's volume term (--volume_power, their default 0.1).
The score only depends on the source PLY and the views, so it is cached (--scores_cache) and the
selection time reported for contrib_* = scoring time + selection time (both stored separately).
"""
import argparse
import hashlib
import json
import math
import os
import time

import numpy as np

from common import timing
from common.diagnostics import sources_path
from common.ply_io import load_ply, save_ply, stats, subset
from common.selection import METHODS, NUMPY_METHODS, select, volume_weight


def file_digest(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def contribution_scores(d, frame_dir, images="images_2", test_cam="cam00"):
    """Sum of blending weights per gaussian over all train views (gradient trick)."""
    import torch
    import torch.nn.functional as F
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    from common.colmap_io import load_cameras
    from common.gs_render import RCam

    dev = "cuda"
    xyz = torch.tensor(d["xyz"], device=dev)
    opac = torch.sigmoid(torch.tensor(d["opacity"], device=dev)).reshape(-1, 1)
    scales = torch.exp(torch.tensor(d["scale"], device=dev))
    rots = F.normalize(torch.tensor(d["rot"], device=dev), dim=-1)
    fields = GaussianRasterizationSettings._fields
    cams = [c for c in load_cameras(frame_dir, images) if c.name != test_cam]
    score = torch.zeros(len(xyz), device=dev, dtype=torch.float64)
    torch.cuda.synchronize(); t0 = time.time()
    for cam in cams:
        rc = RCam(cam)
        kw = dict(image_height=rc.H, image_width=rc.W, tanfovx=math.tan(rc.fovx / 2),
                  tanfovy=math.tan(rc.fovy / 2), bg=torch.zeros(3, device=dev), scale_modifier=1.0,
                  viewmatrix=rc.wv, projmatrix=rc.full, sh_degree=0, campos=rc.campos,
                  prefiltered=False, debug=False, bwd_depth=False, antialiasing=False)
        settings = GaussianRasterizationSettings(**{k: v for k, v in kw.items() if k in fields})
        colors = torch.ones((len(xyz), 3), device=dev, requires_grad=True)
        means2d = torch.zeros_like(xyz, requires_grad=True)
        out = GaussianRasterizer(raster_settings=settings)(
            means3D=xyz, means2D=means2d, shs=None, colors_precomp=colors, opacities=opac,
            scales=scales, rotations=rots, cov3D_precomp=None)
        img = out[0] if isinstance(out, (tuple, list)) else out
        img[0].sum().backward()          # one channel: grad = sum_pixels alpha*T for that channel
        score += colors.grad[:, 0].double()
        del colors, means2d, img, out
    torch.cuda.synchronize()
    sec = time.time() - t0
    return score.cpu().numpy(), sec, [c.name for c in cams]


def load_or_compute_scores(args, d):
    digest = file_digest(args.ply)
    if args.scores_cache and os.path.isfile(args.scores_cache):
        with np.load(args.scores_cache, allow_pickle=False) as z:
            if str(z["src_sha256"]) == digest and str(z["frame_dir"]) == os.path.abspath(args.frame_dir):
                print(f"scores from cache {args.scores_cache}")
                return z["score"], float(z["score_sec"]), list(z["views"]), True
        print("scores cache does not match this PLY/frame -> recomputing")
    score, sec, views = contribution_scores(d, args.frame_dir, args.images, args.test_cam)
    if args.scores_cache:
        os.makedirs(os.path.dirname(os.path.abspath(args.scores_cache)), exist_ok=True)
        np.savez_compressed(args.scores_cache, score=score, score_sec=sec, views=np.array(views),
                            src_sha256=digest, frame_dir=os.path.abspath(args.frame_dir))
    return score, sec, views, False


def select_with_cameras(args, d):
    import torch
    from common.colmap_io import load_cameras
    from common.gs_render import RCam
    from common import redundancy as rd
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cams = [c for c in load_cameras(args.frame_dir, args.images) if c.name != args.test_cam]
    rcams = {c.name: RCam(c, dev) for c in cams}
    t = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)
    xyz, op = t(d["xyz"]), t(d["opacity"])
    n = min(args.n, len(d["xyz"]))
    if args.method in ("redundancy", "redundancy_once"):
        # The removal order does not depend on N, so it is computed once per source PLY and cached;
        # the time reported for every N is the original order computation (what a single N costs).
        cam_list = [(rc.full, rc.W, rc.H) for rc in rcams.values()]
        params = dict(knn=args.knn, box_size=args.box_size, lambda_mercy=args.lambda_mercy,
                      mercy_minimum=args.mercy_minimum, test_cam=args.test_cam)
        cache = (os.path.join(os.path.dirname(os.path.abspath(args.scores_cache)), f"{args.method}_order.npz")
                 if args.scores_cache else None)
        digest = file_digest(args.ply)
        order = None
        if cache and os.path.isfile(cache):
            with np.load(cache, allow_pickle=False) as z:
                if str(z["src_sha256"]) == digest and str(z["params"]) == json.dumps(params, sort_keys=True):
                    order, n_red, order_sec = z["order"], int(z["n_by_redundancy"]), float(z["order_sec"])
                    rounds = json.loads(str(z["rounds"]))
                    print(f"{args.method}: removal order from cache {cache}")
        cached = order is not None
        if order is None:
            fn = rd.redundancy_removal_order if args.method == "redundancy" else rd.redundancy_once_order
            t0 = time.time()
            o, n_red, rounds = fn(xyz, torch.exp(t(d["scale"])), t(d["rot"]), op, cam_list, k=args.knn,
                                 box_size=args.box_size, lam=args.lambda_mercy, minimum=args.mercy_minimum)
            order_sec = time.time() - t0
            order = o.cpu().numpy().astype(np.int64)
            if cache:
                np.savez_compressed(cache, order=order, n_by_redundancy=n_red, order_sec=order_sec,
                                    rounds=json.dumps(rounds), src_sha256=digest,
                                    params=json.dumps(params, sort_keys=True))
        kept, info = rd.select_from_order(torch.from_numpy(order), len(d["xyz"]), n, n_red)
        info.update(params, n_redundant_total=n_red, rounds=rounds, order_sec=order_sec, order_cached=cached)
        args._extra_sec = order_sec if cached else 0.0
    elif args.method == "reproj_dedup":
        sp = sources_path(args.ply)
        if not sp.is_file():
            raise FileNotFoundError(f"reproj_dedup needs per-gaussian source views: {sp}")
        with np.load(sp, allow_pickle=False) as z:
            view_ids, names = z["view_ids"].astype(np.int64), [str(x) for x in z["camera_names"]]
        missing = [nm for nm in names if nm not in rcams]
        if missing:
            raise ValueError(f"source views without a training camera: {missing}")
        views = [dict(view_id=i, name=nm, full_proj=rcams[nm].full, world_view=rcams[nm].wv,
                      W=rcams[nm].W, H=rcams[nm].H) for i, nm in enumerate(names)]
        kept, info = rd.reprojection_select(xyz, torch.from_numpy(view_ids).to(dev), op, views, n,
                                            rel_tol=args.rel_tol, cell_px=args.cell_px)
    else:
        raise ValueError(args.method)
    return kept.cpu().numpy().astype(np.int64), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--frame_dir", help="frame000000 (needed for contrib_*)")
    ap.add_argument("--images", default="images_2")
    ap.add_argument("--test_cam", default="cam00")
    ap.add_argument("--volume_power", type=float, default=0.1, help="LightGaussian v_pow; 0 = pure blending weight")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scores_cache")
    ap.add_argument("--knn", type=int, default=30, help="redundancy: neighbours (official 30)")
    ap.add_argument("--box_size", type=float, default=1.0, help="redundancy: pixel_scale (official 1)")
    ap.add_argument("--lambda_mercy", type=float, default=1.0, help="redundancy: threshold mean+lambda*std (official 1)")
    ap.add_argument("--mercy_minimum", type=float, default=3.0, help="redundancy: minimum threshold (official 3)")
    ap.add_argument("--rel_tol", type=float, default=0.02, help="reproj_dedup: relative depth tolerance")
    ap.add_argument("--cell_px", type=int, default=3, help="reproj_dedup: z-buffer cell size in images_2 pixels")
    args = ap.parse_args()
    if args.n < 1:
        ap.error("--n must be positive")
    if args.method not in ("opacity", "voxel") and not args.frame_dir:
        ap.error("--frame_dir is required for this method")
    if os.path.exists(args.out):
        raise FileExistsError(f"Output already exists: {args.out}")

    d = load_ply(args.ply)
    n_src = len(d["xyz"])
    score_info, score = {}, None
    score_sec = 0.0
    if args.method.startswith("contrib"):
        raw, score_sec, views, cached = load_or_compute_scores(args, d)
        vw = volume_weight(d["scale"], args.volume_power)
        score = raw * vw
        score_info = dict(score_sec=score_sec, score_cached=cached, views=views,
                          volume_power=args.volume_power, n_zero_score=int((raw <= 0).sum()),
                          score_quantiles={q: float(np.quantile(score, q)) for q in (0.1, 0.5, 0.9, 0.99)})

    t0 = time.time()
    if args.method in NUMPY_METHODS:
        idx, info = select(args.method, args.n, xyz=d["xyz"], opacity_logit=d["opacity"], score=score, seed=args.seed)
    else:
        idx, info = select_with_cameras(args, d)
    select_sec = time.time() - t0 + getattr(args, "_extra_sec", 0.0)
    assert len(idx) == min(args.n, n_src) and len(np.unique(idx)) == len(idx)

    out = subset(d, idx)
    save_ply(out, args.out)
    stem = os.path.splitext(args.out)[0]
    np.save(stem + "_indices.npy", idx)
    total_sec = select_sec + score_sec
    timing.write(args.out, "select_" + args.method, total_sec, src_ply=args.ply,
                 init_metadata=dict(timing.load(args.ply).get("init_metadata", {}),
                                    max_gaussians=int(args.n), n_final_gaussians=int(len(idx)),
                                    selection_method=args.method),
                 notes=[f"{args.method} N={args.n} from {os.path.basename(args.ply)}"
                        + (" (scores from cache; score_sec is the original scoring time)"
                           if score_info.get("score_cached") else "")])
    if sources_path(args.ply).is_file():
        with np.load(sources_path(args.ply), allow_pickle=False) as source:
            np.savez_compressed(sources_path(args.out), view_ids=source["view_ids"][idx],
                                camera_names=source["camera_names"])
    st = stats(out)
    rec = dict(src=args.ply, method=args.method, n=int(args.n), n_src=int(n_src), n_out=int(len(idx)),
               select_sec=select_sec, score_sec=score_sec, total_sec=total_sec, seed=args.seed,
               method_info=info, score_info=score_info, stats=st)
    with open(stem + "_select.json", "w") as f:
        json.dump(rec, f, indent=1)
    print(f"{args.method}: {n_src:,} -> {len(idx):,} in {total_sec:.1f}s  saved {args.out}")


if __name__ == "__main__":
    main()
