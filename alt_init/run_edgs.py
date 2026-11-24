"""[env: edgs] EDGS dense correspondence initialization on frame 0 (train views only), no training.

The 1k refine afterwards is our own 05_refine.py (same as DA3+1k), so only the initializer differs.

Steps (timed part marked *):
  build the 3DGS Scene from the train-only COLMAP copy (30_make_train_scene.py)   (not timed: image loading)
  load RoMa weights                                                              (not timed, like DA3 weights)
  * EDGS init_with_corr: RoMa matching, triangulation, Gaussian creation         (official code path)
  * keep exactly N: Gaussians EDGS itself marks invisible (reprojection error above its tolerance,
    opacity logit -10) are dropped, N are drawn from the rest (seed 0). Fewer than N visible -> this N is excluded,
    and all visible points are kept as <out>_ALL.ply instead (condition edgs_all)
  write PLY with sh degree 1 (3DGStream)

Number of Gaussians: EDGS makes ~ num_refs * matches_per_ref points. We use all 20 training views as references
and set matches_per_ref = ceil(oversample * N / num_refs), then cut to exactly N.

  python alt_init/run_edgs.py --edgs_root $ALT/EDGS --scene $ALT_ROOT/scene_train --n 1000000 --out .../init.ply
"""
import argparse
import json
import math
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edgs_root", required=True)
    ap.add_argument("--scene", required=True, help="train-only scene from 30_make_train_scene.py")
    ap.add_argument("--images", default="images_2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--nns_per_ref", type=int, default=3)      # README example value
    ap.add_argument("--num_refs", type=int, default=0, help="0 = all training views")
    ap.add_argument("--oversample", type=float, default=1.5)
    ap.add_argument("--roma_model", default="outdoors", choices=["outdoors", "indoors"])
    ap.add_argument("--sh_degree", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    exp = os.environ.get("EXP") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    edgs_root = os.path.abspath(args.edgs_root)
    out = os.path.abspath(args.out)
    scene = os.path.abspath(args.scene)
    model_dir = os.path.splitext(out)[0] + "_edgs_model"
    os.makedirs(model_dir, exist_ok=True)
    # EDGS imports use paths relative to its repo root (sys.path.append('./submodules/...'))
    os.chdir(edgs_root)
    for p in (edgs_root, os.path.join(edgs_root, "submodules", "gaussian-splatting"),
              os.path.join(edgs_root, "submodules", "RoMa"), exp):
        if p not in sys.path:
            sys.path.insert(0, p)

    import numpy as np
    import torch
    import hydra
    from hydra import compose, initialize_config_dir

    from common import timing
    from common.ply_io import load_ply, save_ply, set_sh_degree, stats, subset

    with open(os.path.join(scene, "subset.json")) as f:
        n_views = len(json.load(f)["kept"])
    num_refs = n_views if args.num_refs <= 0 else min(args.num_refs, n_views)
    matches = int(math.ceil(args.oversample * args.n / num_refs))
    overrides = [
        f"gs.dataset.source_path={scene}", f"gs.dataset.model_path={model_dir}",
        f"gs.dataset.images={args.images}", f"gs.sh_degree={args.sh_degree}",
        "+gs.do_train_test_split=False",        # use all 20 given views (no every-8th split)
        "train.gs_epochs=0", "wandb.mode=disabled", f"seed={args.seed}",
        f"init_wC.num_refs={num_refs}", f"init_wC.nns_per_ref={args.nns_per_ref}",
        f"init_wC.matches_per_ref={matches}", f"init_wC.roma_model={args.roma_model}",
    ]
    with initialize_config_dir(config_dir=os.path.join(edgs_root, "configs"), version_base="1.2"):
        cfg = compose(config_name="train", overrides=overrides)

    from source.trainer import EDGSTrainer
    from source.utils_aux import set_seed
    from romatch import roma_indoor, roma_outdoor

    set_seed(cfg.seed)
    gs = hydra.utils.instantiate(cfg.gs)
    trainer = EDGSTrainer(GS=gs, training_config=cfg.gs.opt, device=cfg.device, log_wandb=False)
    n_train = len(gs.scene.getTrainCameras())
    if n_train != n_views:
        sys.exit(f"EDGS sees {n_train} training views, expected {n_views} (train/test split not disabled?)")
    device = torch.device(cfg.device)
    roma = (roma_indoor if args.roma_model == "indoors" else roma_outdoor)(device=device)
    roma.upsample_preds = False                 # exactly what init_gaussians_with_corr does itself
    roma.symmetric = False
    exclusive = timing.gpu_exclusive()

    torch.cuda.synchronize()
    t0 = time.time()
    trainer.init_with_corr(cfg.init_wC, roma_model=roma)
    torch.cuda.synchronize()
    t_init = time.time() - t0

    raw_ply = os.path.join(model_dir, "edgs_init_raw.ply")
    gs.gaussians.save_ply(raw_ply)
    t1 = time.time()
    d = load_ply(raw_ply)
    total = len(d["xyz"])
    visible = d["opacity"] > -5.0               # EDGS sets rejected points to logit -10
    good = np.flatnonzero(visible)
    rng = np.random.default_rng(args.seed)
    # never pad with points EDGS itself rejected (opacity logit -10).
    # Fewer visible points than N: this N is EXCLUDED and the "all visible points" version is written
    # next to it as <out>_ALL.ply (the screen turns it into the condition edgs_all).
    short = len(good) < args.n
    idx = good if short else np.sort(rng.choice(good, size=args.n, replace=False))
    if short:
        out = os.path.splitext(out)[0] + "_ALL.ply"
    d = set_sh_degree(subset(d, idx), args.sh_degree)
    save_ply(d, out)
    t_cut = time.time() - t1

    report = dict(method="edgs", n_target=args.n, short_of_target=bool(short), n_raw=int(total), n_visible=int(len(good)), n_final=int(len(idx)),
                  num_refs=num_refs, nns_per_ref=args.nns_per_ref, matches_per_ref=matches, n_views=n_views,
                  roma_model=args.roma_model, init_sec=t_init, cut_sec=t_cut, overrides=overrides, stats=stats(d))
    with open(os.path.splitext(out)[0] + "_edgs.json", "w") as f:
        json.dump(report, f, indent=1)
    cams = [c.image_name for c in gs.scene.getTrainCameras()]
    timing.write(out, "edgs_init", t_init, exclusive=exclusive, measurement_required=True,
                 notes=["EDGS init_with_corr (RoMa matching + triangulation), RoMa weight loading excluded",
                        "COLMAP poses shared by all inits (excluded)"],
                 init_metadata=dict(model="edgs", n_views=n_views, camera_names=sorted(cams), seed=args.seed,
                                    matches_per_ref=matches, num_refs=num_refs, nns_per_ref=args.nns_per_ref))
    timing.write(out, "keep_all_visible" if short else "cut_to_n", t_cut, src_ply=out)
    print(json.dumps({k: v for k, v in report.items() if k not in ("stats", "overrides")}, indent=1))
    if short:
        print(f"SHORT: only {len(idx)} visible Gaussians (< N={args.n}) -> N excluded, all points kept in {out}")
        sys.exit(4)


if __name__ == "__main__":
    main()
