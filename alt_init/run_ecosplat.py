"""[env: eco] EcoSplat (CVPR 2026, ZPressor/MVSplat base, IGF checkpoint) as a first-frame initializer.

Same protocol as ZipSplat in the third experiment: the 20 train views of frame 0 (cam00 held out), the
known COLMAP cameras (MVSplat is pose-conditioned, so no Sim3 is needed: Gaussians come out in the COLMAP
frame), then the usual +1k refine / static evaluation is done by the caller.
The model was trained on RealEstate10K at 256x256; we feed H x W = 256 x 352 (aspect of 1352x1014,
both divisible by the 16-px patch shim) with correctly scaled intrinsics. The count is set by the
protect rate rho (inference_rho); an optional --max_gaussians keeps the most opaque (same rule as the DA3 cap).

  python alt_init/run_ecosplat.py --repo /workspace/eco/repo_zp/mvsplat --ckpt .../ecosplat-zpressor-re10k.ckpt \
      --frame_dir /workspace/data/flame_steak/frame000000 --rho 0.4 --out D/init.ply
exit codes: 0 ok, 5 bad input, 6 OOM, 7 non-finite, 8 weights did not load
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common import timing                                    # noqa: E402
from common.colmap_io import load_cameras                    # noqa: E402
from common.ply_io import save_ply, set_sh_degree, stats, subset   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="EcoSplat eco_zpressor branch, mvsplat/ directory")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frame_dir", required=True)
    ap.add_argument("--images", default="images_2")
    ap.add_argument("--test_cam", default="cam00")
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--H", type=int, default=256)
    ap.add_argument("--W", type=int, default=352)
    ap.add_argument("--near", type=float, default=1.0)
    ap.add_argument("--far", type=float, default=100.0)
    ap.add_argument("--max_gaussians", type=int, default=1_000_000)
    ap.add_argument("--sh_degree", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base = os.path.splitext(a.out)[0]

    def fail(code, reason, **extra):
        json.dump(dict(reason=reason, rho=a.rho, **extra), open(base + "_ecosplat_failed.json", "w"), indent=1)
        print("ERROR:", reason)
        sys.exit(code)

    import torch
    import torch.nn.functional as F
    from PIL import Image

    torch.manual_seed(a.seed)
    cams = load_cameras(a.frame_dir, images=a.images, exclude=(a.test_cam,))
    if len(cams) < 2:
        fail(5, f"{len(cams)} views")

    # ---- model (hydra config of the repo, weights from the IGF checkpoint)
    repo = os.path.abspath(a.repo)
    sys.path.insert(0, repo)
    os.chdir(repo)
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=os.path.join(repo, "config"), version_base=None):
        cfg_dict = compose(config_name="main", overrides=[
            "+experiment=re10k_igf", "mode=test", "wandb.mode=disabled",
            "model.encoder.stage1_weights_path=null", f"+model.encoder.igf.inference_rho={a.rho}",
            f"dataset.view_sampler.num_context_views={len(cams)}", f"dataset.image_shape=[{a.H},{a.W}]"])
    from src.config import load_typed_root_config
    from src.global_cfg import set_cfg
    from src.model.encoder import get_encoder
    set_cfg(cfg_dict)
    cfg = load_typed_root_config(cfg_dict)
    encoder, _ = get_encoder(cfg.model.encoder)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    res = encoder.load_state_dict(sd, strict=False)
    print(f"weights: {len(sd)} tensors, missing {len(res.missing_keys)}, unexpected {len(res.unexpected_keys)}")
    if len(res.missing_keys) > 0.05 * max(len(sd), 1):
        fail(8, "checkpoint does not match the model", missing=res.missing_keys[:20])
    encoder = encoder.cuda().eval()

    # ---- inputs: OpenCV c2w, normalized intrinsics, [0,1] images
    imgs, Ks, c2ws = [], [], []
    for c in cams:
        im = np.asarray(Image.open(c.image_path).convert("RGB"), dtype=np.float32) / 255.0
        t = torch.from_numpy(im).permute(2, 0, 1)[None]
        imgs.append(F.interpolate(t, size=(a.H, a.W), mode="bilinear", align_corners=False, antialias=True)[0])
        K = np.array(c.K, dtype=np.float64).copy()
        K[0] /= c.width
        K[1] /= c.height
        Ks.append(K)
        c2ws.append(np.linalg.inv(np.array(c.w2c, dtype=np.float64)))
    V = len(cams)
    dev = "cuda"
    view = dict(image=torch.stack(imgs)[None].to(dev),
                intrinsics=torch.tensor(np.stack(Ks), dtype=torch.float32)[None].to(dev),
                extrinsics=torch.tensor(np.stack(c2ws), dtype=torch.float32)[None].to(dev),
                near=torch.full((1, V), a.near, device=dev), far=torch.full((1, V), a.far, device=dev),
                index=torch.arange(V, device=dev)[None])
    batch = dict(context=view, target={k: v[:, :1] for k, v in view.items()}, scene=["flame_steak"])
    batch = encoder.get_data_shim()(batch)

    try:
        with torch.no_grad():
            encoder(batch["context"], 0, deterministic=True)          # warm-up (excluded from timing)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            exclusive = timing.gpu_exclusive()
            t0 = time.time()
            g = encoder(batch["context"], 0, deterministic=True)
            torch.cuda.synchronize()
            t_infer = time.time() - t0
    except torch.cuda.OutOfMemoryError as e:
        fail(6, "CUDA out of memory", detail=str(e)[:400])
    peak_mb = torch.cuda.max_memory_allocated() / 2**20

    # ---- to a 3DGS PLY (COLMAP frame)
    t1 = time.time()
    means = g.means[0].float()
    if getattr(g, "scales", None) is not None and getattr(g, "rotations", None) is not None:
        scales = g.scales[0].float()
        q = g.rotations[0].float()                           # xyzw
        rot = torch.stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]], 1)
        src = "scales/rotations"
    else:                                                    # from covariance
        evals, evecs = torch.linalg.eigh(g.covariances[0].float())
        scales = evals.clamp_min(1e-12).sqrt()
        R = evecs * torch.sign(torch.det(evecs))[:, None, None]
        from scipy.spatial.transform import Rotation
        qx = Rotation.from_matrix(R.cpu().numpy()).as_quat()  # xyzw
        rot = torch.tensor(qx[:, [3, 0, 1, 2]], device=means.device, dtype=torch.float32)
        src = "covariance eigendecomposition"
    harm = g.harmonics[0].float()                            # (G, 3, d_sh)
    opa = g.opacities[0].float().clamp(1e-6, 1 - 1e-6)
    n_sh = harm.shape[-1]
    k = min(n_sh, 4) - 1
    d = dict(xyz=means.cpu().numpy(), f_dc=harm[:, :, 0].cpu().numpy(),
             f_rest=harm[:, :, 1:1 + k].reshape(len(means), -1).cpu().numpy(),
             opacity=torch.log(opa / (1 - opa)).cpu().numpy(),
             scale=torch.log(scales.clamp_min(1e-7)).cpu().numpy(),
             rot=torch.nn.functional.normalize(rot, dim=1).cpu().numpy())
    n_raw = len(d["xyz"])
    finite = np.all(np.isfinite(np.concatenate([d["xyz"], d["f_dc"], d["f_rest"], d["opacity"][:, None],
                                                d["scale"], d["rot"]], 1)), axis=1)
    n_nonfinite = int((~finite).sum())
    if n_nonfinite > 0.01 * n_raw:
        fail(7, f"{n_nonfinite} of {n_raw} non-finite")
    if n_nonfinite:
        d = subset(d, np.flatnonzero(finite))
    n_capped = 0
    if len(d["xyz"]) > a.max_gaussians:
        keep = np.sort(np.argsort(-d["opacity"], kind="stable")[: a.max_gaussians])
        n_capped = len(d["xyz"]) - len(keep)
        d = subset(d, keep)
    d = set_sh_degree(d, a.sh_degree)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    save_ply(d, a.out)
    t_conv = time.time() - t1

    # depth sanity: fraction of Gaussians in front of each camera within [near, far]
    xyz = d["xyz"]
    infront = []
    for c in cams:
        z = (np.c_[xyz, np.ones(len(xyz))] @ np.array(c.w2c).T)[:, 2]
        infront.append(float(((z > a.near) & (z < a.far)).mean()))
    report = dict(method="ecosplat_zpressor", ckpt=os.path.basename(a.ckpt), rho=a.rho, n_views=V,
                  input_hw=[a.H, a.W], n_raw=int(n_raw), n_gaussians=int(len(d["xyz"])), n_capped_by_opacity=n_capped,
                  n_nonfinite_dropped=n_nonfinite, sh_degree_model=int(round(n_sh ** 0.5)) - 1, shape_source=src,
                  infer_sec=t_infer, convert_sec=t_conv, peak_gpu_mem_mb=peak_mb,
                  frac_in_depth_range_median=float(np.median(infront)), stats=stats(d))
    json.dump(report, open(base + "_ecosplat.json", "w"), indent=1)
    timing.write(a.out, "ecosplat_infer", t_infer, exclusive=exclusive, measurement_required=True,
                 notes=["EcoSplat (ZPressor/MVSplat + IGF) forward, weights loading and warm-up excluded",
                        "camera poses: COLMAP (model is pose-conditioned)"],
                 init_metadata=dict(model="ecosplat_zpressor", n_views=V, rho=a.rho, input_hw=[a.H, a.W], seed=a.seed))
    timing.write(a.out, "convert", t_conv, src_ply=a.out)
    print(json.dumps({k: v for k, v in report.items() if k != "stats"}, indent=1))


if __name__ == "__main__":
    main()
