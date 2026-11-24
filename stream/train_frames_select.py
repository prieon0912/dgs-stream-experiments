"""[env: gs] 3DGStream train_frames.py with selective inside-AABB updates (modes A/B/C/E/EB).

The upstream repo is NOT modified: this file imports it (cwd = $REPO_GS) and patches four hooks at
runtime. Mode A with these hooks is numerically the upstream algorithm (only logging is added).

  hooks
    Scene.__init__                         remember the current scene (train cameras / GT images)
    GaussianModel.training_one_frame_setup after the NTC is loaded: choose this frame's selection
    NeuralTransformationCache.forward      mask = inside_AABB & selection   (instead of inside_AABB)
    GaussianModel.update_by_ntc            before the stage-1 result is baked in: record motion (B, logs)

  usage (same CLI as train_frames.py plus the options below; 08b_run_stream_select.sh calls it)
    cd $REPO_GS && python $EXP/stream/train_frames_select.py --update_mode B --b_tau_xyz 0.01 \
        --selection_log R/selection.jsonl --selection_state R/selection_state.pt \
        --read_config --config_path R/cfg_run.json -o R/out -m R/init -v $SCENE --image images_2 ...

Per streamed frame one JSON line is appended to --selection_log:
  frame, k, mode, n_base, n_inside, n_selected, full_update, fallback, select_sec, bookkeeping_sec,
  io_sec, motion quantiles of the gaussians actually updated, mask stats (C)
"""
import argparse
import glob
import json
import os
import re
import sys
import time

REPO_GS = os.environ.get("REPO_GS", os.getcwd())
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (REPO_GS, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch                                     # noqa: E402
import torch.nn.functional as F                  # noqa: E402

import selective_update as su                    # noqa: E402


def own_args(argv):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--update_mode", choices=su.MODES, default="A")
    ap.add_argument("--selection_log", required=True)
    ap.add_argument("--selection_state", default=None, help="mode B resume state (torch file)")
    ap.add_argument("--min_selected", type=int, default=128, help="fewer -> fall back to a full update")
    ap.add_argument("--no_motion_log", action="store_true")
    ap.add_argument("--dump_motion", default=None,
                    help="pilot only: save per-gaussian |d_xyz| and rotation angle of every frame (fp16 .npy)")
    # B
    ap.add_argument("--b_window", type=int, default=10)
    ap.add_argument("--b_warmup", type=int, default=10)
    ap.add_argument("--b_recheck", type=int, default=30)
    ap.add_argument("--b_tau_xyz", type=float, default=None)
    ap.add_argument("--b_tau_rot_deg", type=float, default=float("inf"),
                    help="rotation threshold in degrees (inf = position only)")
    # C (mask parameters are shared by E/EB)
    ap.add_argument("--c_kappa", type=int, default=2)
    ap.add_argument("--c_tau", type=float, default=25.0)
    ap.add_argument("--c_radius", type=int, default=48)
    # E / EB adaptive box
    ap.add_argument("--e_every", type=int, default=10, help="recompute the box every N streamed frames")
    ap.add_argument("--e_q", type=float, default=0.01, help="box = [q, 1-q] percentile of candidate centers")
    ap.add_argument("--e_margin", type=float, default=0.1, help="widen the box by margin*extent per side")
    ap.add_argument("--e_min_points", type=int, default=1000, help="fewer candidates -> keep the previous box")
    # diagnostics (not part of the formal timing)
    ap.add_argument("--profile_frames", default="", help="comma list of streamed frames to run under torch.profiler")
    ap.add_argument("--profile_out", default=None)
    a, rest = ap.parse_known_args(argv)
    a.profile_frames = {int(x) for x in a.profile_frames.split(",") if x.strip()}
    if a.update_mode in ("B", "EB") and a.b_tau_xyz is None:
        ap.error("--b_tau_xyz is required for mode B (set it from the pilot, see 26_suggest_thresholds.py)")
    return a, rest


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _load_image(path, size_hw, device):
    from PIL import Image
    import numpy as np
    im = np.asarray(Image.open(path).convert("RGB"))
    t = torch.from_numpy(im).permute(2, 0, 1).contiguous().to(device)      # uint8 (3,H,W)
    if tuple(t.shape[1:]) != tuple(size_hw):
        t = F.interpolate(t[None].float(), size=size_hw, mode="bilinear", align_corners=False)[0]
        t = t.round().clamp(0, 255).to(torch.uint8)
    return t


class Selector:
    def __init__(self, cfg, video_path, images_dir):
        self.cfg = cfg
        self.video_path, self.images_dir = video_path, images_dir
        self.frame, self.k = None, None
        self.scene = None
        self.rec = {}
        self.hist = None
        self.prev_images = {}            # C/E/EB: image_name -> uint8 (3,H,W) of the previous frame
        self.accum = su.ChangeAccumulator(cfg.c_tau, cfg.c_radius) if cfg.update_mode in ("E", "EB") else None
        self.box = None                  # (lo, hi) world-space adaptive box
        self.raster = dict(calls=0, pairs=0)
        self.hook_sec = 0.0
        if cfg.update_mode in ("B", "EB") and cfg.selection_state and os.path.isfile(cfg.selection_state):
            self._pending_state = torch.load(cfg.selection_state, map_location="cpu")
        else:
            self._pending_state = None
        os.makedirs(os.path.dirname(os.path.abspath(cfg.selection_log)), exist_ok=True)

    # ------------------------------------------------------------------ frame boundaries
    def begin_frame(self, frame_name):
        self.frame = frame_name
        self.k = int(re.match(r"frame(\d+)", frame_name).group(1))
        self.rec = dict(frame=frame_name, k=self.k, mode=self.cfg.update_mode)
        self.raster = dict(calls=0, pairs=0)

    def end_frame(self, frame_sec):
        t0 = time.time()
        c = self.cfg
        if c.update_mode in ("C", "E", "EB") and self.scene is not None:
            self.prev_images = {cam.image_name: (cam.original_image[:3].clamp(0, 1) * 255).round().to(torch.uint8)
                                for cam in self.scene.getTrainCameras()}
        if c.update_mode in ("B", "EB") and self.hist is not None and c.selection_state:
            tmp = c.selection_state + ".tmp"
            box = None if self.box is None else [self.box[0].cpu(), self.box[1].cpu()]
            acc = None if self.accum is None else dict(acc={k: v.cpu() for k, v in self.accum.acc.items()},
                                                       pairs=self.accum.pairs)
            torch.save(dict(frame_k=self.k, hist=self.hist.state_dict(), box=box, accum=acc), tmp)
            os.replace(tmp, c.selection_state)
        self.rec["io_sec"] = time.time() - t0
        self.rec["frame_sec_upstream"] = frame_sec
        if self.raster["calls"]:
            self.rec["raster_calls"] = self.raster["calls"]
            self.rec["tile_pairs_mean"] = self.raster["pairs"] / self.raster["calls"]
        with open(c.selection_log, "a") as f:
            f.write(json.dumps(self.rec) + "\n")
        self.scene = None

    # ------------------------------------------------------------------ hooks
    def before_stage1(self, g):
        """Called right after training_one_frame_setup (NTC loaded, bounds fixed)."""
        c = self.cfg
        _sync(); t0 = time.time()
        xyz = g._xyz.detach()
        n = xyz.shape[0]
        with torch.no_grad():
            cx = g.ntc.get_contracted_xyz(xyz)
            inside = ((cx >= 0) & (cx <= 1)).all(dim=1)
        sel, full, extra = None, True, {}
        region = inside
        if c.update_mode in ("E", "EB"):
            region, estats = self._adaptive_box(xyz, inside)
            extra.update(estats)
        if c.update_mode in ("B", "EB"):
            if self._pending_state is not None:
                st = self._pending_state
                self._pending_state = None
                if st["frame_k"] == self.k - 1 and st["hist"]["n"] == n:
                    self.hist = su.MotionHistory.from_state(st["hist"], xyz.device)
                    extra["state_resumed"] = True
                else:
                    extra["state_discarded"] = f"state frame {st['frame_k']} / n {st['hist']['n']}"
            if self.hist is None or self.hist.n != n:
                if self.hist is not None:
                    extra["history_reset"] = f"base count changed {self.hist.n} -> {n}"
                self.hist = su.MotionHistory(n, c.b_window, xyz.device)
            full = su.b_is_full_frame(self.k, c.b_warmup, c.b_recheck)
            if not full:
                with torch.no_grad():
                    sel = su.select_by_motion(self.hist, c.b_tau_xyz, c.b_tau_rot_deg) & region
            elif c.update_mode == "EB":
                sel = region                  # B's full frame = the whole adaptive box
        elif c.update_mode == "E":
            sel, full = region, False
        elif c.update_mode == "C":
            full = False
            views = []
            for cam in self.scene.getTrainCameras():
                cur = cam.original_image[:3]
                prev = self.prev_images.get(cam.image_name)
                if prev is None:
                    prev = _load_image(self._prev_path(cam.image_name), cur.shape[1:], cur.device)
                views.append((cur, prev, cam.full_proj_transform, cam.world_view_transform))
            with torch.no_grad():
                active, mstats = su.select_by_change(xyz, views, c.c_kappa, c.c_tau, c.c_radius)
            sel = active & inside
            extra.update(mstats)
        n_inside = int(inside.sum())
        if c.update_mode in ("E", "EB"):
            extra["n_region"] = int(region.sum())
        n_sel = n_inside if sel is None else int(sel.sum())
        fallback = False
        if sel is not None and n_sel < c.min_selected:
            sel, fallback, n_sel, full = None, True, n_inside, True
        g.ntc._sel_mask = sel
        _sync()
        self.rec.update(n_base=n, n_inside=n_inside, n_selected=n_sel, full_update=bool(full),
                        fallback=fallback, select_sec=time.time() - t0, **extra)

    def _adaptive_box(self, xyz, inside):
        """E/EB: accumulate this frame's change masks; recompute the box on box frames (past window only)."""
        c = self.cfg
        st = {}
        if self._pending_state is not None and self._pending_state.get("frame_k") == self.k - 1:
            ps = self._pending_state
            if ps.get("box") is not None:
                self.box = (ps["box"][0].to(xyz.device), ps["box"][1].to(xyz.device))
            if ps.get("accum") is not None:
                self.accum.acc = {k: v.to(xyz.device) for k, v in ps["accum"]["acc"].items()}
                self.accum.pairs = ps["accum"]["pairs"]
            st["box_state_resumed"] = True
        cams = self.scene.getTrainCameras()
        with torch.no_grad():
            for cam in cams:
                cur = cam.original_image[:3]
                prev = self.prev_images.get(cam.image_name)
                if prev is None:
                    prev = _load_image(self._prev_path(cam.image_name), cur.shape[1:], cur.device)
                self.accum.add(cam.image_name, cur, prev)
            self.accum.frame_done()
            if su.e_is_box_frame(self.k, c.e_every) or self.box is None:
                views = [(cam.image_name, cam.full_proj_transform, cam.world_view_transform) for cam in cams]
                lo, hi, bst = su.select_box(xyz, self.accum, views, c.c_kappa, c.e_q, 1.0 - c.e_q, c.e_margin, c.e_min_points)
                st.update(bst)
                st["box_recomputed"] = True
                if lo is not None:
                    self.box = (lo, hi)
                self.accum.reset()
            if self.box is None:
                st["box_none"] = True
                return inside, st
            region = su.in_box(xyz, *self.box) & inside
        st["box_frac_of_inside"] = float(region.sum()) / max(int(inside.sum()), 1)
        return region, st

    def after_stage1(self, g):
        """Called at the start of update_by_ntc: _d_xyz/_d_rot hold the final stage-1 NTC output."""
        c = self.cfg
        _sync(); t0 = time.time()
        observed = getattr(g.ntc, "_last_mask", None)
        if g._d_xyz is not None and observed is not None:
            with torch.no_grad():
                disp = g._d_xyz.float().norm(dim=1)
                angle = su.quat_angle_deg(g._d_rot)
                if c.update_mode in ("B", "EB") and self.hist is not None:
                    self.hist.push(disp, angle, observed)
                _sync()
                self.rec["bookkeeping_sec"] = time.time() - t0
                if not c.no_motion_log:
                    self.rec.update(su.motion_quantiles(disp, angle, observed))
                if c.dump_motion:
                    import numpy as np
                    os.makedirs(c.dump_motion, exist_ok=True)
                    nan = torch.tensor(float("nan"), device=disp.device)
                    for key, v in (("disp", disp), ("rot", angle)):
                        np.save(os.path.join(c.dump_motion, f"{self.frame}_{key}.npy"),
                                torch.where(observed, v, nan).half().cpu().numpy())
        g.ntc._sel_mask = None

    def _prev_path(self, image_name):
        prev = "frame%06d" % (self.k - 1)
        d = os.path.join(self.video_path, prev, self.images_dir)
        hits = sorted(glob.glob(os.path.join(d, image_name + ".*")))
        if not hits:
            raise FileNotFoundError(f"previous-frame image for {image_name} not found in {d}")
        return hits[0]


SELECTOR = None

# kernel-name substrings -> category for the diagnostic profile (unknown kernels go to "other"; the raw top
# kernels are saved too so the grouping can be redone offline)
_CATEGORIES = (
    ("raster", ("render", "preprocess", "duplicateWithKeys", "identifyTileRanges", "computeCov", "Radix",
                "radix", "DeviceScan", "cub::", "rasteriz", "markVisible", "checkFrustum")),
    ("ntc", ("tcnn", "fully_fused", "kernel_grid", "hash", "encoding", "mlp", "Mlp", "network")),
)


def _profiled(fn, a):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        res = fn(*a)
        _sync()
    tot = {"raster": 0.0, "ntc": 0.0, "other": 0.0}
    rows = []
    for e in p.key_averages():
        t = getattr(e, "cuda_time_total", None)
        if t is None:
            t = getattr(e, "device_time_total", 0.0)
        if not t or e.key.startswith("aten::") or e.key.startswith("cuda") or e.key.startswith("Profiler"):
            continue
        cat = "other"
        for name, keys in _CATEGORIES:
            if any(k in e.key for k in keys):
                cat = name; break
        tot[cat] += t / 1e3
        rows.append((t / 1e3, e.key[:120], cat))
    rows.sort(reverse=True)
    out = dict(frame=SELECTOR.frame, gpu_ms_by_category={k: round(v, 2) for k, v in tot.items()},
               top_kernels=[dict(ms=round(t, 3), kernel=k, cat=c) for t, k, c in rows[:60]])
    path = SELECTOR.cfg.profile_out or (SELECTOR.cfg.selection_log + ".profile.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(out) + "\n")
    SELECTOR.rec["profiled"] = True
    return res


def _install_raster_counter():
    """Count Gaussian-tile pairs (num_rendered) of every rasterizer call without touching the package."""
    import diff_gaussian_rasterization as dgr
    real = dgr._C

    class Proxy:
        def __getattr__(self, name):
            return getattr(real, name)

        def rasterize_gaussians(self, *args):
            r = real.rasterize_gaussians(*args)
            if SELECTOR is not None:
                try:
                    SELECTOR.raster["pairs"] += int(r[0]); SELECTOR.raster["calls"] += 1
                except Exception:
                    pass
            return r
    dgr._C = Proxy()


def install_hooks(tf):
    from scene import Scene, GaussianModel
    from ntc import NeuralTransformationCache

    orig_scene_init = Scene.__init__

    def scene_init(self, *a, **k):
        orig_scene_init(self, *a, **k)
        SELECTOR.scene = self

    def ntc_forward(self, xyz):
        contracted_xyz = self.get_contracted_xyz(xyz)
        mask = ((contracted_xyz >= 0) & (contracted_xyz <= 1)).all(dim=1)
        sel = getattr(self, "_sel_mask", None)
        if sel is not None and sel.shape[0] == mask.shape[0]:
            mask = mask & sel
        self._last_mask = mask
        resi = self.model(contracted_xyz[mask])
        masked_d_xyz = resi[:, :3]
        masked_d_rot = resi[:, 3:7]
        d_xyz = torch.full((xyz.shape[0], 3), 0.0, dtype=torch.half, device=xyz.device)
        d_rot = torch.full((xyz.shape[0], 4), 0.0, dtype=torch.half, device=xyz.device)
        d_rot[:, 0] = 1.0
        d_xyz[mask] = masked_d_xyz
        d_rot[mask] = masked_d_rot
        return mask, d_xyz, d_rot

    orig_setup = GaussianModel.training_one_frame_setup

    def setup(self, training_args):
        orig_setup(self, training_args)
        SELECTOR.before_stage1(self)

    orig_update = GaussianModel.update_by_ntc

    def update(self):
        SELECTOR.after_stage1(self)
        orig_update(self)

    orig_train_one_frame = tf.train_one_frame

    def train_one_frame(lp, op, pp, args):
        SELECTOR.begin_frame(os.path.basename(os.path.normpath(args.source_path)))
        prof = SELECTOR.k in SELECTOR.cfg.profile_frames
        t0 = time.time()
        if prof:
            res = _profiled(orig_train_one_frame, (lp, op, pp, args))
        else:
            res = orig_train_one_frame(lp, op, pp, args)
        SELECTOR.end_frame(time.time() - t0)
        return res

    Scene.__init__ = scene_init
    NeuralTransformationCache.forward = ntc_forward
    GaussianModel.training_one_frame_setup = setup
    GaussianModel.update_by_ntc = update
    tf.train_one_frame = train_one_frame


def main():
    global SELECTOR
    cfg, rest = own_args(sys.argv[1:])
    import train_frames as tf
    from argparse import ArgumentParser
    from arguments import ModelParams, PipelineParams, OptimizationParams

    # --- identical to the upstream __main__ block
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--frame_start', type=int, default=1)
    parser.add_argument('--frame_end', type=int, default=150)
    parser.add_argument('--load_iteration', type=int, default=None)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1, 50, 100])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1, 50, 100])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--read_config", action='store_true', default=False)
    parser.add_argument("--config_path", type=str, default=None)
    args = parser.parse_args(rest)
    if args.output_path == "":
        args.output_path = args.model_path
    if args.read_config and args.config_path is not None:
        with open(args.config_path, 'r') as f:
            config = json.load(f)
        for key, value in config.items():
            if key not in ["output_path", "source_path", "model_path", "video_path", "debug_from"]:
                setattr(args, key, value)
    serializable = {k: v for k, v in vars(args).items()
                    if isinstance(v, (int, float, str, bool, list, dict, tuple, type(None)))}
    serializable["selective_update"] = vars(cfg)
    os.makedirs(args.output_path, exist_ok=True)
    with open(os.path.join(args.output_path, "cfg_args.json"), 'w') as f:
        f.write(json.dumps(serializable, default=str))
    # ---
    SELECTOR = Selector(cfg, args.video_path, args.images)
    install_hooks(tf)
    try:
        _install_raster_counter()
    except Exception as e:                       # logging only; never block a run
        print(f"[selective-update] raster counter not installed: {e}")
    print(f"[selective-update] mode={cfg.update_mode} " + json.dumps(vars(cfg), default=str))
    tf.train_frames(lp, op, pp, args)


if __name__ == "__main__":
    main()
