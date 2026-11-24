"""[env: gs] Videos from streaming results: GT and each run side by side, at cam00 / another camera / a moving path.

  # cam00 (test view), saved renders, fast:
  python scripts/11_make_video.py --runs baseline_own:"3DGS 15k" da3_raw:"DA3" --out $RUNS/report/cam00.mp4
  # another real camera (a TRAINING view -> looks better than cam00; for showing, not for scoring):
  python scripts/11_make_video.py --runs baseline_own da3_raw --cam cam05 --out $RUNS/report/cam05.mp4
  # moving virtual camera (no GT exists):
  python scripts/11_make_video.py --runs baseline_own da3_raw --path spiral --out $RUNS/report/spiral.mp4

Sources per frame f of a run:
  cam00, f>=1     <run>/out/frameF/0_rendering2.png (3DGStream's own final render)
  else, f>=1      render( out/frameF/point_cloud/iteration_150/point_cloud.ply          (all gaussians after NTC)
                        + out/frameF/point_cloud/iteration_250/added/point_cloud.ply )  (frame-specific added ones)
  f == 0          render( <run>/init/... ) unless --no_frame0
The re-render path is checked once against 0_rendering2.png at cam00 (prints PSNR; should be > 40 dB).

--path spiral|sweep: LLFF-style path around the average camera, looking at the scene's median depth.
  N3DV cameras form a short forward-facing row: stay inside it (--radius 0.5..1.0 of the camera spread).
"""
import argparse
import os
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

RUNS = os.environ.get("RUNS", "/workspace/runs")
SCENE = os.environ.get("SCENE", "/workspace/data/flame_steak")


# ---------------------------------------------------------------- drawing
def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def label(img, lines, f):
    d = ImageDraw.Draw(img)
    y = 10
    for ln in lines:
        d.text((12, y), ln, font=f, fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
        y += getattr(f, "size", 14) + 6


# ---------------------------------------------------------------- cameras
def look_at_w2c(pos, target, down_hint):
    """OpenCV convention (x right, y down, z forward)."""
    fwd = target - pos; fwd /= np.linalg.norm(fwd)
    right = np.cross(down_hint, fwd); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    Rc2w = np.stack([right, down, fwd], 1)
    w2c = np.eye(4)
    w2c[:3, :3] = Rc2w.T
    w2c[:3, 3] = -Rc2w.T @ pos
    return w2c


def make_path(cams, n, kind, radius, cycles, focus):
    """Poses (w2c) for n frames around the average camera of `cams`."""
    import copy
    R = np.stack([c.w2c[:3, :3].T for c in cams])               # c2w rotations
    centers = np.stack([c.center for c in cams])
    c = centers.mean(0)
    right, down, fwd = (R[:, :, i].mean(0) for i in range(3))
    fwd /= np.linalg.norm(fwd)
    right -= fwd * (right @ fwd); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    off = centers - c
    rx = np.percentile(np.abs(off @ right), 90) * radius
    ry = np.percentile(np.abs(off @ down), 90) * radius
    target = c + focus * fwd
    ref = min(cams, key=lambda k: k.name)                       # intrinsics/size of cam00
    out = []
    for k in range(n):
        th = 2 * np.pi * cycles * k / max(n, 1)
        if kind == "spiral":
            p = c + rx * np.cos(th) * right + ry * np.sin(th) * down
        else:                                                   # sweep: left-right only, starts at center
            p = c + rx * np.sin(th) * right
        cam = copy.copy(ref)
        cam.w2c = look_at_w2c(p, target, down)
        out.append(cam)
    print(f"path {kind}: {n} poses, radius x {rx:.3f} y {ry:.3f}, focus {focus:.2f} "
          f"(camera spread x {np.abs(off @ right).max():.3f})")
    return out


# ---------------------------------------------------------------- gaussians
def frame_plys(run_dir, fid):
    if fid == 0:
        return [os.path.join(run_dir, "init", "point_cloud", "iteration_15000", "point_cloud.ply")]
    pc = os.path.join(run_dir, "out", f"frame{fid:06d}", "point_cloud")
    return [os.path.join(pc, "iteration_150", "point_cloud.ply"),
            os.path.join(pc, "iteration_250", "added", "point_cloud.ply")]


def load_gs(paths):
    from common.gs_render import GS
    from common.ply_io import load_ply
    ds = [load_ply(p) for p in paths if os.path.exists(p)]
    if not ds:
        raise FileNotFoundError(paths[0])
    assert len({x["f_rest"].shape[1] for x in ds}) == 1, f"SH degree differs between {paths}"
    d = {k: np.concatenate([x[k] for x in ds]) for k in ds[0]}
    return GS(d)


def render_np(gs, cam, bg):
    import torch
    from common.gs_render import RCam, render
    with torch.no_grad():
        img = render(gs, RCam(cam), bg=bg).clamp(0, 1)
    return (img.permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)


def check_rerender(run_dir, fid, cam00, bg):
    """Re-rendered cam00 vs 3DGStream's own 0_rendering2.png: validates frame_plys()."""
    gt = np.asarray(Image.open(os.path.join(run_dir, "out", f"frame{fid:06d}", "0_rendering2.png")).convert("RGB"),
                    np.float32)
    if gt.shape[:2] != (cam00.height, cam00.width):
        print(f"  re-render check skipped: 0_rendering2.png {gt.shape[1]}x{gt.shape[0]} vs camera {cam00.width}x{cam00.height}")
        return
    res = {}
    for tag, paths in (("150+added", frame_plys(run_dir, fid)), ("150 only", frame_plys(run_dir, fid)[:1])):
        img = render_np(load_gs(paths), cam00, bg).astype(np.float32)
        res[tag] = 10 * np.log10(255 ** 2 / max(np.mean((img - gt) ** 2), 1e-10))
    print(f"  re-render check frame {fid}: " + ", ".join(f"{k} {v:.1f} dB" for k, v in res.items()))
    if res["150+added"] < 35:
        print("  WARNING: re-render does not match 3DGStream's render (< 35 dB) -> check bg (--bg) / ply layout")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help='run or run:"label"')
    ap.add_argument("--runs_dir", default=os.path.join(RUNS, "stream"))
    ap.add_argument("--out", required=True, help=".mp4")
    ap.add_argument("--cam", default="cam00", help="real camera to render from (cam00 = test view)")
    ap.add_argument("--path", choices=["spiral", "sweep"], default=None, help="moving virtual camera instead of --cam")
    ap.add_argument("--radius", type=float, default=0.7, help="path radius as a fraction of the camera spread")
    ap.add_argument("--cycles", type=float, default=2.0, help="path loops over the whole video")
    ap.add_argument("--focus", type=float, default=0.0, help="look-at distance (0 = median gaussian depth)")
    ap.add_argument("--bg", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="background (3DGStream cfg: black)")
    ap.add_argument("--frames", default="", help="e.g. 0:300 or 0:300:2 (default: all common frames)")
    ap.add_argument("--fps", type=int, default=30, help="N3DV is 30 fps")
    ap.add_argument("--cols", type=int, default=0, help="tiles per row (0 = auto)")
    ap.add_argument("--scale", type=float, default=0.0, help="tile scale (0 = auto: 1.0 for <=2 tiles, else 0.5)")
    ap.add_argument("--no_gt", action="store_true")
    ap.add_argument("--no_frame0", action="store_true")
    ap.add_argument("--crf", type=int, default=18)
    args = ap.parse_args()

    use_png = args.path is None and args.cam == "cam00"         # saved 3DGStream renders available
    show_gt = not args.no_gt and args.path is None
    bg = tuple(args.bg)

    runs = []
    for item in args.runs:
        name, lab = (item.split(":", 1) + [item])[:2]
        R = os.path.join(args.runs_dir, name)
        out = os.path.join(R, "out")
        need = "0_rendering2.png" if use_png else os.path.join("point_cloud", "iteration_150", "point_cloud.ply")
        done = {int(f[5:]) for f in os.listdir(out) if f.startswith("frame") and os.path.exists(os.path.join(out, f, need))}
        psnr = {}
        csv = os.path.join(R, "per_frame.csv")
        if use_png and os.path.exists(csv):                     # PSNR overlay only for the scored view
            import pandas as pd
            df = pd.read_csv(csv)
            psnr = dict(zip(df.frame.astype(int), df.psnr_main))
        runs.append(dict(name=name, label=lab, dir=R, frames=done, psnr=psnr))
    frames = sorted(set.intersection(*[r["frames"] for r in runs]))
    if not frames:
        raise SystemExit("no frame is finished in all runs")
    if any(r["frames"] != set(frames) for r in runs):
        print(f"NOTE: using the {len(frames)} frames common to all runs ({frames[0]}..{frames[-1]})")
    if not args.no_frame0:
        frames = [0] + frames
    if args.frames:
        sl = slice(*[int(x) if x else None for x in args.frames.split(":")])
        frames = [f for f in frames if f in set(range(1000000)[sl])]

    # cameras
    cams = poses = None
    if not use_png or 0 in frames:
        from common.colmap_io import load_cameras
        allcams = load_cameras(os.path.join(SCENE, "frame000000"), "images_2")
        byname = {c.name: c for c in allcams}
        if args.path:
            focus = args.focus
            if focus <= 0:
                from common.ply_io import load_ply
                ref = min(allcams, key=lambda c: c.name)
                xyz = load_ply(frame_plys(runs[0]["dir"], 0)[0])["xyz"]
                z = xyz @ ref.w2c[:3, :3].T[:, 2] + ref.w2c[2, 3]
                focus = float(np.median(z[z > 0]))
            poses = make_path(allcams, len(frames), args.path, args.radius, args.cycles, focus)
        else:
            if args.cam not in byname:
                raise SystemExit(f"--cam {args.cam}: not in {sorted(byname)}")
            cams = byname[args.cam]
        if not use_png:
            first = next((f for f in frames if f > 0), None)
            if first is not None:
                for r in runs:
                    check_rerender(r["dir"], first, byname["cam00"], bg)

    n_tiles = len(runs) + (1 if show_gt else 0)
    cols = args.cols or (n_tiles if n_tiles <= 2 else int(np.ceil(n_tiles / 2)))
    rows = int(np.ceil(n_tiles / cols))
    scale = args.scale or (1.0 if n_tiles <= 2 else 0.5)
    f = font(int(28 * max(scale, 0.5) * 1.4))
    view = f"path {args.path}" if args.path else args.cam
    tmp = tempfile.mkdtemp(prefix="vid_")

    try:
        for k, fid in enumerate(frames):
            fr = f"frame{fid:06d}"
            cam = poses[k] if poses else cams
            tiles = []
            if show_gt:
                tiles.append((Image.open(os.path.join(SCENE, fr, "images_2", f"{args.cam}.png")).convert("RGB"),
                              ["GT " + args.cam, f"frame {fid}"]))
            for r in runs:
                if use_png and fid > 0:
                    im = Image.open(os.path.join(r["dir"], "out", fr, "0_rendering2.png")).convert("RGB")
                else:
                    im = Image.fromarray(render_np(load_gs(frame_plys(r["dir"], fid)), cam, bg))
                lines = [r["label"] + (" (init)" if fid == 0 else "")]
                if not show_gt and len(tiles) == 0:
                    lines.append(f"{view} · frame {fid}")
                if fid in r["psnr"]:
                    lines.append(f"{r['psnr'][fid]:.2f} dB")
                tiles.append((im, lines))
            W0, H0 = tiles[0][0].size
            w, h = int(W0 * scale) // 2 * 2, int(H0 * scale) // 2 * 2   # even sizes for yuv420p
            canvas = Image.new("RGB", (w * cols, h * rows), (0, 0, 0))
            for i, (im, lines) in enumerate(tiles):
                im = im.resize((w, h), Image.LANCZOS)
                label(im, lines, f)
                canvas.paste(im, ((i % cols) * w, (i // cols) * h))
            canvas.save(os.path.join(tmp, f"v_{k:05d}.png"))
            if k % 30 == 0:
                print(f"  {k + 1}/{len(frames)}", flush=True)

        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
                        "-i", os.path.join(tmp, "v_%05d.png"), "-c:v", "libx264", "-crf", str(args.crf),
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"saved {args.out}  ({len(frames)} frames, {len(frames) / args.fps:.1f}s, {cols}x{rows} tiles, view {view})")


if __name__ == "__main__":
    main()
