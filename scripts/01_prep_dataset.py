"""N3DV videos -> 3DGStream dataset layout, disk-friendly and resumable.

Equivalent to README steps 5.1-5.4 (extract, copy_cams.py, convert_frames.py --resize) but:
  * processes frames in chunks and deletes full-res raw/undistorted PNGs right away
    (300 frames x 21 cams full-res would be ~100GB; we keep only images_2 ~13GB)
  * resizes with PIL (Ubuntu's ImageMagick 6 has no `magick` binary that convert_frames.py calls)
  * frame000000 is taken from the suite (the exact images/COLMAP the baseline init was trained on)
  * skips frames that are already complete -> safe to re-run after a pod restart

Output:
  out/distorted/sparse/0          (from suite)
  out/frame000000/{images, images_2, sparse/0}
  out/frameXXXXXX/{images_2, sparse/0}   for 1..n_frames-1
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL import Image

CAMS = [f"cam{i:02d}" for i in range(21)]


def resize_half(src, dst):
    im = Image.open(src)
    im.resize((im.width // 2, im.height // 2), Image.LANCZOS).save(dst)


def frame_done(fdir, cams):
    d = os.path.join(fdir, "images_2")
    return os.path.isdir(os.path.join(fdir, "sparse", "0")) and all(
        os.path.exists(os.path.join(d, f"{c}.png")) for c in cams)


def undistort(raw_dir, distorted_model, out_dir):
    if shutil.which("colmap"):
        cmd = ["colmap", "image_undistorter", "--image_path", raw_dir, "--input_path", distorted_model,
               "--output_path", out_dir, "--output_type", "COLMAP"]
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-2000:])
    else:
        import pycolmap
        pycolmap.undistort_images(out_dir, distorted_model, raw_dir)
    return os.path.join(out_dir, "images")


def process_frame(fid, out, suite_sparse, distorted_model, cams, scratch=None):
    fdir = os.path.join(out, f"frame{fid:06d}")
    raw = os.path.join(scratch or out, f"frame{fid:06d}", "raw")
    tmp = tempfile.mkdtemp(dir=scratch or fdir)
    try:
        und = undistort(raw, distorted_model, tmp)
        os.makedirs(os.path.join(fdir, "images_2"), exist_ok=True)
        for c in cams:
            resize_half(os.path.join(und, f"{c}.png"), os.path.join(fdir, "images_2", f"{c}.png"))
        dst = os.path.join(fdir, "sparse", "0")
        if os.path.exists(os.path.join(fdir, "sparse")):
            shutil.rmtree(os.path.join(fdir, "sparse"))
        shutil.copytree(suite_sparse, dst)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(raw, ignore_errors=True)
    return fid


def extract_chunk(videos, out, a, b, cams, scratch=None):
    """frames [a, b] inclusive, all cams -> out/frameXXXXXX/raw/camYY.png"""
    raw_root = scratch or out
    for c in cams:
        if all(os.path.isfile(os.path.join(raw_root, f"frame{fid:06d}", "raw", f"{c}.png"))
               for fid in range(a, b + 1)):
            continue
        tmp = os.path.join(raw_root, "_extract", c)
        os.makedirs(tmp, exist_ok=True)
        vf = f"select='between(n\\,{a}\\,{b})'"
        cmd = ["ffmpeg", "-threads", "2", "-loglevel", "error", "-y", "-i", os.path.join(videos, f"{c}.mp4"),
               "-vf", vf, "-vsync", "0", "-start_number", str(a), "-threads", "2",
               "-frames:v", str(b - a + 1), os.path.join(tmp, "%06d.png")]
        subprocess.run(cmd, check=True)
        for fid in range(a, b + 1):
            src = os.path.join(tmp, f"{fid:06d}.png")
            if not os.path.exists(src):
                raise RuntimeError(f"ffmpeg did not produce frame {fid} for {c}")
            dst_dir = os.path.join(raw_root, f"frame{fid:06d}", "raw")
            os.makedirs(dst_dir, exist_ok=True)
            shutil.move(src, os.path.join(dst_dir, f"{c}.png"))
    shutil.rmtree(os.path.join(raw_root, "_extract"), ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_frames", type=int, default=300)
    ap.add_argument("--chunk", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    s0 = os.path.join(args.suite, "frame000000")
    suite_sparse = os.path.join(s0, "sparse", "0")
    distorted_src = os.path.join(s0, "distorted")
    distorted_model = os.path.join(args.out, "distorted", "sparse", "0")
    if not os.path.isdir(distorted_model):
        shutil.copytree(distorted_src, os.path.join(args.out, "distorted"), dirs_exist_ok=True)
    cams = [c for c in CAMS if os.path.exists(os.path.join(args.videos, f"{c}.mp4"))]
    print(f"cams: {len(cams)} -> {cams[0]}..{cams[-1]}")

    # frame 0 from the suite (undistorted 'images' + sparse) -> add images_2
    f0 = os.path.join(args.out, "frame000000")
    if not frame_done(f0, cams):
        shutil.copytree(os.path.join(s0, "images"), os.path.join(f0, "images"), dirs_exist_ok=True)
        shutil.copytree(suite_sparse, os.path.join(f0, "sparse", "0"), dirs_exist_ok=True)
        os.makedirs(os.path.join(f0, "images_2"), exist_ok=True)
        for c in cams:
            resize_half(os.path.join(f0, "images", f"{c}.png"), os.path.join(f0, "images_2", f"{c}.png"))
    w0, h0 = Image.open(os.path.join(f0, "images_2", "cam00.png")).size
    print(f"frame000000 ready, images_2 size {w0}x{h0}")

    todo = [f for f in range(1, args.n_frames) if not frame_done(os.path.join(args.out, f"frame{f:06d}"), cams)]
    print(f"{len(todo)} frames to process")
    # Full-resolution intermediates use the container's local disk, not the network volume.
    # The frame selection, undistortion and PIL resizing remain identical.
    with tempfile.TemporaryDirectory(prefix="dgs-prep-") as scratch:
        print("temporary intermediates:", scratch, flush=True)
        for i in range(0, len(todo), args.chunk):
            chunk = todo[i:i + args.chunk]
            a, b = chunk[0], chunk[-1]
            print(f"[chunk] extracting {a}..{b}", flush=True)
            extract_chunk(args.videos, args.out, a, b, cams, scratch)
            with ProcessPoolExecutor(args.workers) as ex:
                futs = [ex.submit(process_frame, f, args.out, suite_sparse, distorted_model, cams, scratch)
                        for f in range(a, b + 1)]
                for fu in as_completed(futs):
                    fid = fu.result()
            print(f"[chunk] done {a}..{b}", flush=True)

    # sanity: size consistency with frame 0
    w, h = Image.open(os.path.join(args.out, f"frame{args.n_frames - 1:06d}", "images_2", "cam00.png")).size
    if (w, h) != (w0, h0):
        print(f"ERROR: size mismatch frame0 {w0}x{h0} vs last {w}x{h}", file=sys.stderr)
        sys.exit(1)
    print("dataset OK:", args.out)


if __name__ == "__main__":
    main()
