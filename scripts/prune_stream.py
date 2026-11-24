"""[env: any] Free disk space of 3DGStream runs + repair a frame cut off by a full disk.

3DGStream saves the FULL gaussian set of every frame (out/frameX/point_cloud/iteration_150/point_cloud.ply,
~100 MB per 1M gaussians) -> 300 frames x several runs fills an 80 GB volume.  That file is only needed to
start the NEXT frame (and to resume), so for all finished frames except the newest --keep we record its
gaussian count in n_points.txt (read by 09_eval_stream.py) and delete it.  Metrics / storage inputs
(0_rendering2.png, NTC.pth, iteration_250/added) are kept.

  python scripts/prune_stream.py $RUNS/stream/baseline_own $RUNS/stream/da3_raw            # prune
  python scripts/prune_stream.py $RUNS/stream/da3_raw --repair                              # + fix cut-off frame

--repair: checks the newest finished frame's full PLY against its header (size = header + N * stride).
A truncated / missing file means the disk filled while saving -> that frame dir is deleted so
08_run_stream.sh resumes from the frame before it.
"""
import argparse
import os
import shutil

PLY_TYPES = {"char": 1, "uchar": 1, "short": 2, "ushort": 2, "int": 4, "uint": 4, "float": 4, "double": 8,
             "int8": 1, "uint8": 1, "int16": 2, "uint16": 2, "int32": 4, "uint32": 4, "float32": 4, "float64": 8}


def ply_header(path):
    """(n_vertices, expected_file_size) for a binary PLY with one vertex element, else None."""
    n, stride, hdr = None, 0, 0
    with open(path, "rb") as f:
        for line in f:
            hdr += len(line)
            if line.startswith(b"element vertex"):
                n = int(line.split()[-1])
            elif line.startswith(b"property") and n is not None:
                stride += PLY_TYPES.get(line.split()[1].decode(), 0)
            elif line.startswith(b"end_header"):
                return n, hdr + n * stride if n is not None else None
            if hdr > 1 << 16:
                break
    return None


def full_ply(fdir):
    return os.path.join(fdir, "point_cloud", "iteration_150", "point_cloud.ply")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run dirs ($RUNS/stream/<name>)")
    ap.add_argument("--keep", type=int, default=3, help="newest finished frames whose full PLY is kept")
    ap.add_argument("--repair", action="store_true")
    args = ap.parse_args()

    for R in args.runs:
        out = os.path.join(R, "out")
        if not os.path.isdir(out):
            print(f"{R}: no out/ dir"); continue
        frames = sorted(f for f in os.listdir(out) if f.startswith("frame"))
        done = [f for f in frames if os.path.exists(os.path.join(out, f, "0_rendering2.png"))]

        if args.repair:
            # unfinished dirs (no rendering) are partial by definition
            for f in sorted(set(frames) - set(done)):
                print(f"{R}: removing unfinished {f}"); shutil.rmtree(os.path.join(out, f))
            while done:
                p = full_ply(os.path.join(out, done[-1]))
                h = ply_header(p) if os.path.exists(p) else None
                ok = h is not None and os.path.getsize(p) == h[1] and \
                    os.path.getsize(os.path.join(out, done[-1], "0_rendering2.png")) > 0
                if ok:
                    print(f"{R}: newest finished frame {done[-1]} is intact ({h[0]:,} gaussians)"); break
                print(f"{R}: {done[-1]} is truncated/missing -> removed, will be recomputed")
                shutil.rmtree(os.path.join(out, done[-1])); done.pop()

        freed = 0
        for f in done[:-args.keep] if args.keep > 0 else done:
            p = full_ply(os.path.join(out, f))
            if not os.path.exists(p):
                continue
            h = ply_header(p)
            with open(os.path.join(out, f, "n_points.txt"), "w") as fp:
                fp.write(str(h[0] if h else "nan"))
            freed += os.path.getsize(p); os.remove(p)
        print(f"{R}: {len(done)} finished frames, freed {freed / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
