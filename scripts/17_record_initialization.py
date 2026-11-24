"""Collect first-frame counts, before/after quality, timings and occupancy into JSON + CSV."""
import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.aabb import load_aabb, load_xyz, point_stats, file_sha256
from common.diagnostics import occupancy, sources_path
from common.result_io import read_json, write_json, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--aabb", required=True)
    parser.add_argument("--pre-ply")
    parser.add_argument("--static-json")
    parser.add_argument("--pre-key", default="pre")
    parser.add_argument("--final-key", default="final")
    parser.add_argument("--voxel-size", type=float)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    box = load_aabb(args.aabb)
    xyz = load_xyz(args.ply)
    source_file = sources_path(args.ply)
    source = None
    if source_file.is_file():
        with np.load(source_file, allow_pickle=False) as data:
            source = data["view_ids"]
    timed = read_json(Path(args.ply).with_name(Path(args.ply).stem + "_timing.json"))
    static = read_json(args.static_json) if args.static_json else {}
    result = dict(run=args.run, ply=str(Path(args.ply).resolve()), init_sha256=file_sha256(args.ply),
                  aabb=box, counts=point_stats(xyz, box), timing=timed,
                  init_metadata=timed.get("init_metadata", {}),
                  quality_final=static.get(args.final_key, {}),
                  quality_pre_refine=static.get(args.pre_key, {}) if args.pre_ply else {},
                  occupancy=occupancy(xyz, box, args.voxel_size, source),
                  pre_ply=args.pre_ply,
                  pre_counts=point_stats(load_xyz(args.pre_ply), box) if args.pre_ply else None)
    write_json(args.out, result)
    flat = dict(run=args.run, **result["counts"], **result["occupancy"],
                init_time_sec=timed.get("init_time_sec"), aabb_id=box["aabb_id"])
    for prefix, quality in (("pre_refine", result["quality_pre_refine"]), ("final", result["quality_final"])):
        for metric in ("psnr", "psnr_png", "ssim", "lpips"):
            flat[f"{prefix}_{metric}"] = quality.get(metric)
    write_csv(Path(args.out).with_suffix(".csv"), [flat])
    print(args.out)


if __name__ == "__main__":
    main()
