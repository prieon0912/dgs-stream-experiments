"""Manually set ONE shared AABB, count centers, or verify stored NTC bounds (CPU)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.aabb import (load_aabb, save_aabb, load_xyz, point_stats,
                         inspect_ntc, bind_run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    default = commands.add_parser("default", help="Write the unchanged Flame Steak box")
    default.add_argument("--out", required=True)
    manual = commands.add_parser("manual", help="Specify world-coordinate corners")
    manual.add_argument("--min", nargs=3, type=float, required=True)
    manual.add_argument("--max", nargs=3, type=float, required=True)
    manual.add_argument("--out", required=True)
    stats = commands.add_parser("stats", help="Count centers inside the SAME box for multiple point sets")
    stats.add_argument("--aabb")
    stats.add_argument("--points", nargs="+", required=True)
    stats.add_argument("--out")
    inspect = commands.add_parser("inspect-ntc", help="Read xyz_bound_min/max saved in an NTC checkpoint")
    inspect.add_argument("--ntc", required=True)
    inspect.add_argument("--expect", help="Fail if the checkpoint differs from this AABB JSON")
    inspect.add_argument("--out")
    bind = commands.add_parser("bind-run", help="Internal: freeze PLY, AABB and NTC-config identity")
    bind.add_argument("--run-dir", required=True)
    bind.add_argument("--aabb")
    bind.add_argument("--ply", required=True)
    bind.add_argument("--ntc-config", required=True)
    bind.add_argument("--ntc-mode", choices=["official", "warmup"], required=True)
    bind.add_argument("--environment", help="Measured environment JSON; freezes code/software/GPU for resume")
    args = parser.parse_args()
    if args.command == "default":
        result = save_aabb(args.out, load_aabb())
    elif args.command == "manual":
        result = save_aabb(args.out, dict(min=args.min, max=args.max, method="manual"))
    elif args.command == "stats":
        box = load_aabb(args.aabb)
        result = dict(aabb=box, point_sets=[dict(path=str(Path(path).resolve()), **point_stats(load_xyz(path), box))
                                           for path in args.points])
    elif args.command == "inspect-ntc":
        result = inspect_ntc(args.ntc, load_aabb(args.expect) if args.expect else None)
    else:
        environment = json.loads(Path(args.environment).read_text(encoding="utf-8-sig")) if args.environment else None
        result = bind_run(args.run_dir, load_aabb(args.aabb), args.ply, args.ntc_config, args.ntc_mode, environment)
    if args.command in ("stats", "inspect-ntc") and args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
