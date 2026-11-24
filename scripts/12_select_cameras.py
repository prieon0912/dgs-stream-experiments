"""Create the 12 camera-selection conditions on CPU; no model or images needed."""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.camera_selection import load_geometry, make_plan, save_json_checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--frame-dir", help="frame000000 containing sparse[/0]/images.bin")
    source.add_argument("--cameras-json", help="Exported camera geometry; also usable on a laptop")
    parser.add_argument("--counts", nargs="+", type=int, default=[4, 8, 12, 18])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--anchor", help="Common reference camera (default: first eligible name)")
    parser.add_argument("--test-cam", default="cam00")
    parser.add_argument("--exclude", nargs="*", default=[], help="Additional held-out cameras")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    cameras = load_geometry(args.frame_dir, args.cameras_json)
    plan = make_plan(cameras, args.counts, args.seed, [args.test_cam] + args.exclude, args.anchor)
    out = Path(args.out_dir)
    # Validate the manifest before writing individual condition files.
    save_json_checked(out / "selection_plan.json", plan)
    save_json_checked(out / "cameras.json", dict(coordinate_system="colmap_world", cameras=cameras))
    for condition in plan["conditions"]:
        save_json_checked(out / "camera_lists" / (condition["condition_id"] + ".json"), condition)
        duplicate = f" (same set as {condition['same_view_set_as']})" if condition["same_view_set_as"] else ""
        print(f"{condition['condition_id']}: {', '.join(condition['camera_names'])}{duplicate}")
    with (out / "selection_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition_id", "method", "n_views", "seed", "anchor",
                                                   "camera_names", "same_view_set_as"])
        writer.writeheader()
        for condition in plan["conditions"]:
            row = {key: condition[key] for key in writer.fieldnames}
            row["camera_names"] = " ".join(row["camera_names"])
            writer.writerow(row)
    print(json.dumps(dict(n_conditions=len(plan["conditions"]), out_dir=str(out.resolve())), indent=2))


if __name__ == "__main__":
    main()
