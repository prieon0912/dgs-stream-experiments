"""Export all planned conditions to organized CSV files, without a GPU."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.collect_results import collect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--path-map", action="append", default=[], help="For downloaded results: /workspace=C:/local/results")
    parser.add_argument("--require-complete", action="store_true", help="Write CSVs, then fail if any condition is not eligible")
    args = parser.parse_args()
    mappings = [tuple(value.split("=", 1)) for value in args.path_map]
    if any(len(pair) != 2 for pair in mappings):
        parser.error("--path-map must be OLD=NEW")
    rows = collect(args.manifest, args.out_dir, mappings)
    eligible = sum(row["comparison_eligible"] for row in rows)
    print(f"CSV exported: {len(rows)} planned conditions, {eligible} eligible; {Path(args.out_dir).resolve()}")
    if args.require_complete and eligible != len(rows):
        raise SystemExit("Some conditions are incomplete or need review; see run_status.csv")


if __name__ == "__main__":
    main()
