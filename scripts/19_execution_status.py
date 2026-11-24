"""CPU execution-status records and queue summaries."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.result_io import read_json, write_json, write_csv
from common.run_audit import audit_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record")
    record.add_argument("--run-dir", required=True)
    record.add_argument("--stage", required=True)
    record.add_argument("--state", required=True)
    record.add_argument("--exit-code", type=int)
    audit = commands.add_parser("audit")
    audit.add_argument("--run-dir", required=True)
    audit.add_argument("--expected-frames", type=int, required=True)
    queue = commands.add_parser("queue")
    queue.add_argument("--queue", required=True)
    queue.add_argument("--runs-dir", required=True)
    queue.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "record":
        write_json(Path(args.run_dir) / "execution_status.json",
                   dict(stage=args.stage, state=args.state, exit_code=args.exit_code,
                        updated_utc=datetime.now(timezone.utc).isoformat()))
    elif args.command == "audit":
        result = audit_run(args.run_dir, args.expected_frames, verify_images=True)
        write_json(Path(args.run_dir) / "completion.json", result)
        if not result["complete"]:
            raise SystemExit("Stream incomplete: see completion.json")
    else:
        rows = []
        for line in Path(args.queue).read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            name = line.split()[0]
            data = read_json(Path(args.runs_dir) / name / "execution_status.json")
            rows.append(dict(run=name, stage=data.get("stage"), state=data.get("state", "not_started"),
                             exit_code=data.get("exit_code"), updated_utc=data.get("updated_utc")))
        write_csv(args.out, rows)
        if any(row["state"] != "complete" or row["exit_code"] != 0 for row in rows):
            raise SystemExit("Queue has incomplete/failed conditions; see queue status CSV")


if __name__ == "__main__":
    main()
