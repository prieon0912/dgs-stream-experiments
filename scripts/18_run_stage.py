"""Run one initialization stage and atomically mark verified outputs for safe resume."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.aabb import file_sha256
from common.result_io import read_json, write_json
from common.provenance import environment_fingerprint


def run_stage(marker, signature, environment_files, outputs, command):
    marker = Path(marker)
    environments = [environment_fingerprint(read_json(path)) for path in environment_files]
    contract = dict(signature=signature, environments=environments, command=command)
    previous = read_json(marker)
    if previous:
        if previous.get("contract") != contract:
            raise ValueError("Stage settings/code/environment changed. Use a new experiment tag and output directory.")
        for path, digest in previous["outputs"].items():
            if not Path(path).is_file() or file_sha256(path) != digest:
                raise ValueError(f"Completed stage output changed/missing: {path}")
        print(f"SKIP verified stage: {marker}")
        return
    # Preserve partial outputs before retrying; never mistake them for completed outputs.
    stamp = str(time.time_ns())
    for path in map(Path, outputs):
        if path.exists():
            if not path.is_file():
                raise ValueError(f"Expected a file, found a directory: {path}")
            path.rename(path.with_name(path.name + ".incomplete_" + stamp))
    subprocess.run(command, check=True)
    if not all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in outputs):
        raise RuntimeError("Stage exited without producing all expected output files")
    write_json(marker, dict(contract=contract, outputs={str(path): file_sha256(path) for path in outputs}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--signature", required=True)
    parser.add_argument("--environment", action="append", default=[])
    parser.add_argument("--outputs", nargs="+", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A command after -- is required")
    run_stage(args.marker, args.signature, args.environment, args.outputs, command)


if __name__ == "__main__":
    main()
