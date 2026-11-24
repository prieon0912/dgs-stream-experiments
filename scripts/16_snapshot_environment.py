"""Record actual software/GPU/code versions and a reusable source revision lock."""
import argparse
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.result_io import write_json


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def snapshot():
    versions, direct = {}, {}
    for name in ("torch", "torchvision", "numpy", "tinycudann", "depth-anything-3", "gsplat", "lpips", "plyfile"):
        try:
            distribution = importlib.metadata.distribution(name)
            versions[name] = distribution.version
            import json
            direct[name] = json.loads(distribution.read_text("direct_url.json") or "null")
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    repositories = {}
    for name, path in (("3DGStream", os.environ.get("REPO_GS", "/workspace/3DGStream")),
                       ("Depth-Anything-3", os.environ.get("REPO_DA3", "/workspace/Depth-Anything-3"))):
        repositories[name] = dict(path=path, commit=command(["git", "-C", path, "rev-parse", "HEAD"]),
                                  changes=command(["git", "-C", path, "status", "--porcelain"]))
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for folder in ("common", "scripts", "runpod", "stream"):
        for path in sorted((root / folder).glob("*")):
            if path.suffix in (".py", ".sh"):
                digest.update(str(path.relative_to(root)).replace("\\", "/").encode())
                digest.update(path.read_bytes())
    return dict(python=sys.version, platform=platform.platform(), packages=versions, direct_urls=direct,
                gpu=command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
                nvcc=command(["nvcc", "--version"]), repositories=repositories, experiment_code_sha256=digest.hexdigest(),
                pip_freeze=command([sys.executable, "-m", "pip", "freeze"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lock-out")
    args = parser.parse_args()
    data = snapshot()
    write_json(args.out, data)
    if args.lock_out:
        direct = data["direct_urls"].get("tinycudann") or {}
        lock = dict(gs_commit=data["repositories"]["3DGStream"]["commit"],
                    da3_commit=data["repositories"]["Depth-Anything-3"]["commit"],
                    tcnn_commit=direct.get("vcs_info", {}).get("commit_id"))
        if any(not value for value in lock.values()):
            raise SystemExit("Cannot write a revision lock: install both repositories and tinycudann first")
        write_json(args.lock_out, lock)
    print(args.out)


if __name__ == "__main__":
    main()
