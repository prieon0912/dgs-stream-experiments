"""CPU-only log parsing and completion checks; never substitute PNG PSNR for float PSNR."""
import math
import re
from pathlib import Path

from common.result_io import read_json, finite_number


NUMBER = r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?"
FRAME_RE = re.compile(r"Optimizing .*?(frame\d+)")
EVAL_RE = re.compile(r"\[ITER (\d+)\] Evaluating test: L1 (" + NUMBER + r") PSNR (" + NUMBER + r")")
TIME_RE = re.compile(r"Frame (frame\d+) finished in (" + NUMBER + r") seconds")
STAGES = {key: re.compile(label + r": (" + NUMBER + r")")
          for key, label in (("t_prep", "Preparation"), ("t_s1", "Stage 1"), ("t_s2", "Stage 2"))}


def parse_log(path):
    rows, current = {}, None
    path = Path(path)
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = FRAME_RE.search(line)
            if match:
                current = match.group(1)
                rows[current] = {}  # A retried frame replaces old partial entries.
            if current is not None:
                match = EVAL_RE.search(line)
                if match:
                    iteration, value = int(match.group(1)), float(match.group(3))
                    rows[current][f"psnr_it{iteration}"] = value
                    rows[current]["psnr_main"] = value
                    rows[current]["psnr_source"] = "upstream_float_log"
                    rows[current]["last_eval_iteration"] = iteration
                for key, expression in STAGES.items():
                    match = expression.search(line)
                    if match:
                        rows[current][key] = float(match.group(1))
            match = TIME_RE.search(line)
            if match:
                rows.setdefault(match.group(1), {})["t_frame"] = float(match.group(2))
    return rows


def ply_count(path):
    path = Path(path)
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            return None
        for _ in range(2000):
            line = handle.readline()
            if not line:
                return None
            if line.startswith(b"element vertex "):
                try:
                    return int(line.split()[-1])
                except ValueError:
                    return None
            if line.startswith(b"end_header"):
                return None
    return None


def valid_ply(path):
    path = Path(path)
    if not path.is_file():
        return False
    sizes = {"float": 4, "float32": 4, "double": 8, "float64": 8, "char": 1, "uchar": 1,
             "int8": 1, "uint8": 1, "short": 2, "ushort": 2, "int16": 2, "uint16": 2,
             "int": 4, "uint": 4, "int32": 4, "uint32": 4}
    count, stride, binary, vertex = None, 0, False, False
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            return False
        for _ in range(2000):
            line = handle.readline()
            if not line:
                return False
            words = line.split()
            if words[:1] == [b"format"]:
                binary = words[1].startswith(b"binary_")
            elif words[:1] == [b"element"]:
                vertex = words[1] == b"vertex"
                if vertex:
                    count = int(words[2])
            elif words[:1] == [b"property"] and vertex:
                kind = words[1].decode()
                if kind not in sizes:
                    return False
                stride += sizes[kind]
            elif words[:1] == [b"end_header"]:
                if count is None or count < 0 or stride == 0:
                    return False
                if binary:
                    return path.stat().st_size == handle.tell() + count * stride
                try:
                    from plyfile import PlyData
                    return len(PlyData.read(path)["vertex"]) == count
                except (OSError, ValueError):
                    return False
    return False


def frame_paths(run, frame, config=None):
    config = config or read_json(Path(run) / "cfg_run.json")
    first = int(config.get("iterations", 150))
    final = first + int(config.get("iterations_s2", 100))
    folder = Path(run) / "out" / f"frame{frame:06d}"
    return dict(folder=folder, png=folder / "0_rendering2.png", ntc=folder / "NTC.pth",
                added=folder / "point_cloud" / f"iteration_{final}" / "added/point_cloud.ply",
                full=folder / "point_cloud" / f"iteration_{first}" / "point_cloud.ply",
                final_iteration=final)


def audit_run(run, expected_frames=None, verify_images=False):
    run = Path(run)
    config_issue = None
    try:
        config = read_json(run / "cfg_run.json")
        if not isinstance(config, dict):
            raise ValueError("Invalid run configuration")
    except (OSError, ValueError):
        config, config_issue = {}, "cfg_run_invalid"
    expected_frames = expected_frames if expected_frames is not None else config.get("frame_end")
    if expected_frames is None or int(expected_frames) < 2:
        raise ValueError("Expected frame count must be known and >= 2")
    expected_frames = int(expected_frames)
    logs = parse_log(run / "train.log")
    records, issues = [], [config_issue] if config_issue else []
    initial = run / "init/point_cloud/iteration_15000/point_cloud.ply"
    initial_ok = valid_ply(initial)
    if not initial_ok:
        issues.append("initial_ply_missing_or_invalid")
    for frame in range(1, expected_frames):
        paths = frame_paths(run, frame, config)
        log = logs.get(f"frame{frame:06d}", {})
        problems = []
        if not paths["png"].is_file() or paths["png"].stat().st_size == 0:
            problems.append("render_missing")
        elif verify_images:
            try:
                from PIL import Image
                with Image.open(paths["png"]) as image:
                    image.verify()
            except (OSError, ValueError):
                problems.append("render_corrupt")
        if not finite_number(log.get("t_frame")):
            problems.append("completion_log_missing")
        for key in ("t_prep", "t_s1", "t_s2"):
            if not finite_number(log.get(key)) or float(log[key]) < 0:
                problems.append(key + "_missing_or_invalid")
        if not finite_number(log.get(f"psnr_it{paths['final_iteration']}")):
            problems.append("final_float_psnr_missing")
        for key in ("ntc", "added"):
            if not paths[key].is_file() or paths[key].stat().st_size == 0:
                problems.append(key + "_payload_missing")
        if paths["added"].is_file() and not valid_ply(paths["added"]):
            problems.append("added_ply_invalid")
        records.append(dict(frame=frame, complete=not problems, problems=problems, **log))
    unexpected = []
    if (run / "out").is_dir():
        for folder in (run / "out").iterdir():
            match = re.fullmatch(r"frame(\d+)", folder.name)
            if match and int(match.group(1)) not in range(1, expected_frames):
                unexpected.append(folder.name)
    if unexpected:
        issues.append("unexpected_frame_directories")
    incomplete = [row["frame"] for row in records if not row["complete"]]
    complete = initial_ok and not incomplete and not unexpected and not config_issue
    return dict(complete=complete, expected_frames=expected_frames,
                completed_stream_frames=sum(row["complete"] for row in records),
                missing_or_invalid_frames=incomplete, issues=issues, unexpected_frames=unexpected,
                frame_checks=records)
