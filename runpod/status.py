"""Print a small status snapshot without importing Torch or using the GPU."""
import json
import re
from pathlib import Path


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def tail(path, size=65536):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(max(0, stream.seek(0, 2) - size))
            return stream.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def main():
    root = Path("/workspace")
    manifest = read_json(root / "dgs_exp/jobs/camera15_cap1m_s0/experiment_manifest.json")
    stage_file = root / "pipeline.stage"
    stage = stage_file.read_text().strip() if stage_file.is_file() else "not_started"
    states = []
    for condition in manifest.get("conditions", []):
        run = root / "runs/stream" / condition["run"]
        status = read_json(run / "execution_status.json")
        state = status.get("state", "not_started")
        item = {"run": condition["run"], "state": state}
        if state == "running":
            log = tail(run / "train.log")
            finished = re.findall(r"Frame frame(\d+) finished in ([\d.]+) seconds", log)
            if finished:
                item.update(last_completed_frame=int(finished[-1][0]), last_frame_seconds=float(finished[-1][1]))
            item["log_tail"] = log.splitlines()[-2:]
        states.append(item)
    report = {"stage": stage, "conditions": len(states),
              "complete": sum(row["state"] == "complete" for row in states),
              "failed": [row["run"] for row in states if row["state"] == "failed"],
              "active": [row for row in states if row["state"] == "running"],
              "data_progress": tail(root / "data-prep.log").splitlines()[-2:],
              "pipeline_tail": tail(root / "pipeline.log").splitlines()[-3:]}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
