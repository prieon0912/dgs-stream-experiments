"""Init-time bookkeeping shared by every script that produces an init PLY.

Each init PLY  X.ply  gets a sidecar  X_timing.json:
  {"steps": {"da3_infer": 41.2, "convert": 6.3, "refine_1k": 95.0},   # seconds, in pipeline order
   "init_time_sec": 142.5,                                          # sum of steps (None if unknown)
   "gpu": "NVIDIA A100 80GB", "gpu_exclusive": true, "notes": [...]}

Derived inits (subsample, refine) inherit the steps of their source, so the final number always
answers "how long from frame-0 images (+ known poses) to this init".

Timing protocol (same for every init):
  * COLMAP / poses are excluded for all inits (both pipelines start from the same known poses)
  * model weight loading and python/CUDA start-up are recorded separately, not in init_time_sec
    (exception: baseline 3DGS wall clock includes ~5-10 s start-up; noted in its json)
  * measured with the GPU otherwise idle -> gpu_exclusive must be true for numbers you report
"""
import json
import os
import subprocess
import time
from common.result_io import tri_all


def timing_path(ply):
    return os.path.splitext(ply)[0] + "_timing.json"


def gpu_name():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip().splitlines()[0]
    except Exception:
        return None


def gpu_exclusive():
    """True if the GPU is idle right now (call it just before the timed section, while this
    process is not computing). Inside RunPod containers nvidia-smi usually cannot list other
    processes' PIDs, so we also sample utilization."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.split()
        others = [p for p in out if p.strip().isdigit() and int(p) != os.getpid()]
        if others:
            return False
        # CUDA/model loading or a pre-check render in this very process can leave
        # a stale busy utilization sample. Wait for three quiet samples first.
        util = []
        for _ in range(10):
            u = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True).stdout.split()
            util.append(int(u[0]))
            if len(util) >= 3 and max(util[-3:]) <= 5:
                return True
            time.sleep(0.5)
        return False
    except Exception:
        return None


def load(ply):
    p = timing_path(ply)
    if os.path.exists(p):
        return json.load(open(p))
    return dict(steps={}, init_time_sec=None, notes=["no timing recorded for source"], gpu=None, gpu_exclusive=None)


def write(out_ply, step, sec, src_ply=None, exclusive=None, notes=(), measurement_required=False, **extra):
    t = load(src_ply) if src_ply else dict(steps={}, notes=[], gpu=gpu_name(), gpu_exclusive=None)
    t = dict(t)
    t["steps"] = dict(t.get("steps", {}))
    if step is not None:
        t["steps"][step] = float(sec)
    known = t.get("init_time_sec", 0) is not None or not src_ply
    t["init_time_sec"] = float(sum(t["steps"].values())) if known else None
    if exclusive is not None or measurement_required:
        t["gpu_exclusive"] = tri_all([t.get("gpu_exclusive"), exclusive]) if src_ply else exclusive
    t["gpu"] = t.get("gpu") or gpu_name()
    t["notes"] = list(t.get("notes", [])) + list(notes)
    t.update(extra)
    with open(timing_path(out_ply), "w") as f:
        json.dump(t, f, indent=1)
    if t.get("gpu_exclusive") is False:
        print("WARNING: GPU was shared during a timed step -> do not report this init time")
    return t
