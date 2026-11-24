"""Optional short diagnostic replay. CUDA synchronization changes speed: NEVER use as formal timing."""
import argparse
import functools
import os
from pathlib import Path
import runpy
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.result_io import read_json, write_json, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Existing formal run; init, cfg and NTC are read-only")
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--scene", default=os.environ.get("SCENE", "/workspace/data/flame_steak"))
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    source = Path(args.run_dir).resolve()
    diagnostic = source / "diagnostics"
    if diagnostic.exists():
        raise FileExistsError("Diagnostics already exist. Preserve them and choose a separate copy of the source run to replay again.")
    config = read_json(source / "cfg_run.json")
    if not config:
        raise FileNotFoundError("Run cfg_run.json is required")
    init = source / "init"
    ntc = source / "ntc.pth"
    if not ntc.is_file():
        ntc = Path(config["ntc_path"])
    if not ntc.is_file() or not (init / "point_cloud/iteration_15000/point_cloud.ply").is_file():
        raise FileNotFoundError("Source init or NTC is missing")
    config.update(frame_start=1, frame_end=args.frames + 1, ntc_path=str(ntc), quiet=False)
    diagnostic.mkdir(parents=True)
    write_json(diagnostic / "cfg_profile.json", config)
    write_json(diagnostic / "scope.json", dict(diagnostic_only=True, source_run=str(source), frames=args.frames,
               note="Replay with per-operation CUDA synchronization; changes total latency; excluded from formal timing."))
    repository = Path(os.environ.get("REPO_GS", "/workspace/3DGStream")).resolve()
    sys.path.insert(0, str(repository))
    import torch
    import gaussian_renderer
    from scene import Scene
    from scene.gaussian_model import GaussianModel
    current = {"frame": None}
    events = []

    def timed(label, function, gpu=True):
        @functools.wraps(function)
        def wrapper(*items, **keywords):
            if gpu:
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
            wall = time.perf_counter()
            value = function(*items, **keywords)
            if gpu:
                end.record()
                torch.cuda.synchronize()
            events.append(dict(frame=current["frame"], component=label,
                               grad_enabled=torch.is_grad_enabled(), gpu_elapsed_ms=start.elapsed_time(end) if gpu else None,
                               wall_ms=(time.perf_counter() - wall) * 1000, diagnostic_only=True))
            return value
        return wrapper

    original_scene_init = Scene.__init__
    def scene_init(self, dataset, *items, **keywords):
        current["frame"] = Path(dataset.source_path).name
        return original_scene_init(self, dataset, *items, **keywords)
    Scene.__init__ = scene_init
    GaussianModel.query_ntc = timed("ntc_forward", GaussianModel.query_ntc)
    gaussian_renderer.render = timed("render", gaussian_renderer.render)
    torch.autograd.backward = timed("backward", torch.autograd.backward)
    Scene.save = timed("save_ply", Scene.save, gpu=False)
    Scene.dump_NTC = timed("save_ntc", Scene.dump_NTC, gpu=False)
    sys.argv = [str(repository / "train_frames.py"), "--read_config", "--config_path", str(diagnostic / "cfg_profile.json"),
                "-o", str(diagnostic / "out"), "-m", str(init), "-v", args.scene,
                "--image", "images_2", "--first_load_iteration", "15000"]
    try:
        runpy.run_path(str(repository / "train_frames.py"), run_name="__main__")
    finally:
        write_csv(diagnostic / "profile_events.csv", events,
                  ["frame", "component", "grad_enabled", "gpu_elapsed_ms", "wall_ms", "diagnostic_only"])
        print(f"Diagnostic events saved in {diagnostic}; excluded from formal timing")


if __name__ == "__main__":
    main()
