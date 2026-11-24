"""Generate a reproducible 15-condition experiment, staged execution and CSV export."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.aabb import load_aabb, save_aabb
from common.camera_selection import save_json_checked, validate_view_names


def quote(value):
    return shlex.quote(str(value))


def write_checked(path, text):
    path = Path(path)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"Different job already exists: {path}; use a new output directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def generate_jobs(plan, aabb, output, tag="camera15", server_root="/workspace/runs/camera_experiment",
                  frame_dir="/workspace/data/flame_steak/frame000000", refine_iters=1000,
                  max_gaussians=0, expected_frames=300, include_baselines=True):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", tag):
        raise ValueError("Invalid tag")
    if refine_iters < 1 or max_gaussians < 0 or expected_frames < 2:
        raise ValueError("Invalid refinement count, Gaussian limit or frame count")
    rows = plan.get("conditions", [])
    if not rows:
        raise ValueError("Selection plan has no conditions")
    seen = set()
    for row in rows:
        identifier = row["condition_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier) or identifier in seen:
            raise ValueError("Invalid or duplicate condition ID")
        seen.add(identifier)
        validate_view_names(row["camera_names"], ["cam00"] + row.get("excluded_cameras", []))
        if len(row["camera_names"]) != row["n_views"]:
            raise ValueError("Camera count mismatch")
    root = PurePosixPath(server_root)
    if not root.is_absolute() or any(character.isspace() for character in str(root)):
        raise ValueError("server_root must be an absolute Linux path without whitespace")
    output = Path(output)
    settings = dict(selection_plan=plan, aabb=aabb, tag=tag, server_root=str(root), frame_dir=frame_dir,
                    refine_iters=refine_iters, max_gaussians=max_gaussians, expected_frames=expected_frames,
                    include_baselines=include_baselines)
    save_json_checked(output / "job_settings.json", settings)
    save_aabb(output / "aabb.json", aabb)
    all_views = [plan["anchor"]] + sorted(name for name in plan["eligible_cameras"] if name != plan["anchor"])
    all_view_config = dict(condition_id="all_views", method="all_views", n_views=len(all_views),
                           camera_names=all_views, seed=plan["seed"], anchor=plan["anchor"],
                           excluded_cameras=plan["excluded_cameras"])
    save_json_checked(output / "views/all_views.json", all_view_config)
    for row in rows:
        save_json_checked(output / "views" / (row["condition_id"] + ".json"), row)
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    header = ["#!/usr/bin/env bash", "set -eEuo pipefail",
              'JOB_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)',
              'source "${EXP:-/workspace/dgs_exp}/runpod/env.sh"',
              'export AABB_CONFIG="$JOB_DIR/aabb.json"', f"export N_FRAMES={expected_frames}",
              'cd "$EXP"', f"# settings_sha256: {signature}"]
    commands = header + [
        'CURRENT_RUN=""',
        'on_failure() {', '  local code=$?',
        '  if [ -n "$CURRENT_RUN" ]; then',
        '    python "$EXP/scripts/19_execution_status.py" record --run-dir "$RUNS/stream/$CURRENT_RUN" '
        '--stage initialization --state failed --exit-code "$code" || true',
        '  fi', '  exit "$code"', '}', 'trap on_failure ERR',
        'conda activate gs',
        'python "$EXP/scripts/16_snapshot_environment.py" --out "$JOB_DIR/environment_gs.json" '
        '--lock-out "$JOB_DIR/versions.lock.json"',
        'conda activate da3',
        'python "$EXP/scripts/16_snapshot_environment.py" --out "$JOB_DIR/environment_da3.json"',
        'conda activate gs', '']

    def stage(marker, outputs, command):
        return ('python "$EXP/scripts/18_run_stage.py" --marker ' + quote(marker) + ' --signature ' + quote(signature) +
                ' --environment "$JOB_DIR/environment_gs.json" --environment "$JOB_DIR/environment_da3.json" --outputs ' +
                " ".join(quote(path) for path in outputs) + " -- " + command)

    conditions = []
    stream_root = "/workspace/runs/stream"

    def add_condition(suffix, label, kind, group, method, views, final, pre=None, iterations=0):
        name = tag + "_" + suffix
        folder = root / name
        condition = dict(run=name, label=label, kind=kind, group=group, method=method, n_views=len(views),
                         camera_names=views, seed=plan["seed"], refine_iters=iterations,
                         max_gaussians=max_gaussians if kind == "da3" else None,
                         final_ply=str(final), pre_ply=str(pre) if pre else None,
                         initialization_record=str(folder / "initialization.json"),
                         environment_gs=str(folder / "environment_gs.json"),
                         environment_da3=str(folder / "environment_da3.json"))
        conditions.append(condition)
        return condition, folder

    def begin(condition, folder):
        commands.extend([f'CURRENT_RUN={quote(condition["run"])}', f"mkdir -p {quote(folder)}",
                         'python "$EXP/scripts/19_execution_status.py" record --run-dir "$RUNS/stream/$CURRENT_RUN" '
                         '--stage initialization --state running', "conda activate gs"])

    def record(condition, folder):
        static = folder / "static"
        command = ('python "$EXP/scripts/04_check_and_eval.py" --frame_dir ' + quote(frame_dir) +
                   ' --aabb "$AABB_CONFIG" --out ' + quote(static) +
                   ' --ply ' + quote("final=" + condition["final_ply"]))
        if condition["pre_ply"]:
            command += " --ply " + quote("pre=" + condition["pre_ply"])
        commands.append(stage(folder / "quality.done.json", [static / "static.json", static / "static.csv"], command))
        command = ('python "$EXP/scripts/17_record_initialization.py" --run ' + quote(condition["run"]) +
                   ' --ply ' + quote(condition["final_ply"]) + ' --aabb "$AABB_CONFIG" --static-json ' +
                   quote(static / "static.json") + ' --out ' + quote(folder / "initialization.json"))
        if condition["pre_ply"]:
            command += " --pre-ply " + quote(condition["pre_ply"])
        commands.extend([stage(folder / "record.done.json", [folder / "initialization.json", folder / "initialization.csv"], command),
                     f'cp "$JOB_DIR/environment_gs.json" {quote(folder / "environment_gs.json")}',
                     f'cp "$JOB_DIR/environment_da3.json" {quote(folder / "environment_da3.json")}',
                     'python "$EXP/scripts/19_execution_status.py" record --run-dir "$RUNS/stream/$CURRENT_RUN" '
                     '--stage initialization --state init_ready --exit-code 0', ''])

    def da3_source(folder, view_id):
        raw, converted = folder / "raw.pt", folder / "converted.ply"
        commands.extend([f"mkdir -p {quote(folder)}", "conda activate da3"])
        command = ('python "$EXP/scripts/02_run_da3.py" --frame_dir ' + quote(frame_dir) +
                   f' --views-file "$JOB_DIR/views/{view_id}.json" --process_res 504 --out ' + quote(raw))
        commands.append(stage(folder / "infer.done.json", [raw, folder / "raw_meta.json"], command))
        commands.append("conda activate gs")
        command = ('python "$EXP/scripts/03_convert_da3.py" --raw ' + quote(raw) + ' --out ' + quote(converted) +
                   ' --border 0.03125 --depth_prune 1.0 --min_opacity 0.005 --sh1 keep '
                   '--max-align-residual 0.02 --save-sources')
        commands.append(stage(folder / "convert.done.json", [converted, folder / "converted_timing.json",
                              folder / "converted_convert.json", folder / "converted_sources.npz"], command))
        if not max_gaussians:
            return converted
        limited = folder / "limited.ply"
        command = ('python "$EXP/scripts/06_subsample.py" --ply ' + quote(converted) + ' --out ' +
                   quote(limited) + f" --topk {max_gaussians}")
        commands.append(stage(folder / "limit.done.json", [limited, folder / "limited_timing.json",
                                                           folder / "limited_sources.npz"], command))
        return limited

    def refine(condition, folder):
        command = ('python "$EXP/scripts/05_refine.py" --ply ' + quote(condition["pre_ply"]) + ' --out ' +
                   quote(condition["final_ply"]) + ' --frame_dir ' + quote(frame_dir) +
                   f" --iters {refine_iters} --seed {plan['seed']}")
        final = PurePosixPath(condition["final_ply"])
        commands.append(stage(folder / "refine.done.json", [condition["final_ply"],
                              str(final.with_name(final.stem + "_timing.json")),
                              str(final.with_name(final.stem + "_refine.json")),
                              str(final.with_name(final.stem + "_sources.npz"))], command))

    if include_baselines:
        base_name = tag + "_baseline_own"
        base_ply = PurePosixPath("/workspace/runs/inits") / (base_name + ".ply")
        condition, folder = add_condition("baseline_own", "3DGS 15k", "3dgs", "baseline", "3dgs", all_views, base_ply)
        begin(condition, folder)
        commands.append(stage(folder / "train.done.json", [base_ply, base_ply.with_name(base_name + "_timing.json")],
                        f'bash "$EXP/scripts/00_train_baseline.sh" 15000 {quote(base_name)}'))
        record(condition, folder)
        shared = root / (tag + "_shared_all_views")
        commands.append('CURRENT_RUN=' + quote(tag + "_da3_raw"))
        selected = da3_source(shared, "all_views")
        condition, folder = add_condition("da3_raw", "DA3 all views", "da3", "baseline", "all_views", all_views, selected)
        begin(condition, folder)
        record(condition, folder)
        refined_folder = root / (tag + "_da3_1k")
        condition, folder = add_condition("da3_1k", "DA3+1k all views", "da3", "baseline", "all_views",
                                          all_views, refined_folder / "refined.ply", selected, refine_iters)
        begin(condition, folder)
        refine(condition, folder)
        record(condition, folder)
    for row in rows:
        suffix = row["condition_id"] + f"_da3_{refine_iters}"
        folder = root / (tag + "_" + suffix)
        final = folder / "refined.ply"
        pre = folder / ("limited.ply" if max_gaussians else "converted.ply")
        condition, folder = add_condition(suffix, row["condition_id"] + "+1k", "da3", "camera_selection",
                                          row["method"], row["camera_names"], final, pre, refine_iters)
        begin(condition, folder)
        da3_source(folder, row["condition_id"])
        refine(condition, folder)
        record(condition, folder)
    manifest = dict(schema_version=2, expected_frames=expected_frames, test_camera="cam00", stream_root=stream_root,
                    aabb=aabb, process_res=504, max_gaussians=max_gaussians,
                    filter_policy=dict(border=.03125, depth_prune=1., min_opacity=.005, sh1="keep"),
                    training_cameras=all_views, seed=plan["seed"], conditions=conditions)
    save_json_checked(output / "experiment_manifest.json", manifest)
    write_checked(output / "run_initializations.sh", "\n".join(commands) + "\n")
    queue = [f"{condition['run']} {condition['final_ply']}" for condition in conditions]
    write_checked(output / "queue.txt", "\n".join(queue) + "\n")
    stream = header + ['bash "$EXP/scripts/run_queue.sh" "$JOB_DIR/queue.txt"']
    write_checked(output / "run_streams.sh", "\n".join(stream) + "\n")
    names = " ".join(quote(condition["run"]) for condition in conditions)
    evaluate = header + ["conda activate gs", "evaluation_failed=0",
                         'python "$EXP/scripts/09_eval_stream.py" --runs ' + names +
                         f" --expected-frames {expected_frames} --lpips_every 1 || evaluation_failed=1",
                         'python "$EXP/scripts/15_collect_results.py" --manifest "$JOB_DIR/experiment_manifest.json" '
                         '--out-dir ' + quote(root / "results") + ' --require-complete || evaluation_failed=1',
                         'if [ "$evaluation_failed" -eq 0 ]; then',
                         '  python "$EXP/scripts/10_report.py" --out ' + quote(root / "results/figures") + " --runs " +
                         " ".join(quote(condition["run"] + ":" + condition["label"]) for condition in conditions),
                         "else", '  echo "Some runs need review. CSVs were exported with explicit status."', "fi",
                         'exit "$evaluation_failed"']
    write_checked(output / "evaluate.sh", "\n".join(evaluate) + "\n")
    preflight = header + ["conda activate gs"]
    identifiers = (["all_views"] if include_baselines else []) + [row["condition_id"] for row in rows]
    for identifier in identifiers:
        preflight.append('python "$EXP/scripts/02_run_da3.py" --frame_dir ' + quote(frame_dir) +
                         f' --views-file "$JOB_DIR/views/{identifier}.json" --out /tmp/{tag}_{identifier}.pt --dry-run')
    write_checked(output / "preflight.sh", "\n".join(preflight) + "\n")
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--aabb", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tag", default="camera15")
    parser.add_argument("--server-root", default="/workspace/runs/camera_experiment")
    parser.add_argument("--frame-dir", default="/workspace/data/flame_steak/frame000000")
    parser.add_argument("--refine-iters", type=int, default=1000)
    parser.add_argument("--max-gaussians", type=int, default=0, help="One common policy for all DA3 conditions; 0=no cap")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--only-camera-conditions", action="store_true", help="Omit the three baseline conditions")
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8-sig"))
    generate_jobs(plan, load_aabb(args.aabb), args.out_dir, args.tag, args.server_root, args.frame_dir,
                  args.refine_iters, args.max_gaussians, args.frames, not args.only_camera_conditions)
    count = len(plan["conditions"]) + (0 if args.only_camera_conditions else 3)
    print(f"Generated {count} conditions and CSV-export commands in {Path(args.out_dir).resolve()}; no GPU work started.")


if __name__ == "__main__":
    main()
