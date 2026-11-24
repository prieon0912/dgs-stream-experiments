"""CPU-only consolidation. Missing results remain blank; planned conditions stay visible."""
import csv
from pathlib import Path

from common.aabb import aabb_id, file_sha256
from common.result_io import read_json as strict_read_json, write_json, write_csv, finite_number
from common.run_audit import audit_run
from common.provenance import environment_fingerprint


SUMMARY_FIELDS = [
    "run", "label", "group", "method", "init_model", "n_views", "camera_names", "seed", "refine_iters", "max_gaussians",
    "status", "complete", "quality_data_complete", "comparison_eligible", "time_reportable", "issues",
    "expected_frames", "completed_stream_frames", "gpu", "aabb_id", "aabb_verified",
    "aabb_min_x", "aabb_min_y", "aabb_min_z", "aabb_max_x", "aabb_max_y", "aabb_max_z",
    "experiment_code_sha256", "software_fingerprint",
    "raw_gaussians", "converted_gaussians", "final_gaussians", "aabb_inside_count", "aabb_inside_fraction",
    "pre_refine_psnr", "pre_refine_ssim", "pre_refine_lpips", "post_refine_psnr", "post_refine_ssim", "post_refine_lpips",
    "refine_psnr_gain", "init_psnr", "init_ssim", "init_lpips", "stream_psnr", "stream_psnr_png", "stream_ssim", "stream_lpips",
    "first30_psnr", "last30_psnr", "last30_minus_first30_psnr", "lpips_every", "lpips_sampled_frames",
    "da3_infer_s", "convert_s", "refine_s", "init_time_s", "ntc_warmup_s", "stage1_s_per_frame", "stage2_s_per_frame",
    "preparation_s_per_frame", "stream_train_s_per_frame", "stream_wall_s_per_frame", "stream_wall_total_s",
    "measured_pipeline_s", "init_mb", "payload_mb_per_frame", "storage_avg_mb", "stream_peak_device_memory_mb",
    "alignment_residual_rel", "alignment_pass", "voxel_size", "occupied_voxels", "mean_centers_per_voxel",
    "fraction_points_in_multi_view_voxels"]


def read_json(path):
    try:
        data = strict_read_json(path)
        return data if isinstance(data, dict) else {"_read_error": "Expected a JSON object", "path": str(path)}
    except (OSError, ValueError) as error:
        return {"_read_error": str(error), "path": str(path)}


def csv_rows(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for row in rows:
        converted = {}
        for key, value in row.items():
            converted[key] = float(value) if finite_number(value) else None if value == "" else value
        result.append(converted)
    return result


def resolver(mappings):
    mappings = sorted(mappings, key=lambda pair: len(pair[0]), reverse=True)
    def resolve(path):
        text = str(path)
        for old, new in mappings:
            if text == old or text.startswith(old.rstrip("/") + "/"):
                text = str(Path(new) / text[len(old):].lstrip("/"))
                break
        return Path(text)
    return resolve


def read_initialization(condition, resolve):
    record = read_json(resolve(condition["initialization_record"]))
    if record:
        return record
    ply = resolve(condition["final_ply"])
    timing = read_json(ply.with_name(ply.stem + "_timing.json"))
    return dict(timing=timing, init_metadata=timing.get("init_metadata", {}))


def collect(manifest_path, output, mappings=()):
    manifest = strict_read_json(manifest_path)
    conditions = manifest.get("conditions", [])
    if not conditions:
        raise ValueError("Experiment manifest contains no conditions")
    if len({condition["run"] for condition in conditions}) != len(conditions):
        raise ValueError("Duplicate runs in manifest")
    resolve = resolver(mappings)
    expected = int(manifest["expected_frames"])
    stream_root = resolve(manifest["stream_root"])
    expected_box = aabb_id(manifest["aabb"])
    summaries, initializations, per_frame, statuses, occupancy_rows, settings, profiles = [], [], [], [], [], [], []
    measured_gpu, measured_code, measured_software = {}, {}, {}
    for condition in conditions:
        name = condition["run"]
        run = stream_root / name
        record = read_initialization(condition, resolve)
        timing, metadata = record.get("timing", {}), record.get("init_metadata", {})
        count, overlap = record.get("counts", {}), record.get("occupancy", {})
        pre, final_quality = record.get("quality_pre_refine", {}), record.get("quality_final", {})
        summary = read_json(run / "run_summary.json")
        measured_time = summary.get("time", {})
        contract = read_json(run / "run_contract.json")
        environment = read_json(run / "environment_gs.json")
        if not environment:
            environment = read_json(resolve(condition["environment_gs"]))
        gpu = timing.get("gpu")
        if not gpu and environment.get("gpu"):
            gpu = environment["gpu"].splitlines()[0].split(",")[0].strip()
        code = environment.get("experiment_code_sha256")
        if gpu:
            measured_gpu[name] = gpu.strip()
        if code:
            measured_code[name] = code
            measured_software[name] = environment_fingerprint(environment)
        audit = audit_run(run, expected, verify_images=True)
        issues = list(audit["issues"])
        for field, data in (("initialization", record), ("summary", summary), ("contract", contract), ("environment", environment)):
            if "_read_error" in data:
                issues.append(field + "_json_invalid")
        if audit["missing_or_invalid_frames"]:
            issues.append("stream_incomplete")
        if not record.get("counts"):
            issues.append("initialization_record_missing")
        if contract.get("aabb_id") != expected_box:
            issues.append("aabb_unverified_or_mismatch")
        final_ply = resolve(condition["final_ply"])
        if contract and final_ply.is_file() and contract.get("init_sha256") != file_sha256(final_ply):
            issues.append("initial_ply_changed")
        if not code:
            issues.append("code_version_unverified")
        if not gpu:
            issues.append("gpu_unverified")
        if condition["kind"] == "da3":
            if metadata.get("alignment_pass") is not True:
                issues.append("alignment_not_approved")
            if metadata.get("max_gaussians") != manifest["max_gaussians"]:
                issues.append("gaussian_limit_policy_unverified_or_mismatch")
            if metadata.get("camera_names") != condition["camera_names"]:
                issues.append("camera_list_unverified_or_mismatch")
            if metadata.get("refine_iters") != condition["refine_iters"]:
                issues.append("refine_iterations_unverified_or_mismatch")
            if metadata.get("filter_policy") != manifest.get("filter_policy"):
                issues.append("filter_policy_unverified_or_mismatch")
            if metadata.get("process_res") != manifest.get("process_res"):
                issues.append("process_resolution_unverified_or_mismatch")
            if condition["refine_iters"] and not all(finite_number(quality.get(metric))
                    for quality in (pre, final_quality) for metric in ("psnr", "ssim", "lpips")):
                issues.append("before_after_quality_missing")
        cached_frames = csv_rows(run / "per_frame.csv")
        frame_ids = [int(row["frame"]) for row in cached_frames if finite_number(row.get("frame"))]
        interval = summary.get("lpips_every", 1)
        interval = interval if isinstance(interval, int) and interval >= 1 else 1
        table_ok = (len(cached_frames) == expected and len(frame_ids) == expected and
                    sorted(frame_ids) == list(range(expected)) and
                    all(all(finite_number(row.get(metric)) for metric in ("psnr_main", "psnr_png", "ssim"))
                        and row.get("psnr_source") == ("float_render" if row["frame"] == 0 else "upstream_float_log")
                        and (row["frame"] % interval != 0 or finite_number(row.get("lpips"))) for row in cached_frames))
        metric_ok = (summary.get("quality_data_complete") is True and
                     summary.get("completion", {}).get("expected_frames") == expected and table_ok and
                     summary.get("source_init_sha256") == contract.get("init_sha256"))
        if not metric_ok:
            issues.append("metrics_incomplete")
        execution = read_json(run / "execution_status.json")
        if execution.get("exit_code") not in (None, 0):
            issues.append("execution_failed")
        complete = audit["complete"]
        eligible = complete and metric_ok and not issues
        if not final_ply.is_file() and not run.exists():
            status = "not_started"
        elif execution.get("state") in ("alignment_failed", "skipped_no_init"):
            status = execution["state"]
        elif execution.get("exit_code") not in (None, 0):
            status = "failed"
        elif not complete:
            status = "init_ready" if final_ply.is_file() and not (run / "out").exists() else "stream_incomplete"
        elif not metric_ok:
            status = "awaiting_evaluation"
        else:
            status = "complete" if eligible else "needs_review"
        if metadata.get("alignment_residual_rel") is None and condition.get("initialization_record"):
            # A failed alignment saves no PLY metadata; report the residual from the alignment check.
            init_dir = resolve(condition["initialization_record"]).parent
            pre_ply = condition.get("pre_ply")
            candidates = [init_dir / "converted_alignment.json"]
            if pre_ply:
                candidates.append(resolve(pre_ply).parent / "converted_alignment.json")
            for candidate in candidates:
                report = read_json(candidate)
                if report.get("residual_median_rel") is not None:
                    metadata = dict(metadata, alignment_residual_rel=report["residual_median_rel"],
                                    alignment_pass=report.get("alignment_pass"))
                    break
        stream, storage = summary.get("stream", {}), summary.get("storage", {})
        frame0 = next((row for row in cached_frames if row.get("frame") == 0), {})
        steps = timing.get("steps", {})
        after = final_quality if condition["refine_iters"] else {}
        row = dict(run=name, label=condition["label"], group=condition["group"], method=condition["method"],
                   init_model=metadata.get("model", timing.get("da3_model")),
                   n_views=condition["n_views"], camera_names=condition["camera_names"], seed=condition["seed"],
                   refine_iters=condition["refine_iters"], max_gaussians=condition.get("max_gaussians"), status=status,
                   complete=complete, quality_data_complete=metric_ok, comparison_eligible=eligible,
                   time_reportable=summary.get("time_reportable") is True and eligible,
                   issues=issues, expected_frames=expected, completed_stream_frames=audit["completed_stream_frames"],
                   gpu=gpu, aabb_id=contract.get("aabb_id"), experiment_code_sha256=code,
                   aabb_verified=contract.get("aabb_id") == expected_box,
                   software_fingerprint=environment_fingerprint(environment),
                   raw_gaussians=metadata.get("n_raw_gaussians"), converted_gaussians=metadata.get("n_converted_gaussians"),
                   final_gaussians=count.get("n_total", metadata.get("n_final_gaussians")),
                   aabb_inside_count=count.get("n_inside"), aabb_inside_fraction=count.get("inside_fraction"),
                   pre_refine_psnr=pre.get("psnr"), pre_refine_ssim=pre.get("ssim"), pre_refine_lpips=pre.get("lpips"),
                   post_refine_psnr=after.get("psnr"), post_refine_ssim=after.get("ssim"), post_refine_lpips=after.get("lpips"),
                   refine_psnr_gain=after["psnr"] - pre["psnr"] if finite_number(after.get("psnr")) and finite_number(pre.get("psnr")) else None,
                   init_psnr=frame0.get("psnr_main", final_quality.get("psnr")),
                   init_ssim=frame0.get("ssim", final_quality.get("ssim")),
                   init_lpips=frame0.get("lpips", final_quality.get("lpips")),
                   stream_psnr=stream.get("psnr"), stream_psnr_png=stream.get("psnr_png"),
                   stream_ssim=stream.get("ssim"), stream_lpips=stream.get("lpips"),
                   first30_psnr=summary.get("first30_psnr"), last30_psnr=summary.get("last30_psnr"),
                   last30_minus_first30_psnr=summary.get("drift_psnr"), lpips_every=summary.get("lpips_every"),
                   lpips_sampled_frames=stream.get("lpips_sampled_frames"), da3_infer_s=steps.get("da3_infer"),
                   convert_s=steps.get("convert"), refine_s=steps.get(f"refine_{condition['refine_iters']}"),
                   init_time_s=timing.get("init_time_sec"), ntc_warmup_s=measured_time.get("ntc_warmup_sec"),
                   stage1_s_per_frame=measured_time.get("stage1_sec_per_frame"),
                   stage2_s_per_frame=measured_time.get("stage2_sec_per_frame"),
                   preparation_s_per_frame=measured_time.get("stream_prep_sec_per_frame"),
                   stream_train_s_per_frame=measured_time.get("stream_train_sec_per_frame"),
                   stream_wall_s_per_frame=measured_time.get("stream_wall_sec_per_frame"),
                   stream_wall_total_s=measured_time.get("stream_wall_sec_total"),
                   measured_pipeline_s=measured_time.get("measured_pipeline_sec"),
                   init_mb=storage.get("init_mb"), payload_mb_per_frame=storage.get("per_frame_payload_mb"),
                   storage_avg_mb=storage.get("paper_avg_mb"), stream_peak_device_memory_mb=summary.get("peak_gpu_mem_mb"),
                   alignment_residual_rel=metadata.get("alignment_residual_rel"), alignment_pass=metadata.get("alignment_pass"),
                   voxel_size=overlap.get("voxel_size"), occupied_voxels=overlap.get("occupied_voxels"),
                   mean_centers_per_voxel=overlap.get("mean_centers_per_occupied_voxel"),
                   fraction_points_in_multi_view_voxels=overlap.get("fraction_points_in_multi_view_voxels"))
        actual_bounds = contract.get("aabb", {})
        for edge in ("min", "max"):
            for index, axis in enumerate("xyz"):
                row[f"aabb_{edge}_{axis}"] = actual_bounds.get(edge, [None] * 3)[index]
        summaries.append(row)
        initializations.append({key: row.get(key) for key in SUMMARY_FIELDS if key not in
                                ("stream_psnr", "stream_psnr_png", "stream_ssim", "stream_lpips")})
        occupancy_rows.append(dict(run=name, **overlap))
        settings.append({**condition, "actual_model": row["init_model"], "actual_metadata": metadata, "actual_environment": environment,
                         "expected_aabb": manifest["aabb"], "actual_aabb": contract.get("aabb")})
        checks = {row["frame"]: row for row in audit["frame_checks"]}
        measured = {int(row["frame"]): row for row in cached_frames if finite_number(row.get("frame"))}
        for frame in range(expected):
            values = measured.get(frame, {})
            check = checks.get(frame, {})
            per_frame.append(dict(run=name, frame=frame, frame_complete=bool(check.get("complete")) if frame else bool(frame0),
                                  frame_issues=check.get("problems", []) if frame else [] if frame0 else ["frame0_metrics_missing"],
                                  psnr_float=values.get("psnr_main"), psnr_source=values.get("psnr_source", "missing"),
                                  psnr_png=values.get("psnr_png"), ssim=values.get("ssim"), lpips=values.get("lpips"),
                                  n_points=values.get("n_points"), n_added=values.get("n_added"),
                                  preparation_s=values.get("t_prep", check.get("t_prep")),
                                  stage1_s=values.get("t_s1", check.get("t_s1")),
                                  stage2_s=values.get("t_s2", check.get("t_s2")),
                                  frame_wall_s=values.get("t_frame", check.get("t_frame")),
                                  payload_mb=values.get("storage_mb")))
        statuses.append(dict(run=name, status=status, complete=complete, comparison_eligible=eligible,
                             expected_frames=expected, completed_stream_frames=audit["completed_stream_frames"],
                             missing_or_invalid_frames=audit["missing_or_invalid_frames"], issues=issues,
                             execution_stage=execution.get("stage"), exit_code=execution.get("exit_code")))
        profiles.extend(dict(run=name, **event) for event in csv_rows(run / "diagnostics/profile_events.csv"))
    # Reject cross-GPU / cross-code formal comparisons even when individual runs are valid.
    for row, status in zip(summaries, statuses):
        for measured, reason in ((measured_gpu, "mixed_gpu_models"), (measured_code, "mixed_experiment_code"),
                                 (measured_software, "mixed_software_environment")):
            if len(set(measured.values())) > 1 and row["run"] in measured:
                row["issues"].append(reason)
                row["comparison_eligible"] = row["time_reportable"] = False
                status["comparison_eligible"] = False
                if row["status"] == "complete":
                    row["status"] = status["status"] = "needs_review"
    output = Path(output)
    initializations = [dict(row) for row in summaries]
    write_csv(output / "summary.csv", summaries, SUMMARY_FIELDS)
    valid = [dict(row) for row in summaries if row["comparison_eligible"]]
    time_columns = [key for key in SUMMARY_FIELDS if key.endswith("_s") or key.endswith("_s_per_frame")]
    for row in valid:
        if not row["time_reportable"]:
            for key in time_columns:
                row[key] = None
    write_csv(output / "summary_valid.csv", valid, SUMMARY_FIELDS)
    init_fields = ["run", "label", "method", "init_model", "n_views", "camera_names", "seed", "refine_iters", "max_gaussians",
                   "status", "raw_gaussians", "converted_gaussians", "final_gaussians", "aabb_id",
                   "aabb_inside_count", "aabb_inside_fraction", "pre_refine_psnr", "pre_refine_ssim", "pre_refine_lpips",
                   "post_refine_psnr", "post_refine_ssim", "post_refine_lpips", "refine_psnr_gain",
                   "init_psnr", "init_ssim", "init_lpips", "da3_infer_s", "convert_s", "refine_s", "init_time_s",
                   "alignment_residual_rel", "alignment_pass"]
    write_csv(output / "initialization.csv", initializations, init_fields)
    write_csv(output / "per_frame.csv", per_frame)
    write_csv(output / "run_status.csv", statuses)
    write_csv(output / "spatial_diagnostics.csv", occupancy_rows,
              ["run", "voxel_size", "n_points", "occupied_voxels", "mean_centers_per_occupied_voxel",
               "p95_centers_per_occupied_voxel", "fraction_points_in_multi_center_voxels", "n_centers_inside_aabb",
               "multi_view_voxels", "fraction_points_in_multi_view_voxels", "diagnostic_scope"])
    flat_settings = []
    for setting in settings:
        environment = setting["actual_environment"]
        flat = {key: value for key, value in setting.items()
                if key not in ("actual_environment", "actual_metadata", "expected_aabb", "actual_aabb")}
        flat.update(expected_frames=expected, process_res=manifest.get("process_res"),
                    training_cameras=manifest.get("training_cameras"), filter_policy=manifest.get("filter_policy"),
                    expected_aabb_id=expected_box, actual_aabb_id=aabb_id(setting["actual_aabb"]) if setting["actual_aabb"] else None,
                    python=environment.get("python", "").splitlines()[0] if environment.get("python") else None,
                    torch_version=environment.get("packages", {}).get("torch"),
                    tinycudann_version=environment.get("packages", {}).get("tinycudann"),
                    gs_commit=environment.get("repositories", {}).get("3DGStream", {}).get("commit"),
                    da3_commit=environment.get("repositories", {}).get("Depth-Anything-3", {}).get("commit"))
        for edge in ("min", "max"):
            for index, axis in enumerate("xyz"):
                flat[f"planned_aabb_{edge}_{axis}"] = manifest["aabb"][edge][index]
        flat_settings.append(flat)
    write_csv(output / "experiment_conditions.csv", flat_settings)
    write_json(output / "experiment_conditions.json", settings)
    write_csv(output / "profile_components.csv", profiles,
              ["run", "frame", "component", "grad_enabled", "gpu_elapsed_ms", "wall_ms", "diagnostic_only"])
    write_json(output / "collection.json", dict(expected_conditions=len(conditions),
               completed_conditions=sum(row["complete"] for row in summaries),
               comparable_conditions=len(valid), measurements_present=bool(valid),
               note="Blank fields are unmeasured/unavailable, not zero. Occupancy is a center-overlap proxy. "
                    "measured_pipeline_s excludes calibration, weight loading, evaluation and diagnostics."))
    write_csv(output / "columns.csv", [dict(column=key, explanation=column_description(key)) for key in SUMMARY_FIELDS])
    return summaries


def column_description(key):
    special = {
        "status": "실행 상태: not_started/init_ready/stream_incomplete/awaiting_evaluation/failed/needs_review/complete",
        "comparison_eligible": "완료·지표·AABB·카메라·정책·GPU·실험 코드 확인을 통과한 조건",
        "time_reportable": "필요 시간값이 모두 있고 초기화·warm-up·스트리밍의 GPU 단독 사용이 확인됨",
        "camera_names": "DA3 입력 카메라 목록; 3DGS 기준 조건은 학습 카메라 목록",
        "raw_gaussians": "DA3 원시 예측 개수, 필터링 전",
        "converted_gaussians": "좌표 변환·공통 필터 후 개수, 개수 상한 적용 전",
        "final_gaussians": "실제 스트리밍 초기화에 투입한 개수",
        "aabb_inside_fraction": "공통 AABB 안에 중심이 있는 비율; 실제 움직임 비율이 아님",
        "stream_psnr": "최종 학습 단계 로그의 float PSNR, 프레임 1..N-1 평균; PNG로 대체하지 않음",
        "stream_psnr_png": "8-bit 렌더링 PNG의 PSNR, 프레임 1..N-1 평균",
        "stream_ssim": "8-bit 렌더링의 SSIM 평균",
        "stream_lpips": "8-bit 렌더링의 LPIPS-vgg 평균; 표본 수/간격을 함께 확인",
        "measured_pipeline_s": "기록된 초기화 + NTC warm-up + 스트리밍 wall 합, 초; 실제 통화 지연이 아님",
        "stream_peak_device_memory_mb": "스트리밍 중 2초 간격 GPU 전체 메모리 표본 최댓값, MB; 순간 피크를 놓칠 수 있음",
        "alignment_pass": "경험적 정렬 검토 기준 통과 여부; 영상 품질 보장이 아님",
        "fraction_points_in_multi_view_voxels": "여러 뷰 출신 중심이 같은 복셀에 있는 점 비율; 불필요한 중복의 증명은 아님",
        "issues": "검토/제외 사유 목록",
    }
    if key in special:
        return special[key]
    if "psnr" in key:
        return "PSNR, dB; pre/post는 1,000회 보정 전/후, gain은 후-전"
    if "ssim" in key or "lpips" in key:
        return "영상 품질 지표, 무차원; 측정되지 않으면 빈칸"
    if key.endswith("_s") or key.endswith("_s_per_frame"):
        return "시간, 초 또는 초/프레임; 미측정은 빈칸"
    return "실험 설정 또는 측정값; 미측정은 빈칸, 세부 정의는 README_RESULTS.md 참고"
