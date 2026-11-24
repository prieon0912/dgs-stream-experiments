#!/usr/bin/env bash
# One detached session owns the GPU for the complete 15-condition experiment.
set -eEuo pipefail
source "$(dirname "$0")/env.sh"
cd "$EXP"
JOB_DIR=${JOB_DIR:-$EXP/jobs/camera15_cap1m_s0}
MANIFEST=$JOB_DIR/experiment_manifest.json
RESULT_DIR=${RESULT_DIR:-$RUNS/camera_capped/results}
CURRENT_STAGE=starting

exec 9>"$WS/experiment.lock"
if ! flock -n 9; then
  echo "Another experiment driver is already running."
  exit 1
fi

stage() {
  CURRENT_STAGE=$1
  printf '%s\n' "$CURRENT_STAGE" > "$WS/pipeline.stage"
  echo "$(date -Is) stage=$CURRENT_STAGE"
}

on_exit() {
  local code=$?
  if [ "$code" -ne 0 ]; then
    printf 'failed: %s (exit %s)\n' "$CURRENT_STAGE" "$code" > "$WS/pipeline.stage"
    echo "Pipeline failed during $CURRENT_STAGE; exit=$code."
    # Keep unfinished conditions visible in the exported CSVs.
    conda activate gs || true
    python "$EXP/scripts/15_collect_results.py" --manifest "$MANIFEST" --out-dir "$RESULT_DIR" || true
    if [ -n "${AUTO_STOP_POD_ID:-}" ]; then
      echo "Failure records saved. Automatic STOP is scheduled in ${AUTO_STOP_DELAY_SECONDS:-600} seconds."
      sleep "${AUTO_STOP_DELAY_SECONDS:-600}"
      python "$EXP/runpod/stop_pod.py" --pod-id "$AUTO_STOP_POD_ID" || true
    fi
  fi
}
trap on_exit EXIT

if [ -n "${SMOKE_PID:-}" ]; then
  stage waiting_for_smoke
  while kill -0 "$SMOKE_PID" 2>/dev/null; do sleep 10; done
  if ! grep -q '^SMOKE_CAP1M_OK$' "$WS/smoke_cap1m.log"; then
    echo "The capped smoke test did not finish successfully."
    exit 1
  fi
fi
stage waiting_for_data
if [ -n "${DATA_PREP_PID:-}" ]; then
  while kill -0 "$DATA_PREP_PID" 2>/dev/null; do sleep 15; done
  if ! grep -q '^dataset OK:' "$WS/data-prep.log"; then
    echo "Data preparation ended without its success marker. Inspect data-prep.log."
    exit 1
  fi
fi

conda activate gs
python - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["SCENE"])
missing = []
for frame in range(300):
    folder = root / f"frame{frame:06d}"
    required = [folder / "images_2" / f"cam{cam:02d}.png" for cam in range(21)]
    required += [folder / "sparse" / "0" / name for name in ("cameras.bin", "images.bin", "points3D.bin")]
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(str(path))
if missing:
    raise SystemExit(f"Dataset incomplete: {len(missing)} missing/empty files. Examples: {missing[:8]}")
print("Dataset verified: 300 frames, 21 cameras per frame.", flush=True)
PY

stage preflight
bash "$JOB_DIR/preflight.sh"
stage initializations
bash "$JOB_DIR/run_initializations.sh"

stage streaming
stream_exit=0
# Run the existing queue in this tmux session, retaining its final exit status.
INSIDE_QUEUE=1 bash "$JOB_DIR/run_streams.sh" || stream_exit=$?

stage evaluation
evaluation_exit=0
bash "$JOB_DIR/evaluate.sh" || evaluation_exit=$?
if [ "$stream_exit" -ne 0 ] || [ "$evaluation_exit" -ne 0 ]; then
  echo "One or more conditions need review; inspect queue_status.csv and result CSVs."
  exit 1
fi

stage complete
echo "Results: $RESULT_DIR"
if [ -n "${AUTO_STOP_POD_ID:-}" ]; then
  echo "Automatic STOP is scheduled in ${AUTO_STOP_DELAY_SECONDS:-600} seconds; results remain in /workspace."
  sleep "${AUTO_STOP_DELAY_SECONDS:-600}"
  python "$EXP/runpod/stop_pod.py" --pod-id "$AUTO_STOP_POD_ID" || {
    echo "Automatic STOP was not confirmed. Stop the Pod through the Runpod console."
    printf 'complete; automatic stop unconfirmed\n' > "$WS/pipeline.stage"
  }
else
  echo "The Pod remains running. Download and verify results before stopping or terminating it."
fi
