#!/usr/bin/env bash
# Driver for the pruning x update-mode experiment on a NEW pod (run inside tmux).
#
#   bash runpod/run_prune_update.sh                 # setup -> source -> phase 1 (N=1M), then stop for review
#   (after writing phase2.env)  bash runpod/run_prune_update.sh   # pilot on the chosen N -> 12 runs -> evaluation
#   AUTO_PHASE2=1 bash runpod/run_prune_update.sh   # no review stop: take the suggested N, pilot, 12 runs, evaluation
#   STOP_POD=1 ...                                  # stop (not terminate) this pod when the driver finishes successfully
#
# Every step is resumable; re-running the driver skips finished work.
set -eEuo pipefail
source "$(dirname "$0")/env.sh"
JOB=$EXP/jobs/prune_update_s0
source "$JOB/config.sh"
cd "$EXP"
mkdir -p "$ROOT"
exec 9>"$WS/prune_update.lock"
flock -n 9 || { echo "another driver is running"; exit 1; }
stage() { echo "$1" > "$WS/prune_update.stage"; echo "$(date -Is) stage=$1"; }
trap 'echo "failed: $(cat "$WS/prune_update.stage" 2>/dev/null)" > "$WS/prune_update.stage"' ERR

stage setup
command -v tmux >/dev/null && command -v ffmpeg >/dev/null || bash "$EXP/runpod/00_setup_pod.sh"
[ -x "$CONDA_ROOT/envs/gs/bin/python" ] || bash "$EXP/runpod/01_install_gs.sh"
[ -d "$SCENE/frame000299" ] || bash "$EXP/runpod/03_get_data.sh"

stage source
if [ ! -f "$SRC_PLY" ] && [ ! -x "$CONDA_ROOT/envs/da3/bin/python" ]; then bash "$EXP/runpod/02_install_da3.sh"; fi
bash "$JOB/00_prepare_source.sh"

stage phase1
bash "$JOB/10_phase1_screen.sh"

if [ ! -f "$PHASE2_ENV" ]; then
  if [ "${AUTO_PHASE2:-0}" = "1" ]; then
    grep -E '^N_' "$PHASE2_ENV.suggested" > "$PHASE2_ENV" || true
    for m in $METHODS; do grep -q "^N_$m=" "$PHASE2_ENV" || { echo "AUTO_PHASE2: no passing N for $m"; rm -f "$PHASE2_ENV"; exit 1; }; done
    echo "AUTO_PHASE2: using $(tr '\n' ' ' < "$PHASE2_ENV")"
  else
    stage review
    echo "Review $RESULTS/phase1.csv. Try other counts with jobs/prune_update_s0/10_phase1_screen.sh --n ...;"
    echo "when done, write N_<method>= lines to $PHASE2_ENV (see $PHASE2_ENV.suggested) and re-run this driver."
    exit 0
  fi
fi

stage pilot                                  # B threshold from the chosen N (skipped if B_TAU_XYZ is set)
grep -q '^B_TAU_XYZ=' "$PHASE2_ENV" || bash "$JOB/20_pilot_motion.sh"

stage phase2
bash "$JOB/30_phase2_streams.sh"

stage evaluate
bash "$JOB/40_evaluate.sh"
stage complete

if [ "${STOP_POD:-0}" = "1" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
  conda activate gs
  python "$EXP/runpod/stop_pod.py" --pod-id "$RUNPOD_POD_ID"
fi
