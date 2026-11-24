#!/usr/bin/env bash
# Driver for the third experiment (alternative first-frame initializers) on the experiment-2 pod (run in tmux).
#
#   bash runpod/run_alt_init.sh                 # check data/env -> install taming/edgs/zipsplat envs -> frame-0 screen
#   STOP_POD=1 bash runpod/run_alt_init.sh      # stop (not terminate) this pod at the end, ALSO when something failed
#   SCREEN_ARGS="--n 600000" bash runpod/run_alt_init.sh   # extra options for the screen
#   WITH_BASELINE=1 bash runpod/run_alt_init.sh  # also re-run 3DGS 15k (not used in this run)
#
# Failure policy: one method failing never stops the others.
#   * each method's env is installed separately; a method whose install fails is skipped (logged)
#   * the screen runs only the methods that installed; a failing condition is logged and the rest continue
#   * only problems that affect every method (no GPU, missing data, no disk) stop the run (preflight)
# Status: /workspace/alt_init.stage, per-method install status: $ALT_RUNS/install_status.txt
set -uo pipefail
source "$(dirname "$0")/env.sh"
JOB=$EXP/jobs/alt_init_s0
source "$JOB/config.sh"
cd "$EXP" || exit 1
mkdir -p "$ALT_RUNS"
exec 9>"$WS/alt_init.lock"
flock -n 9 || { echo "another driver is running"; exit 1; }
stage() { echo "$1" > "$WS/alt_init.stage"; echo "$(date -Is) stage=$1"; }

finish() {   # always runs (success or failure): stop the pod if asked, so a failure does not keep billing
  local rc=$?
  if [ "${STOP_POD:-0}" = "1" ]; then
    if [ -n "${RUNPOD_POD_ID:-}" ]; then
      echo "$(date -Is) stopping pod $RUNPOD_POD_ID (exit code $rc)"
      (conda activate gs && python "$EXP/runpod/stop_pod.py" --pod-id "$RUNPOD_POD_ID") \
        || echo "WARN: automatic stop failed -> stop the pod in the console"
    else
      echo "WARN: RUNPOD_POD_ID not set -> cannot stop automatically, stop the pod in the console"
    fi
  fi
}
trap finish EXIT

stage setup
command -v tmux >/dev/null && command -v ffmpeg >/dev/null || bash "$EXP/runpod/00_setup_pod.sh"
[ -x "$CONDA_ROOT/envs/gs/bin/python" ] || bash "$EXP/runpod/01_install_gs.sh" || { stage "failed: setup (env gs)"; exit 1; }
[ -d "$FRAME0/sparse" ] || bash "$EXP/runpod/03_get_data.sh" || { stage "failed: setup (data)"; exit 1; }

stage preflight
bash "$EXP/runpod/05_preflight_alt_init.sh" || { stage "failed: preflight (affects every method)"; exit 1; }

stage install
OK_METHODS=(); BAD_METHODS=()
: > "$ALT_RUNS/install_status.txt"
for m in $ALT_METHODS; do
  if bash "$EXP/runpod/04_install_alt_init.sh" "$m" > "$ALT_RUNS/install_$m.log" 2>&1; then
    OK_METHODS+=("$m"); echo "$m ok" | tee -a "$ALT_RUNS/install_status.txt"
  else
    BAD_METHODS+=("$m"); echo "$m FAILED (see install_$m.log)" | tee -a "$ALT_RUNS/install_status.txt"
    tail -n 30 "$ALT_RUNS/install_$m.log"
  fi
done
if [ ${#OK_METHODS[@]} -eq 0 ]; then stage "failed: install (all methods)"; exit 1; fi
METHODS_CSV=$(IFS=,; echo "${OK_METHODS[*]}")

stage "screen (${METHODS_CSV})"
# shellcheck disable=SC2086
bash "$JOB/10_alt_init_screen.sh" --methods "$METHODS_CSV" ${SCREEN_ARGS:-}
SCREEN_RC=$?

if [ ${#BAD_METHODS[@]} -eq 0 ] && [ $SCREEN_RC -eq 0 ]; then
  stage complete
else
  stage "complete with problems: install failed [${BAD_METHODS[*]:-none}], screen exit $SCREEN_RC (failed conditions listed in the screen log)"
fi
