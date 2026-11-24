#!/usr/bin/env bash
# Phase 2: 4 pruning methods (each at its chosen N) x update modes A/B/C, 300 frames each = 12 runs.
#   bash jobs/prune_update_s0/30_phase2_streams.sh            # all modes, order A -> B -> C
#   bash jobs/prune_update_s0/30_phase2_streams.sh A          # only mode A (e.g. to compare pruning first)
# Runs are sequential (exclusive GPU => comparable frame times) and resumable; a failed run is
# recorded and the next one starts. Needs phase2.env (copy and edit phase2.env.suggested).
set -eEuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
conda activate gs
cd "$EXP"
[ -f "$PHASE2_ENV" ] || { echo "missing $PHASE2_ENV (copy $PHASE2_ENV.suggested after checking it)"; exit 1; }
source "$PHASE2_ENV"
: "${B_TAU_XYZ:?set B_TAU_XYZ in phase2.env (20_pilot_motion.sh suggests one)}"
B_ARGS=(--b_tau_xyz "$B_TAU_XYZ" --b_tau_rot_deg "${B_TAU_ROT_DEG:-inf}" --b_window "${B_WINDOW:-10}"
        --b_warmup "${B_WARMUP:-10}" --b_recheck "${B_RECHECK:-30}")
C_ARGS=(--c_kappa "${C_KAPPA:-2}" --c_tau "${C_TAU:-25}" --c_radius "${C_RADIUS:-48}")

MODES=${*:-A B C}
FAILED=()
for mode in $MODES; do
  for m in $METHODS; do
    var=N_$m; n=${!var:?"set $var in phase2.env"}
    INIT=$(init_dir "$m" "$n")/refined.ply
    NAME=$(run_name "$m" "$n" "$mode")
    case $mode in A) ARGS=();; B) ARGS=("${B_ARGS[@]}");; C) ARGS=("${C_ARGS[@]}");; *) echo "bad mode $mode"; exit 1;; esac
    if [ ! -f "$INIT" ]; then echo "missing init $INIT"; FAILED+=("$NAME"); continue; fi
    echo "=== $(date) $NAME"
    if ! bash "$EXP/scripts/08b_run_stream_select.sh" "$NAME" "$INIT" "$mode" "${ARGS[@]}"; then
      python "$EXP/scripts/19_execution_status.py" record --run-dir "$RUNS/stream/$NAME" \
        --stage streaming --state failed --exit-code 1 || true
      FAILED+=("$NAME")
    fi
  done
done
echo "$(date -Is) phase2 finished; failed: ${FAILED[*]:-none}" | tee -a "$ROOT/phase2_status.log"
[ ${#FAILED[@]} -eq 0 ]
