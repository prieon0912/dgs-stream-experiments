#!/usr/bin/env bash
# Mode-B threshold pilot, run right before phase 2 (after the N per method is fixed in phase2.env):
# 30 frames of mode A on the CHOSEN init, per-gaussian motion dumped, then B_TAU_XYZ is appended to phase2.env.
#   bash jobs/prune_update_s0/20_pilot_motion.sh [method]      (default: opacity, at N_opacity from phase2.env)
# One pilot gives one B_TAU_XYZ, used for all four pruning methods. Not part of the 12 conditions.
set -eEuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
conda activate gs
cd "$EXP"
[ -f "$PHASE2_ENV" ] || { echo "missing $PHASE2_ENV: choose N per method first (phase 1)"; exit 1; }
source "$PHASE2_ENV"
M=${1:-${PILOT_METHOD:-opacity}}
var=N_$M; N=${!var:?"set $var in phase2.env first"}
INIT=$(init_dir "$M" "$N")/refined.ply
[ -f "$INIT" ] || { echo "missing $INIT -> run 10_phase1_screen.sh --n $N --methods $M"; exit 1; }
NAME=${TAG}_pilot_A_${M}_${N}
DUMP=$RUNS/stream/$NAME/motion
if [ ! -f "$DUMP/threshold_suggestion.json" ]; then
  N_FRAMES=31 bash "$EXP/scripts/08b_run_stream_select.sh" "$NAME" "$INIT" A --dump_motion "$DUMP"
fi
if grep -q '^B_TAU_XYZ=' "$PHASE2_ENV"; then
  echo "B_TAU_XYZ already set in $PHASE2_ENV (delete that line to replace it with the pilot suggestion)"
  python "$EXP/scripts/26_suggest_thresholds.py" --dump "$DUMP"
else
  echo "# pilot: $NAME" >> "$PHASE2_ENV"
  python "$EXP/scripts/26_suggest_thresholds.py" --dump "$DUMP" --write-env "$PHASE2_ENV"
fi
grep '^B_TAU_XYZ=' "$PHASE2_ENV"
