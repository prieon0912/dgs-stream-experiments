#!/usr/bin/env bash
# Evaluate the 12 phase-2 runs (per-frame cam00 PSNR/SSIM/LPIPS, frame times) and build the tables.
set -eEuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
conda activate gs
cd "$EXP"
source "$PHASE2_ENV"
RUNS_LIST=()
for mode in A B C; do
  for m in $METHODS; do
    var=N_$m; NAME=$(run_name "$m" "${!var}" "$mode")
    [ -d "$RUNS/stream/$NAME" ] && RUNS_LIST+=("$NAME")
  done
done
[ ${#RUNS_LIST[@]} -gt 0 ] || { echo "no phase-2 runs found"; exit 1; }
status=0
python "$EXP/scripts/09_eval_stream.py" --runs "${RUNS_LIST[@]}" --expected-frames "$N_FRAMES" --lpips_every 1 || status=1
python "$EXP/scripts/25_summarize_phase2.py" --runs "${RUNS_LIST[@]}" --tag "$TAG" --out "$RESULTS" || status=1
exit $status
