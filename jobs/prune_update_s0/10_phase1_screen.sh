#!/usr/bin/env bash
# Phase 1 (frame 0 only): pruning method x the gaussian counts YOU choose.
#   select (23) -> 1k refine (05) -> cam00 / train-view quality + AABB stats (04) -> record (17)
#
#   bash jobs/prune_update_s0/10_phase1_screen.sh                       # N = 1000000, all 4 methods
#   bash jobs/prune_update_s0/10_phase1_screen.sh --n 600000            # add another count after looking at the trend
#   bash jobs/prune_update_s0/10_phase1_screen.sh --n 800000,500000 --methods voxel,contrib_rank
#
# Results accumulate: every (method, N) ever run stays in results/phase1.csv, so you can go up or down step by step.
# The reference for "allowed" is always opacity top-1,000,000 (the current pipeline); it is added automatically
# if it has not been run yet. Every step is skipped when its output exists (safe to re-run).
set -eEuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

NS=${PHASE1_NS:-1000000}
METHOD_LIST=$METHODS
while [ $# -gt 0 ]; do
  case $1 in
    --n) NS=${2//,/ }; shift 2;;
    --methods) METHOD_LIST=${2//,/ }; shift 2;;
    -h|--help) sed -n 2,12p "$0"; exit 0;;
    *) echo "unknown option $1 (use --n N1,N2 and/or --methods m1,m2)"; exit 1;;
  esac
done
for n in $NS; do
  [[ "$n" =~ ^[1-9][0-9]*$ ]] || { echo "invalid N: $n"; exit 1; }
done
for m in $METHOD_LIST; do
  [[ " $ALL_METHODS " == *" $m "* ]] || { echo "unknown method $m (choose from: $ALL_METHODS)"; exit 1; }
done

conda activate gs
cd "$EXP"
[ -f "$SRC_PLY" ] || { echo "missing $SRC_PLY -> run 00_prepare_source.sh first"; exit 1; }
mkdir -p "$ROOT/scores" "$RESULTS"
python "$EXP/scripts/16_snapshot_environment.py" --out "$ROOT/environment_gs.json"

JOBS=("opacity:1000000")                    # reference first (skipped quickly if already done)
for m in $METHOD_LIST; do for n in $NS; do
  [ "$m:$n" = "opacity:1000000" ] || JOBS+=("$m:$n")
done; done
echo "phase 1 conditions: ${JOBS[*]}"

FAILED=()
for job in "${JOBS[@]}"; do
  m=${job%%:*}; n=${job##*:}
  D=$(init_dir "$m" "$n")
  mkdir -p "$D"
  echo "=== $(date) $m N=$n -> $D"
  (
    set -eEuo pipefail
    [ -f "$D/selected.ply" ] || python "$EXP/scripts/23_select_gaussians.py" --ply "$SRC_PLY" \
      --out "$D/selected.ply" --method "$m" --n "$n" --seed "$SEED" \
      --frame_dir "$FRAME0" --scores_cache "$ROOT/scores/contrib_scores.npz"
    [ -f "$D/refined.ply" ] || python "$EXP/scripts/05_refine.py" --ply "$D/selected.ply" \
      --out "$D/refined.ply" --frame_dir "$FRAME0" --iters "$REFINE_ITERS" --seed "$SEED"
    [ -f "$D/static/static.json" ] || python "$EXP/scripts/04_check_and_eval.py" --frame_dir "$FRAME0" \
      --aabb "$AABB_CONFIG" --out "$D/static" --train --ply final="$D/refined.ply" --ply pre="$D/selected.ply"
    [ -f "$D/initialization.json" ] || python "$EXP/scripts/17_record_initialization.py" \
      --run "$(basename "$D")" --ply "$D/refined.ply" --aabb "$AABB_CONFIG" \
      --static-json "$D/static/static.json" --out "$D/initialization.json" --pre-ply "$D/selected.ply"
  ) 2>&1 | tee -a "$D/phase1.log" || FAILED+=("$m/$n")
done
python "$EXP/scripts/24_summarize_phase1.py" --root "$ROOT" --tag "$TAG" --out "$RESULTS" \
  --write-env "$PHASE2_ENV.suggested"
if [ ${#FAILED[@]} -gt 0 ]; then echo "FAILED: ${FAILED[*]}"; exit 1; fi
echo "phase 1 done -> $RESULTS/phase1.csv (trend per method printed above)"
