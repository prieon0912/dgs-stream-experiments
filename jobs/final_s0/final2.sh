#!/usr/bin/env bash
# Final 300-frame runs: per init {A+R*, M*+R*} (SET=16 -> {A, M*, R*, M*+R*}), M*/R* decided from this pod's pilots.
#   bash jobs/final_s0/final2.sh BE   -> opacity1m, redundancy600k
#   bash jobs/final_s0/final2.sh AC   -> reproj600k, zipsplat05
#   add "plan" as 2nd arg to only print the decision without running anything.
# Resumable: finished runs (state/<name>.done) are skipped. A failed run is recorded and skipped.
# The pod is never stopped by this script.
set -uo pipefail
ROLE=${1:?BE|AC}; ONLY=${2:-run}
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
conda activate gs
cd "$EXP"
case $ROLE in BE) MY_INITS="opacity1m redundancy600k";; AC) MY_INITS="reproj600k zipsplat05";; *) exit 1;; esac
W=main_$ROLE
LOG=$FINAL/final2_$ROLE.log
log() { echo "$(date -Is) [$W] $*" | tee -a "$LOG"; }
mkdir -p "$STATE/claims" "$STORE" "$RESULTS" "$RUNS/stream"
B_TAU=0.00880432
if [ -f "$FINAL_ENV" ]; then source "$FINAL_ENV"; B_TAU=${B_TAU_XYZ:-$B_TAU}; fi
E_EVERY=${E_EVERY:-$(cat "$FINAL/e_every.txt" 2>/dev/null)}
[ -n "$E_EVERY" ] || { log "no E_EVERY (final.env / e_every.txt) -> abort"; exit 1; }

PLAN=$RESULTS/final_plan_$ROLE
python "$EXP/scripts/42_decide_mr.py" --store "$STORE" --state "$STATE" --inits $MY_INITS \
  --e_every "$E_EVERY" --set "${SET:-8}" --out "$PLAN" 2>&1 | tee -a "$LOG"
if [ "${PIPESTATUS[0]}" = 2 ] && [ "${FORCE:-0}" != 1 ]; then
  log "some pilots missing (see above) -> abort; FORCE=1 to run with the partial plan"; exit 1; fi
[ -f "$PLAN.env" ] || { log "decision failed -> abort"; exit 1; }
source "$PLAN.env"
log "plan: $(tr '\n' ' ' < "$PLAN.env") B_TAU=$B_TAU E_EVERY=$E_EVERY"
[ "$ONLY" = plan ] && exit 0

N=300
run_one() {   # name init mode fast args...
  local name=$1 init=$2 mode=$3 fast=$4; shift 4
  [ -f "$STATE/$name.done" ] || [ -f "$STATE/$name.failed" ] && return 0
  log "start $name ($mode fast=$fast, $N frames) $*"
  local ok=1 pre=()
  [ "$fast" = 1 ] && pre=(env DGR_MODE=3 PYTHONPATH=/workspace/dgr_fast_pkg)
  if "${pre[@]}" env N_FRAMES=$N bash "$EXP/scripts/08b_run_stream_select.sh" "$name" "$init" "$mode" "$@" \
       > "$RUNS/stream/$name.driver.log" 2>&1; then
    N_FRAMES=$N python "$EXP/scripts/09_eval_stream.py" --runs "$name" --expected-frames "$N" \
       --lpips_every 1 >> "$RUNS/stream/$name.driver.log" 2>&1 || ok=0
  else ok=0; fi
  mkdir -p "$STORE/$name"
  rsync -a --exclude 'out/frame*/*.ply' --exclude 'out/frame*/point_cloud' "$RUNS/stream/$name/" "$STORE/$name/" \
    && cp "$RUNS/stream/$name.driver.log" "$STORE/$name/" || ok=0
  if [ $ok = 1 ]; then touch "$STATE/$name.done"; log "done $name"
  else touch "$STATE/$name.failed"; log "FAILED $name"; fi
  rm -rf "$RUNS/stream/$name/out"
}
args_for() {
  case $1 in
    A)  ARGS=() ;;
    B)  ARGS=(--b_tau_xyz "$B_TAU" "${B_FIXED[@]}") ;;
    C)  ARGS=("${C_FIXED[@]}") ;;
    EB) ARGS=(--b_tau_xyz "$B_TAU" "${B_FIXED[@]}" "${C_FIXED[@]}" "${E_FIXED[@]}" --e_every "$E_EVERY") ;;
  esac
}
for code in $MY_INITS; do
  v=PLAN_$code
  for variant in ${!v}; do
    fast=0; mode=$variant
    [[ $variant == *R ]] && { fast=1; mode=${variant%R}; }
    args_for "$mode"
    run_one "final_s0_main_${code}_${variant}" "$INITS/$code.ply" "$mode" "$fast" "${ARGS[@]}"
  done
done
python "$EXP/scripts/41_summarize_final.py" --store "$STORE" --state "$STATE" --tag final_s0_main \
  --inits $MY_INITS --modes A B C EB AR BR CR EBR --out "$RESULTS" >> "$FINAL/summary.log" 2>&1 || true
log "final2 finished (pod left running)"
touch "$STATE/final2_$ROLE.finished"
