#!/usr/bin/env bash
# One worker per pod; the two A6000 pods do NOT share storage, so the work is split by role up front:
#   role BE : pilot p_A (31 frames, motion dump -> B_TAU_XYZ), p_B (31), p_EB10 (61), p_EB30 (61),
#             decide E_EVERY (N with PSNR within 0.1 dB of the best EB pilot, then the faster),
#             then main B and EB for the 4 inits (300 frames)
#   role AC : pilot p_C (31), then main A and C for the 4 inits (A and C need no pilot value)
#   role ALL: everything on one pod (fallback)
# A failed run is recorded and skipped; the worker never stops the pod.
#   bash jobs/final_s0/worker.sh <worker-label> <BE|AC|ALL>
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
conda activate gs
cd "$EXP"
W=${1:-$(hostname)}
ROLE=${2:-ALL}
case $ROLE in BE) MY_MODES="B EB";; AC) MY_MODES="A C";; ALL) MY_MODES=$MAIN_MODES;; *) echo "role BE|AC|ALL"; exit 1;; esac
want() { [[ " $MY_MODES " == *" $1 "* ]]; }
mkdir -p "$STATE/claims" "$STORE" "$RESULTS" "$RUNS/stream"
log() { echo "$(date -Is) [$W] $*" | tee -a "$FINAL/worker_$W.log"; }

wait_for() {   # file, seconds
  local f=$1 t=${2:-14400} s=0
  while [ ! -f "$f" ]; do sleep 30; s=$((s + 30)); [ $s -ge $t ] && return 1; done
}

run_one() {    # name init mode frames args...
  local name=$1 init=$2 mode=$3 frames=$4; shift 4
  [ -f "$STATE/$name.done" ] || [ -f "$STATE/$name.failed" ] && return 0
  mkdir "$STATE/claims/$name" 2>/dev/null || return 0
  echo "$W $(date -Is)" > "$STATE/claims/$name/by"
  log "start $name ($mode, $frames frames) $*"
  local ok=1
  if N_FRAMES=$frames bash "$EXP/scripts/08b_run_stream_select.sh" "$name" "$init" "$mode" "$@" \
       > "$RUNS/stream/$name.driver.log" 2>&1; then
    N_FRAMES=$frames python "$EXP/scripts/09_eval_stream.py" --runs "$name" --expected-frames "$frames" \
       --lpips_every 1 >> "$RUNS/stream/$name.driver.log" 2>&1 || ok=0
  else
    ok=0
  fi
  mkdir -p "$STORE/$name"
  rsync -a --exclude 'out/frame*/*.ply' --exclude 'out/frame*/point_cloud' "$RUNS/stream/$name/" "$STORE/$name/" \
    && cp "$RUNS/stream/$name.driver.log" "$STORE/$name/" || ok=0
  if [ $ok = 1 ]; then touch "$STATE/$name.done"; log "done $name"
  else touch "$STATE/$name.failed"; log "FAILED $name (see $STORE/$name/$name.driver.log)"; fi
  # free local disk: frame PLYs are not needed after evaluation
  rm -rf "$RUNS/stream/$name/out"
  return 0
}

P=${TAG}_pilot
PINIT=$INITS/$PILOT_INIT.ply

# ---------------- pilot
if want C; then run_one ${P}_C_$PILOT_INIT "$PINIT" C 31 "${C_FIXED[@]}" --profile_frames "$PROFILE_FRAMES_PILOT"; fi
if want B; then
run_one ${P}_A_$PILOT_INIT "$PINIT" A 31 --dump_motion "$RUNS/stream/${P}_A_$PILOT_INIT/motion" \
        --profile_frames "$PROFILE_FRAMES_PILOT"
if [ -f "$STATE/${P}_A_$PILOT_INIT.done" ] && [ ! -f "$PILOT_ENV" ] && mkdir "$STATE/claims/tau" 2>/dev/null; then
  python "$EXP/scripts/26_suggest_thresholds.py" --dump "$STORE/${P}_A_$PILOT_INIT/motion" \
    --write-env "$PILOT_ENV.tmp" > "$FINAL/pilot_tau.log" 2>&1 && mv "$PILOT_ENV.tmp" "$PILOT_ENV"
  log "tau: $(cat "$PILOT_ENV" 2>/dev/null || echo FAILED)"
fi
if [ ! -f "$PILOT_ENV" ]; then log "no B_TAU_XYZ (pilot A failed or tau script failed) -> pilot B/EB and main B/EB cannot run"; fi
B_TAU=$(sed -n 's/^B_TAU_XYZ=//p' "$PILOT_ENV" 2>/dev/null | tail -n1)
if [ -n "$B_TAU" ]; then
  run_one ${P}_B_$PILOT_INIT "$PINIT" B 31 --b_tau_xyz "$B_TAU" "${B_FIXED[@]}" --profile_frames "$PROFILE_FRAMES_PILOT"
  for n in $E_CANDIDATES; do
    run_one ${P}_EB${n}_$PILOT_INIT "$PINIT" EB 61 --b_tau_xyz "$B_TAU" "${B_FIXED[@]}" "${C_FIXED[@]}" \
            "${E_FIXED[@]}" --e_every "$n" --profile_frames "$PROFILE_FRAMES_PILOT"
  done
fi

# ---------------- decide (once; whoever gets here first after both EB pilots ended)
for n in $E_CANDIDATES; do
  f=${P}_EB${n}_$PILOT_INIT
  while [ ! -f "$STATE/$f.done" ] && [ ! -f "$STATE/$f.failed" ] && [ -n "$B_TAU" ]; do sleep 30; done
done
if [ ! -f "$FINAL_ENV" ] && mkdir "$STATE/claims/decide" 2>/dev/null; then
  python "$EXP/scripts/40_decide_final.py" --store "$STORE" --pilot-prefix "$P" --init "$PILOT_INIT" \
    --candidates $E_CANDIDATES --pilot-env "$PILOT_ENV" --out "$FINAL_ENV" --report "$RESULTS/pilot_report.json" \
    > "$FINAL/decide.log" 2>&1
  log "decision: $(tr '\n' ' ' < "$FINAL_ENV" 2>/dev/null || echo FAILED)"
fi
[ -f "$FINAL_ENV" ] || log "no final.env -> main B/EB runs fall back to pilot tau and E_EVERY=10"
source "$FINAL_ENV" 2>/dev/null || true
B_TAU=${B_TAU_XYZ:-$B_TAU}; E_EVERY=${E_EVERY:-10}
fi   # want B

# ---------------- main: longest first (big inits, mode A) so the two pods finish together
B_TAU=${B_TAU:-}; E_EVERY=${E_EVERY:-10}
for mode in $MY_MODES; do
  for code in $INIT_CODES; do
    name=${TAG}_${code}_${mode}
    case $mode in
      A)  args=() ;;
      B)  args=(--b_tau_xyz "$B_TAU" "${B_FIXED[@]}") ;;
      C)  args=("${C_FIXED[@]}") ;;
      EB) args=(--b_tau_xyz "$B_TAU" "${B_FIXED[@]}" "${C_FIXED[@]}" "${E_FIXED[@]}" --e_every "$E_EVERY") ;;
    esac
    if [[ $mode == B || $mode == EB ]] && [ -z "$B_TAU" ]; then
      [ -f "$STATE/$name.failed" ] || { touch "$STATE/$name.failed"; log "FAILED $name: no B_TAU_XYZ"; }
      continue
    fi
    run_one "$name" "$INITS/$code.ply" "$mode" "$N_FRAMES" "${args[@]}"
  done
done

# ---------------- tables (whoever finishes last; re-runnable by hand)
python "$EXP/scripts/41_summarize_final.py" --store "$STORE" --state "$STATE" --tag "$TAG" \
  --inits $INIT_CODES --modes $MY_MODES --out "$RESULTS" >> "$FINAL/summary.log" 2>&1 || true
log "worker finished (pod is left running)"
touch "$STATE/worker_$W.finished"
