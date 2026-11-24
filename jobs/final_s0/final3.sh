#!/usr/bin/env bash
# Final 300-frame ladder (chosen 2026-11-24 20:22): per init  A -> A+R* -> A+R*+fusedSSIM -> M*+R*+fusedSSIM
#   bash jobs/final_s0/final3.sh BE   -> opacity1m, redundancy600k
#   bash jobs/final_s0/final3.sh AC   -> reproj600k, zipsplat05
# R* = DGR_MODE=3 with /workspace/dgr_fast_pkg; fused = --fused_ssim on with /workspace/fused_ssim_pkg.
# M* per init from the 30-frame pilots (rule: PSNR loss <= 0.1 dB vs A, fastest; results/mstar_pilot_results.csv).
# Resumable (state/<name>.done|failed skipped); a failed run is recorded and the queue goes on; the pod is never stopped.
set -uo pipefail
ROLE=${1:?BE|AC}
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
conda activate gs
cd "$EXP"
case $ROLE in BE) MY_INITS="opacity1m redundancy600k";; AC) MY_INITS="reproj600k zipsplat05";; *) exit 1;; esac
declare -A MSTAR=([opacity1m]=EB [redundancy600k]=B [reproj600k]=EB [zipsplat05]=EB)
W=ladder_$ROLE
LOG=$FINAL/final3_$ROLE.log
log() { echo "$(date -Is) [$W] $*" | tee -a "$LOG"; }
mkdir -p "$STATE" "$STORE" "$RESULTS" "$RUNS/stream"
B_TAU=0.00880432
[ -f "$FINAL_ENV" ] && { source "$FINAL_ENV"; B_TAU=${B_TAU_XYZ:-$B_TAU}; }
E_EVERY=${E_EVERY:-$(cat "$FINAL/e_every.txt" 2>/dev/null)}
E_EVERY=${E_EVERY:-30}
N=300
log "start ladder inits=[$MY_INITS] B_TAU=$B_TAU E_EVERY=$E_EVERY M*=opacity1m:EB redundancy600k:B reproj600k:EB zipsplat05:EB"

args_for() {
  case $1 in
    A)  ARGS=() ;;
    B)  ARGS=(--b_tau_xyz "$B_TAU" "${B_FIXED[@]}") ;;
    C)  ARGS=("${C_FIXED[@]}") ;;
    EB) ARGS=(--b_tau_xyz "$B_TAU" "${B_FIXED[@]}" "${C_FIXED[@]}" "${E_FIXED[@]}" --e_every "$E_EVERY") ;;
  esac
}
run_one() {   # name init mode fast fused
  local name=$1 init=$2 mode=$3 fast=$4 fused=$5
  [ -f "$STATE/$name.done" ] || [ -f "$STATE/$name.failed" ] && return 0
  args_for "$mode"
  local pp="" extra=() envs=(N_FRAMES=$N)
  [ "$fast" = 1 ] && { envs+=(DGR_MODE=3); pp=/workspace/dgr_fast_pkg; }
  [ "$fused" = 1 ] && { pp=${pp:+$pp:}/workspace/fused_ssim_pkg; extra=(--fused_ssim on); }
  [ -n "$pp" ] && envs+=(PYTHONPATH=$pp)
  log "start $name (mode=$mode R*=$fast fused=$fused) ${ARGS[*]} ${extra[*]}"
  local ok=1
  if env "${envs[@]}" bash "$EXP/scripts/08b_run_stream_select.sh" "$name" "$init" "$mode" "${ARGS[@]}" "${extra[@]}" \
       > "$RUNS/stream/$name.driver.log" 2>&1; then
    N_FRAMES=$N python "$EXP/scripts/09_eval_stream.py" --runs "$name" --expected-frames "$N" \
       --lpips_every 1 >> "$RUNS/stream/$name.driver.log" 2>&1 || ok=0
  else ok=0; fi
  mkdir -p "$STORE/$name"
  rsync -a --exclude 'out/frame*/*.ply' --exclude 'out/frame*/point_cloud' "$RUNS/stream/$name/" "$STORE/$name/" \
    && cp "$RUNS/stream/$name.driver.log" "$STORE/$name/" || ok=0
  if [ $ok = 1 ]; then touch "$STATE/$name.done"; log "done $name"
  else touch "$STATE/$name.failed"; log "FAILED $name (see $STORE/$name/$name.driver.log)"; fi
  rm -rf "$RUNS/stream/$name/out"
}
for code in $MY_INITS; do
  m=${MSTAR[$code]}
  run_one "final_s0_L_${code}_A"            "$INITS/$code.ply" A    0 0
  run_one "final_s0_L_${code}_AR"           "$INITS/$code.ply" A    1 0
  run_one "final_s0_L_${code}_ARF"          "$INITS/$code.ply" A    1 1
  run_one "final_s0_L_${code}_${m}RF"       "$INITS/$code.ply" "$m" 1 1
done
log "ladder finished (pod left running)"
touch "$STATE/final3_$ROLE.finished"
