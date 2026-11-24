#!/usr/bin/env bash
# Third experiment, frame 0 only: Taming 3DGS / EDGS / ZipSplat as the first-frame initializer.
#
#   bash jobs/alt_init_s0/10_alt_init_screen.sh                         # all methods, default counts
#   bash jobs/alt_init_s0/10_alt_init_screen.sh --methods edgs --n 600000
#   bash jobs/alt_init_s0/10_alt_init_screen.sh --methods zipsplat --compressions 1.0,0.25
#   bash jobs/alt_init_s0/10_alt_init_screen.sh --with-baseline          # also re-run 3DGS 15k (e.g. after a pod migration)
#
# Per condition:  init (method's own env)  ->  [1k refine for edgs / zipsplat, env gs]
#                 ->  cam00 + train-view quality, AABB counts (04, env gs)  ->  record (17)
#   taming   : exactly N, full 15k optimization, no extra refine (like the 3DGS 15k baseline)
#   edgs     : exactly N, EDGS init + the same 1k refine as DA3+1k
#   zipsplat : count set by --compressions, ZipSplat (+TokenBA) + Sim3 to COLMAP + the same 1k refine
#   3dgs15k  : (only with --with-baseline) our 3DGS 15k baseline re-run on this pod/GPU, same evaluation
#   da3ref   : machine check (off by default, DA3REF=1 to enable, ~2 min): the experiment-2 DA3 opacity-1M selection refined again
#              (1k) on THIS machine. Its refine time vs the stored one = speed ratio of the two machines
#              (needed after a pod migration: old timings were measured on the previous machine)
# Every step is skipped when its output exists (safe to re-run). One condition failing does not stop the rest.
set -eEuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

METHOD_LIST=$ALT_METHODS; NS=$ALT_NS; ZCS=$ZIP_COMPRESSIONS; BASELINE=${WITH_BASELINE:-0}
while [ $# -gt 0 ]; do
  case $1 in
    --methods) METHOD_LIST=${2//,/ }; shift 2;;
    --n) NS=${2//,/ }; shift 2;;
    --compressions) ZCS=${2//,/ }; shift 2;;
    --with-baseline) BASELINE=1; shift;;
    --no-baseline) BASELINE=0; shift;;
    -h|--help) sed -n 2,19p "$0"; exit 0;;
    *) echo "unknown option $1"; exit 1;;
  esac
done
for n in $NS; do [[ "$n" =~ ^[1-9][0-9]*$ ]] || { echo "invalid N: $n"; exit 1; }; done
for z in $ZCS; do python3 -c "import sys; z=float('$z'); sys.exit(0 if 0<z<=1 else 1)" || { echo "invalid compression $z"; exit 1; }; done
for m in $METHOD_LIST; do [[ " $ALT_METHODS " == *" $m "* ]] || { echo "unknown method $m"; exit 1; }; done

cd "$EXP"
mkdir -p "$ALT_RUNS/init" "$RESULTS"
conda activate gs
[ -f "$ALT_SCENE/subset.json" ] || python scripts/30_make_train_scene.py --frame_dir "$FRAME0" --out "$ALT_SCENE"
python scripts/16_snapshot_environment.py --out "$ALT_RUNS/environment_gs.json"
[ -f "$REF_DIR/static/static.json" ] || echo "NOTE: reference $REF_DIR not found -> summary shows absolute values only"

JOBS=()
[ "${DA3REF:-0}" = 1 ] && [ -f "$REF_DIR/selected.ply" ] && JOBS+=("da3ref:1000000")
[ "$BASELINE" = 1 ] && JOBS+=("3dgs15k:15000")
for m in $METHOD_LIST; do
  if [ "$m" = zipsplat ]; then for z in $ZCS; do JOBS+=("zipsplat:$z"); done
  else for n in $NS; do JOBS+=("$m:$n"); done; fi
done
echo "conditions: ${JOBS[*]}"

# each condition runs in its own process (one_condition.sh) so that `set -e` really stops a failed
# condition there, while this loop records the failure and continues with the next condition
FAILED=()
for job in "${JOBS[@]}"; do
  m=${job%%:*}; p=${job##*:}
  D=$(cond_dir "$m" "$p"); mkdir -p "$D"
  echo "=== $(date) $m $p -> $D"
  bash "$JOB_DIR/one_condition.sh" "$m" "$p" 2>&1 | tee -a "$D/alt_init.log" || FAILED+=("$m/$p")
done

# EDGS counts that were excluded: the run with the most visible points becomes "edgs_all" (all points kept)
BEST=$(python3 alt_init/edgs_short.py best "$ALT_RUNS/init" "$TAG")
if [ -n "$BEST" ]; then
  D=$(cond_dir edgs all); mkdir -p "$D"
  echo "=== $(date) edgs all visible points (from $BEST) -> $D"
  bash "$JOB_DIR/one_condition.sh" edgs_all all "$BEST" 2>&1 | tee -a "$D/alt_init.log" || FAILED+=("edgs/all")
fi

conda activate gs
python scripts/34_summarize_alt_init.py --root "$ALT_RUNS/init" --tag "$TAG" --ref "$REF_DIR" --out "$RESULTS"
if [ ${#FAILED[@]} -gt 0 ]; then echo "FAILED: ${FAILED[*]}"; exit 1; fi
echo "alt-init screen done -> $RESULTS/alt_init.csv"
