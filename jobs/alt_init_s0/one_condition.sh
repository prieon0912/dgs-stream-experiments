#!/usr/bin/env bash
# One condition of the third experiment (called by 10_alt_init_screen.sh, one process per condition).
#   bash jobs/alt_init_s0/one_condition.sh <method> <param> [source_dir for edgs_all]
# Runs as its own process on purpose: inside a `( ... ) || handle_failure` subshell bash silently ignores
# `set -e`, so a failed step would not stop the condition. Here every failure stops THIS condition only.
# Exit codes: 0 done (or excluded/skipped), non-zero failed (the screen logs it and continues).
set -eEuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
cd "$EXP"
m=${1:?method}; p=${2:?param}
if [ "$m" = edgs_all ]; then D=$(cond_dir edgs all); else D=$(cond_dir "$m" "$p"); fi
mkdir -p "$D"

evaluate() {   # dir, with_pre(0/1)
  local D=$1 pre=$2
  conda activate gs
  [ -f "$D/final.ply" ] || { echo "missing $D/final.ply"; return 1; }
  local extra=(); [ "$pre" = 1 ] && extra=(--ply pre="$D/init.ply")
  if [ ! -f "$D/static/static.json" ]; then
    python scripts/04_check_and_eval.py --frame_dir "$FRAME0" \
      --aabb "$AABB_CONFIG" --out "$D/static" --train --ply final="$D/final.ply" "${extra[@]}"
  fi
  local rec=(); [ "$pre" = 1 ] && rec=(--pre-ply "$D/init.ply")
  if [ ! -f "$D/initialization.json" ]; then
    python scripts/17_record_initialization.py --run "$(basename "$D")" \
      --ply "$D/final.ply" --aabb "$AABB_CONFIG" --static-json "$D/static/static.json" \
      --out "$D/initialization.json" "${rec[@]}"
  fi
}

as_final() {   # no refine: final = init (copy incl. timing sidecar)
  local D=$1
  [ -f "$D/init.ply" ] || { echo "missing $D/init.ply"; return 1; }
  if [ ! -f "$D/final.ply" ]; then
    cp "$D/init.ply" "$D/final.ply"; cp "$D/init_timing.json" "$D/final_timing.json"
  fi
}

refine() {
  local D=$1
  conda activate gs
  [ -f "$D/init.ply" ] || { echo "missing $D/init.ply"; return 1; }
  if [ ! -f "$D/final.ply" ]; then
    python scripts/05_refine.py --ply "$D/init.ply" --out "$D/final.ply" \
      --frame_dir "$FRAME0" --iters "$REFINE_ITERS" --seed "$SEED"
  fi
}

case $m in
  da3ref)
    if [ ! -f "$D/init.ply" ]; then
      cp "$REF_DIR/selected.ply" "$D/init.ply"; cp "$REF_DIR/selected_timing.json" "$D/init_timing.json"
    fi
    refine "$D"; evaluate "$D" 0;;
  3dgs15k)
    if [ ! -f "$D/init.ply" ]; then
      NAME=${TAG}_3dgs15k
      [ -f "$RUNS/inits/$NAME.ply" ] || bash scripts/00_train_baseline.sh 15000 "$NAME"
      cp "$RUNS/inits/$NAME.ply" "$D/init.ply"; cp "$RUNS/inits/${NAME}_timing.json" "$D/init_timing.json"
    fi
    as_final "$D"; evaluate "$D" 0;;
  taming)
    [ -f "$D/init.ply" ] || bash scripts/31_init_taming.sh "$p" "$D/init.ply" "$TAMING_ITERS"
    as_final "$D"; evaluate "$D" 0;;
  edgs)
    [ -f "$D/excluded.json" ] && { echo "excluded earlier (fewer visible points than N)"; exit 0; }
    if [ ! -f "$D/init.ply" ]; then
      conda activate edgs
      rc=0
      python alt_init/run_edgs.py --edgs_root "$EDGS_ROOT" --scene "$ALT_SCENE" --n "$p" --out "$D/init.ply" --seed "$SEED" || rc=$?
      if [ $rc = 4 ]; then   # not enough visible points: exclude this N, keep the all-points version
        python3 alt_init/edgs_short.py exclude "$D"
        exit 0
      fi
      [ $rc = 0 ] || exit $rc
    fi
    refine "$D"; evaluate "$D" 1;;
  edgs_all)                  # param = "all", $3 = the excluded run whose visible points are all kept
    SRC=${3:?source dir}
    if [ ! -f "$D/init.ply" ]; then
      cp "$SRC/init_ALL.ply" "$D/init.ply"; cp "$SRC/init_ALL_timing.json" "$D/init_timing.json"
      cp "$SRC/init_ALL_edgs.json" "$D/init_edgs.json"; echo "$SRC" > "$D/source_run.txt"
    fi
    refine "$D"; evaluate "$D" 1;;
  zipsplat)
    if [ ! -f "$D/init.ply" ]; then
      conda activate zipsplat
      python alt_init/run_zipsplat.py --frame_dir "$FRAME0" --out "$D/init.ply" --compression "$p" --seed "$SEED"
    fi
    refine "$D"; evaluate "$D" 1;;
  *) echo "unknown method $m"; exit 2;;
esac
echo "condition done: $m $p"
