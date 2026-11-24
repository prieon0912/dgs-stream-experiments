# Shared settings of the pruning x update-mode experiment (sourced by every script in this folder).
JOB_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$JOB_DIR/../../runpod/env.sh"
export AABB_CONFIG="$JOB_DIR/aabb.json"           # same fixed AABB as the camera experiment
export TAG=prune_s0
export ROOT=$RUNS/prune_update                      # phase-1 inits, scores, summaries
export RESULTS=$ROOT/results

# DA3 all-views output before any cap (2,935,193 gaussians in the camera experiment).
# Either copy it from the old pod (README_PRUNE_UPDATE.md, "source PLY") or let 00_prepare_source.sh regenerate it.
export SRC_DIR=$ROOT/source_all_views
export SRC_PLY=$SRC_DIR/converted.ply

export METHODS="opacity voxel contrib_rank contrib_sample"           # phase-1 default list and phase-2 methods
export ALL_METHODS="$METHODS redundancy redundancy_once reproj_dedup"           # selectable with --methods
export PHASE1_NS=${PHASE1_NS:-"1000000 800000 600000 400000"}   # override with --n on 10_phase1_screen.sh
export REFINE_ITERS=1000
export SEED=0

# phase-2 choices (written by 24_summarize_phase1.py --write-env / 26_suggest_thresholds.py, editable by hand)
export PHASE2_ENV=$JOB_DIR/phase2.env

init_dir()  { echo "$ROOT/init/${TAG}_$1_$2"; }        # method N
run_name()  { echo "${TAG}_$1_$2_$3"; }               # method N mode
