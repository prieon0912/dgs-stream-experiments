# Shared settings of the third experiment: alternative first-frame initializers (frame 0 only).
JOB_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$JOB_DIR/../../runpod/env.sh"
source "$EXP/alt_init/alt_env.sh"
export AABB_CONFIG="$EXP/jobs/prune_update_s0/aabb.json"     # same fixed AABB as experiments 1 and 2
export TAG=alt_s0
export RESULTS=$ALT_RUNS/results

export ALT_METHODS="taming edgs zipsplat"
export ALT_NS=${ALT_NS:-"1000000 800000 600000 400000"}       # same counts as the pruning experiment
export ZIP_COMPRESSIONS=${ZIP_COMPRESSIONS:-"1.0 0.5"}        # ZipSplat cannot hit an exact N; these set its count
export REFINE_ITERS=1000                                      # EDGS / ZipSplat get exactly the DA3+1k refine
export TAMING_ITERS=15000                                     # = our 3DGS 15k baseline schedule
export SEED=0

# reference for the 0.1 / 0.3 / 0.5 dB rule: DA3 all views, opacity top-1,000,000 + 1k (experiment 2, same pod)
export REF_DIR=$RUNS/prune_update/init/prune_s0_opacity_1000000

cond_dir() { echo "$ALT_RUNS/init/${TAG}_$1_$2"; }            # method param
