# Final experiment: 4 first-frame inits x update modes A/B/C/EB, 300 frames (16 runs) + a 30/60-frame pilot.
# Two (or more) pods share ONE network volume mounted at /workspace; each pod runs one worker.sh.
JOB_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export RUNS=${RUNS:-/root/runs}                       # per-pod LOCAL disk while a run trains (fast, no network IO)
source "$JOB_DIR/../../runpod/env.sh"
export AABB_CONFIG="$JOB_DIR/aabb.json"               # same fixed AABB as every earlier experiment
export TAG=final_s0
export FINAL=$WS/final                                # shared: inits, queue state, results
export INITS=$FINAL/inits                             # <code>.ply (+ <code>_timing.json)
export STATE=$FINAL/state                             # claims/, *.done, *.failed
export STORE=$FINAL/runs                              # compact copy of every finished run (no frame PLYs)
export RESULTS=$FINAL/results
export PILOT_ENV=$FINAL/pilot_tau.env                 # B_TAU_XYZ from the A pilot
export FINAL_ENV=$FINAL/final.env                     # B_TAU_XYZ + E_EVERY used by all 16 main runs

export INIT_CODES="opacity1m redundancy600k reproj600k zipsplat05"
export MAIN_MODES="A B C EB"
export PILOT_INIT=opacity1m
export E_CANDIDATES="10 30"
export PROFILE_FRAMES_PILOT="25,26"                   # torch.profiler breakdown (pilot only; never in main runs)

B_FIXED=(--b_window 10 --b_warmup 10 --b_recheck 30 --b_tau_rot_deg inf)
C_FIXED=(--c_kappa 2 --c_tau 25 --c_radius 48)
E_FIXED=(--e_q 0.01 --e_margin 0.1 --e_min_points 1000)
