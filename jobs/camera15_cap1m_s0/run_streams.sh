#!/usr/bin/env bash
set -eEuo pipefail
JOB_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${EXP:-/workspace/dgs_exp}/runpod/env.sh"
export AABB_CONFIG="$JOB_DIR/aabb.json"
export N_FRAMES=300
cd "$EXP"
# settings_sha256: d0ab6c1ecaa76010f68cfc9d8d14882511013f3e10d10d4d9ccf2fc682821821
bash "$EXP/scripts/run_queue.sh" "$JOB_DIR/queue.txt"
