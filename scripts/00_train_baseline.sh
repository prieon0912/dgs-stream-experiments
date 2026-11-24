#!/usr/bin/env bash
# [env: gs]  Optimization-based baseline init, trained by us so its time is measurable.
# 3DGStream README step 4: vanilla 3DGS (repo's train.py) on frame000000 with --sh_degree 1.
#
#   bash scripts/00_train_baseline.sh [iterations=15000] [name=baseline_own]
#
# Output: $RUNS/inits/<name>.ply  + <name>_timing.json + <name>_train.log
# Timing = wall clock of train.py (includes ~5-10 s python/CUDA start-up and image loading; noted in json).
# --eval -> cam00 (index 0 after sorting) is held out, same as in 3DGStream streaming.
# Run with the GPU idle (no queue running), otherwise the time is not reportable.
set -euo pipefail
source "$(dirname "$0")/../runpod/env.sh"
conda activate gs

ITERS=${1:-15000}; NAME=${2:-baseline_own}
OUT=$RUNS/inits; M=$OUT/${NAME}_model
mkdir -p "$OUT"
if [ -f "$OUT/$NAME.ply" ]; then echo "Output already exists: $OUT/$NAME.ply"; exit 1; fi

BUSY=$(python -c "from common.timing import gpu_exclusive as g; print(0 if g() else 1)")
if [ "$BUSY" -gt 0 ]; then echo "WARNING: GPU is busy (queue running?) -> timing will be marked non-exclusive"; fi

nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 1000 > "$OUT/${NAME}_gpu.log" &
SMI=$!; trap "kill $SMI 2>/dev/null || true" EXIT

cd "$REPO_GS"
T0=$(date +%s.%N)
python train.py -s "$FRAME0" -m "$M" --images images_2 --sh_degree 1 --eval \
  --iterations "$ITERS" --test_iterations "$ITERS" --save_iterations "$ITERS" \
  --port $((6009 + RANDOM % 1000)) 2>&1 | tee "$OUT/${NAME}_train.log"
T1=$(date +%s.%N)
kill $SMI 2>/dev/null || true

cp "$M/point_cloud/iteration_$ITERS/point_cloud.ply" "$OUT/$NAME.ply"
PEAK=$(sort -n "$OUT/${NAME}_gpu.log" | tail -n1)
python - "$OUT/$NAME.ply" "$T0" "$T1" "$BUSY" "$ITERS" "$PEAK" <<'EOF'
import sys
from common import timing
from common.colmap_io import load_cameras
import os
ply, t0, t1, busy, iters, peak = sys.argv[1:]
cameras = load_cameras(os.environ["FRAME0"], exclude=("cam00",))
timing.write(ply, f"3dgs_train_{int(iters)//1000}k", float(t1) - float(t0), exclusive=(int(busy) == 0),
             notes=["wall clock of 3DGStream/train.py incl. start-up + image loading",
                    "densification on (graphdeco defaults), sh_degree 1, images_2, cam00 held out"],
             peak_gpu_mem_mb=float(peak))
data = timing.load(ply)
data["init_metadata"] = dict(model="3dgs", n_views=len(cameras), camera_names=[camera.name for camera in cameras],
                             training_iters=int(iters), seed=0)
import json
with open(timing.timing_path(ply), "w") as handle:
    json.dump(data, handle, indent=2)
print(open(timing.timing_path(ply)).read())
EOF
