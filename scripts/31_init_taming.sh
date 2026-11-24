#!/usr/bin/env bash
# [env: taming]  Taming 3DGS (budget-constrained 3DGS training) on frame 0 -> exactly N Gaussians.
# Same settings as our 3DGS 15k baseline (00_train_baseline.sh) except the densification rule:
#   15k iterations, sh_degree 1, images_2, graphdeco densification window (500..15000),
#   + Taming's score-based densification to exactly N (--mode final_count) and --sh_lower (paper "budget" setting).
# Views: the train-only scene (cam00 removed, 30_make_train_scene.py), run WITHOUT --eval
#   (Taming's --eval would hold out every 8th view instead of cam00).
# No 1k refine afterwards: this already is a full optimization (like the 3DGS 15k baseline).
# Two deliberate deviations from the paper defaults (both needed for a fair / compatible comparison):
#   --ho_iteration ITERS+1 : Taming's "high-opacity" switch would store abs() opacities (not sigmoid logits)
#                            in the saved PLY; 3DGStream cannot read those. With 15k iterations it would fire at 15000.
#   --test_iterations -1   : no in-training evaluation (it renders + LPIPS on 5 views inside the timed run);
#                            our evaluation happens afterwards with 04_check_and_eval.py like every other init.
#
#   bash scripts/31_init_taming.sh <N> <out.ply> [iterations=15000]
# Output: <out.ply> + <out>_timing.json (wall clock of train.py, start-up included like the baseline) + log
set -euo pipefail
source "$(dirname "$0")/../runpod/env.sh"
source "$EXP/alt_init/alt_env.sh"

N=${1:?N}; OUT=${2:?out.ply}; ITERS=${3:-15000}
[[ "$N" =~ ^[1-9][0-9]*$ ]] || { echo "invalid N: $N"; exit 1; }
[ -d "$ALT_SCENE/sparse/0" ] || { echo "missing $ALT_SCENE -> run scripts/30_make_train_scene.py first"; exit 1; }
[ -f "$OUT" ] && { echo "Output already exists: $OUT"; exit 1; }
M=${OUT%.ply}_taming_model
mkdir -p "$(dirname "$OUT")"
conda activate taming

BUSY=$(python -c "from common.timing import gpu_exclusive as g; print(0 if g() else 1)")
[ "$BUSY" -gt 0 ] && echo "WARNING: GPU is busy -> timing will be marked non-exclusive"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 1000 > "${OUT%.ply}_gpu.log" &
SMI=$!; trap "kill $SMI 2>/dev/null || true" EXIT

cd "$TAMING_ROOT"
T0=$(date +%s.%N)
python train.py -s "$ALT_SCENE" -m "$M" --images images_2 --sh_degree 1 \
  --iterations "$ITERS" --test_iterations -1 --save_iterations "$ITERS" --ho_iteration $((ITERS + 1)) \
  --mode final_count --budget "$N" --sh_lower 2>&1 | tee "${OUT%.ply}_train.log"
T1=$(date +%s.%N)
kill $SMI 2>/dev/null || true
cd "$EXP"

# Taming switches opacity to abs() ("high-opacity") at --ho_iteration and then saves raw abs values;
# 3DGStream / our refine read sigmoid logits -> that file would be silently wrong. We keep it disabled.
if grep -q "Release opacity limit" "${OUT%.ply}_train.log"; then
  echo "ERROR: Taming switched to abs opacity; PLY opacities are not logits -> not usable"; exit 1; fi
grep -q "Sigmoid rendering mode" "${OUT%.ply}_train.log" || echo "WARNING: could not confirm sigmoid opacity mode in log"
cp "$M/point_cloud/iteration_$ITERS/point_cloud.ply" "$OUT"
PEAK=$(sort -n "${OUT%.ply}_gpu.log" | tail -n1)
python - "$OUT" "$T0" "$T1" "$BUSY" "$ITERS" "$PEAK" "$N" <<'EOF'
import json, sys
from common import timing
from common.ply_io import load_ply, sh_degree_of
out, t0, t1, busy, iters, peak, n = sys.argv[1:]
d = load_ply(out)
kept = json.load(open(__import__("os").environ["ALT_SCENE"] + "/subset.json"))["kept"]
timing.write(out, f"taming_train_{int(iters)//1000}k", float(t1) - float(t0), exclusive=(int(busy) == 0),
             notes=["wall clock of taming-3dgs/train.py incl. start-up + image loading (same rule as 3DGS baseline)",
                    "final_count budget, sh_lower, sh_degree 1, images_2, train-only scene (cam00 removed)"],
             peak_gpu_mem_mb=float(peak),
             init_metadata=dict(model="taming", n_views=len(kept), camera_names=kept, training_iters=int(iters),
                                budget=int(n), seed=0))
print("gaussians:", len(d["xyz"]), "target:", n, "sh degree:", sh_degree_of(d))
print(open(timing.timing_path(out)).read())
EOF
