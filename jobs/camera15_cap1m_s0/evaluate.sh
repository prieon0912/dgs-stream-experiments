#!/usr/bin/env bash
set -eEuo pipefail
JOB_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${EXP:-/workspace/dgs_exp}/runpod/env.sh"
export AABB_CONFIG="$JOB_DIR/aabb.json"
export N_FRAMES=300
cd "$EXP"
# settings_sha256: d0ab6c1ecaa76010f68cfc9d8d14882511013f3e10d10d4d9ccf2fc682821821
conda activate gs
evaluation_failed=0
python "$EXP/scripts/09_eval_stream.py" --runs camera15_cap1m_s0_baseline_own camera15_cap1m_s0_da3_raw camera15_cap1m_s0_da3_1k camera15_cap1m_s0_random_04_da3_1000 camera15_cap1m_s0_random_08_da3_1000 camera15_cap1m_s0_random_12_da3_1000 camera15_cap1m_s0_random_18_da3_1000 camera15_cap1m_s0_position_fps_04_da3_1000 camera15_cap1m_s0_position_fps_08_da3_1000 camera15_cap1m_s0_position_fps_12_da3_1000 camera15_cap1m_s0_position_fps_18_da3_1000 camera15_cap1m_s0_direction_fps_04_da3_1000 camera15_cap1m_s0_direction_fps_08_da3_1000 camera15_cap1m_s0_direction_fps_12_da3_1000 camera15_cap1m_s0_direction_fps_18_da3_1000 --expected-frames 300 --lpips_every 1 || evaluation_failed=1
python "$EXP/scripts/15_collect_results.py" --manifest "$JOB_DIR/experiment_manifest.json" --out-dir /workspace/runs/camera_capped/results --require-complete || evaluation_failed=1
# Figures are drawn for whatever runs finished, even when some conditions were skipped.
python "$EXP/scripts/10_report.py" --skip-unfinished --out /workspace/runs/camera_capped/results/figures --runs 'camera15_cap1m_s0_baseline_own:3DGS 15k' 'camera15_cap1m_s0_da3_raw:DA3 all views' 'camera15_cap1m_s0_da3_1k:DA3+1k all views' camera15_cap1m_s0_random_04_da3_1000:random_04+1k camera15_cap1m_s0_random_08_da3_1000:random_08+1k camera15_cap1m_s0_random_12_da3_1000:random_12+1k camera15_cap1m_s0_random_18_da3_1000:random_18+1k camera15_cap1m_s0_position_fps_04_da3_1000:position_fps_04+1k camera15_cap1m_s0_position_fps_08_da3_1000:position_fps_08+1k camera15_cap1m_s0_position_fps_12_da3_1000:position_fps_12+1k camera15_cap1m_s0_position_fps_18_da3_1000:position_fps_18+1k camera15_cap1m_s0_direction_fps_04_da3_1000:direction_fps_04+1k camera15_cap1m_s0_direction_fps_08_da3_1000:direction_fps_08+1k camera15_cap1m_s0_direction_fps_12_da3_1000:direction_fps_12+1k camera15_cap1m_s0_direction_fps_18_da3_1000:direction_fps_18+1k || echo "Figure generation failed for some runs; CSVs are still valid."
if [ "$evaluation_failed" -ne 0 ]; then
  echo "Some runs need review (skipped/failed conditions are listed with explicit status in the CSVs)."
fi
exit "$evaluation_failed"
