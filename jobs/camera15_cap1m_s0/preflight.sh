#!/usr/bin/env bash
set -eEuo pipefail
JOB_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${EXP:-/workspace/dgs_exp}/runpod/env.sh"
export AABB_CONFIG="$JOB_DIR/aabb.json"
export N_FRAMES=300
cd "$EXP"
# settings_sha256: d0ab6c1ecaa76010f68cfc9d8d14882511013f3e10d10d4d9ccf2fc682821821
conda activate gs
python "$EXP/scripts/02_run_da3.py" --frame_dir /workspace/data/flame_steak/frame000000 --views-file "$JOB_DIR/views/all_views.json" --out /tmp/camera15_cap1m_s0_all_views.pt --dry-run
python "$EXP/scripts/02_run_da3.py" --frame_dir /workspace/data/flame_steak/frame000000 --views-file "$JOB_DIR/views/random_04.json" --out /tmp/camera15_cap1m_s0_random_04.pt --dry-run
python "$EXP/scripts/02_run_da3.py" --frame_dir /workspace/data/flame_steak/frame000000 --views-file "$JOB_DIR/views/random_08.json" --out /tmp/camera15_cap1m_s0_random_08.pt --dry-run
python "$EXP/scripts/02_run_da3.py" --frame_dir /workspace/data/flame_steak/frame000000 --views-file "$JOB_DIR/views/random_12.json" --out /tmp/camera15_cap1m_s0_random_12.pt --dry-run
python "$EXP/scripts/02_run_da3.py" --frame_dir /workspace/data/flame_steak/frame000000 --views-file "$JOB_DIR/views/random_18.json" --out /tmp/camera15_cap1m_s0_random_18.pt --dry-run
python "$EXP/scripts/02_run_da3.py" --frame_dir /workspace/data/flame_steak/frame000000 --views-file "$JOB_DIR/views/position_fps_04.json" --out /tmp/camera15_cap1m_s0_position_fps_04.pt --dry-run
python "$EXP/scripts/02_run_da3.py" --frame_dir /workspace/data/flame_steak/frame000000 --views-file "$JOB_DIR/views/position_fps_08.json" --out /tmp/camera15_cap1m_s0_position_fps_08.pt --dry-run
python "$EXP/scripts/02_run_da3.py" --frame_dir /workspace/data/flame_steak/frame000000 --views-file "$JOB_DIR/views/position_fps_12.json" --out /tmp/camera15_cap1m_s0_position_fps_12.pt --dry-run
python "$EXP/scripts/02_run_da3.py" --frame_dir /workspace/data/flame_steak/frame000000 --views-file "$JOB_DIR/views/position_fps_18.json" --out /tmp/camera15_cap1m_s0_position_fps_18.pt --dry-run
python "$EXP/scripts/02_run_da3.py" --frame_dir /workspace/data/flame_steak/frame000000 --views-file "$JOB_DIR/views/direction_fps_04.json" --out /tmp/camera15_cap1m_s0_direction_fps_04.pt --dry-run
python "$EXP/scripts/02_run_da3.py" --frame_dir /workspace/data/flame_steak/frame000000 --views-file "$JOB_DIR/views/direction_fps_08.json" --out /tmp/camera15_cap1m_s0_direction_fps_08.pt --dry-run
python "$EXP/scripts/02_run_da3.py" --frame_dir /workspace/data/flame_steak/frame000000 --views-file "$JOB_DIR/views/direction_fps_12.json" --out /tmp/camera15_cap1m_s0_direction_fps_12.pt --dry-run
python "$EXP/scripts/02_run_da3.py" --frame_dir /workspace/data/flame_steak/frame000000 --views-file "$JOB_DIR/views/direction_fps_18.json" --out /tmp/camera15_cap1m_s0_direction_fps_18.pt --dry-run
