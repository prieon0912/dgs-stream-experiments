# source this in every new terminal:  source /workspace/dgs_exp/runpod/env.sh
# RunPod: only /workspace (volume) survives pod stop/restart. Everything big lives there.

export EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # this repo (upload to /workspace/dgs_exp)
export WS=${WS:-/workspace}

export REPO_GS=$WS/3DGStream
export REPO_DA3=$WS/Depth-Anything-3
export SUITE=$REPO_GS/test/flame_steak_suite
export BASE_INIT_PLY=$SUITE/flame_steak_init/point_cloud/iteration_15000/point_cloud.ply

export DATA=$WS/data
export RAW_VIDEOS=$DATA/raw/flame_steak          # cam00.mp4 ... cam20.mp4
export SCENE=$DATA/flame_steak                    # 3DGStream format: frame000000 ... frame000299
export FRAME0=$SCENE/frame000000
export RUNS=${RUNS:-$WS/runs}

export N_FRAMES=${N_FRAMES:-300}                  # 150 if time is short (applies to ALL inits)
export DA3_MODEL=${DA3_MODEL:-depth-anything/DA3-GIANT-1.1}

# caches -> volume, so re-downloads don't happen after pod restart
export CONDA_ROOT=$WS/miniconda3
export HF_HOME=$WS/.cache/huggingface
export TORCH_HOME=$WS/.cache/torch
export PIP_CACHE_DIR=$WS/.cache/pip
export QT_QPA_PLATFORM=offscreen                  # headless colmap
export PYTHONUNBUFFERED=1
export PYTHONPATH="$EXP${PYTHONPATH:+:$PYTHONPATH}"
export AABB_CONFIG=${AABB_CONFIG:-$EXP/configs/aabb_original.json}
export VERSIONS_LOCK=${VERSIONS_LOCK:-$EXP/runpod/versions.lock.json}

if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
fi
gs_env()  { conda activate gs; }
da3_env() { conda activate da3; }

mkdir -p "$DATA" "$RUNS" 2>/dev/null || true
