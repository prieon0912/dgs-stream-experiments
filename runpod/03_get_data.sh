#!/usr/bin/env bash
# Downloads N3DV flame_steak (1.2GB) and builds the 3DGStream-format dataset (undistorted, images_2).
# Run inside tmux; prep takes ~30-60 min for 300 frames (CPU bound, resumable).
set -euo pipefail
source "$(dirname "$0")/env.sh"
conda activate gs

mkdir -p "$DATA/raw"
if [ ! -f "$RAW_VIDEOS/cam00.mp4" ]; then
  wget -q --show-progress -c -O "$DATA/raw/flame_steak.zip" \
    https://github.com/facebookresearch/Neural_3D_Video/releases/download/v1.0/flame_steak.zip
  unzip -q -o "$DATA/raw/flame_steak.zip" -d "$DATA/raw"
fi
ls "$RAW_VIDEOS" | head -n 25

python "$EXP/scripts/01_prep_dataset.py" \
  --videos "$RAW_VIDEOS" --suite "$SUITE" --out "$SCENE" --n_frames "$N_FRAMES" \
  --workers "${PREP_WORKERS:-6}"
