#!/usr/bin/env bash
# Make sure the uncapped DA3 all-views PLY exists ($SRC_PLY). Skips if it is already there
# (e.g. copied from the old pod). Otherwise: DA3 inference + conversion with exactly the settings
# of the camera experiment (camera15_cap1m_s0 shared_all_views).
set -eEuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
mkdir -p "$SRC_DIR"
if [ -f "$SRC_PLY" ]; then
  echo "source PLY present: $SRC_PLY"
else
  conda activate da3
  [ -f "$SRC_DIR/raw.pt" ] || python "$EXP/scripts/02_run_da3.py" --frame_dir "$FRAME0" \
    --views-file "$JOB_DIR/views/all_views.json" --process_res 504 --out "$SRC_DIR/raw.pt"
  conda activate gs
  python "$EXP/scripts/03_convert_da3.py" --raw "$SRC_DIR/raw.pt" --out "$SRC_PLY" \
    --border 0.03125 --depth_prune 1.0 --min_opacity 0.005 --sh1 keep --max-align-residual 0.02 --save-sources
fi
conda activate gs
python - "$SRC_PLY" <<'EOF'
import sys
from common.ply_io import load_ply
n = len(load_ply(sys.argv[1])["xyz"])
print(f"source gaussians: {n:,}")
if n < 1_000_000:
    sys.exit("source has fewer than 1M gaussians: the 1M condition would not prune anything")
EOF
