#!/usr/bin/env bash
# conda env "da3": Depth Anything 3 (Giant, GS head). Separate env so torch/xformers can't break 3DGStream.
# torch/xformers pinned to cu118 wheels so gsplat builds with the pod's nvcc 11.8.
set -euo pipefail
source "$(dirname "$0")/env.sh"

if ! conda env list | grep -q "^da3 "; then conda create -y -q -n da3 python=3.10; fi
conda activate da3

pip install -q torch==2.4.1 torchvision==0.19.1 xformers==0.0.28.post1 --index-url https://download.pytorch.org/whl/cu118
if [ ! -d "$REPO_DA3" ]; then git clone https://github.com/ByteDance-Seed/Depth-Anything-3 "$REPO_DA3"; fi
if [ -f "$VERSIONS_LOCK" ]; then
  DA3_REV=$(python "$EXP/scripts/20_version_lock.py" --lock "$VERSIONS_LOCK" --key da3_commit)
  git -C "$REPO_DA3" checkout --detach "$DA3_REV"
fi
pip install -q -e "$REPO_DA3"
pip install -q "numpy<2" scipy plyfile pillow "setuptools<70" wheel   # pkg_resources for CUDA ext builds
# imported by DA3 core code but missing from its pyproject dependencies
pip install -q addict matplotlib tqdm
export TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1)"
pip install --no-build-isolation "git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70" \
  || echo "WARN: gsplat build failed. Our pipeline reads prediction.gaussians directly; only DA3's own gs export/video needs gsplat."

# pre-download weights into $HF_HOME (volume) so pod restarts don't re-download ~5GB
python - <<EOF
import torch
from depth_anything_3.api import DepthAnything3
m = DepthAnything3.from_pretrained("$DA3_MODEL")
print("loaded", "$DA3_MODEL", sum(p.numel() for p in m.parameters())/1e9, "B params")
print("torch", torch.__version__, torch.cuda.get_device_name(0))
EOF
echo "da3 env OK"
