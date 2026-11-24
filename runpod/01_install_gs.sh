#!/usr/bin/env bash
# conda env "gs": 3DGStream (+ its rasterizer fork, simple-knn, tiny-cuda-nn) and all our eval/refine scripts.
# Matches the tested config in the 3DGStream README: CUDA 11.8 + torch 2.0.1. Takes ~15-25 min (tcnn build).
set -euo pipefail
source "$(dirname "$0")/env.sh"

if ! conda env list | grep -q "^gs "; then conda create -y -q -n gs python=3.10; fi
conda activate gs

pip install -q torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
# torch 2.0.1's cpp_extension does `from pkg_resources import packaging`:
#   setuptools >= 70 drops pkg_resources.packaging, >= 81 drops pkg_resources -> CUDA extension builds fail
pip install -q "setuptools<70" wheel
pip install -q "numpy<2" plyfile tqdm commentjson torchviz tensorboard opencv-python-headless \
  scipy lpips matplotlib pandas tabulate pillow ninja pycolmap "kornia<0.8"   # kornia: imported by 3DGStream train.py

# --- 3DGStream (no --recursive: SIBR_viewers is huge and not needed)
if [ ! -d "$REPO_GS" ]; then git clone https://github.com/SJoJoK/3DGStream "$REPO_GS"; fi
cd "$REPO_GS"
if [ -f "$VERSIONS_LOCK" ]; then
  GS_REV=$(python "$EXP/scripts/20_version_lock.py" --lock "$VERSIONS_LOCK" --key gs_commit)
  git checkout --detach "$GS_REV"
fi
git submodule update --init --recursive submodules/diff-gaussian-rasterization
git submodule update --init submodules/simple-knn || {
  echo "gitlab.inria.fr unreachable -> using github mirror for simple-knn"
  rm -rf submodules/simple-knn
  git clone https://github.com/camenduru/simple-knn submodules/simple-knn; }

GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
export TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1)"
export TCNN_CUDA_ARCHITECTURES=$GPU_ARCH
echo "building for sm_$GPU_ARCH"

pip install --no-build-isolation submodules/diff-gaussian-rasterization   # 3DGStream fork (returns img, radii, depth)
pip install --no-build-isolation submodules/simple-knn

# tiny-cuda-nn: "1.7" in the 3DGStream README is the package version, there is no v1.7 git tag
# (tags: v1.1..v1.6, v2.0) -> install master
TCNN_SOURCE="git+https://github.com/NVlabs/tiny-cuda-nn#subdirectory=bindings/torch"
if [ -f "$VERSIONS_LOCK" ]; then
  TCNN_REV=$(python "$EXP/scripts/20_version_lock.py" --lock "$VERSIONS_LOCK" --key tcnn_commit)
  TCNN_SOURCE="git+https://github.com/NVlabs/tiny-cuda-nn@$TCNN_REV#subdirectory=bindings/torch"
fi
pip install --no-build-isolation "$TCNN_SOURCE"

python - <<'EOF'
import torch, tinycudann, diff_gaussian_rasterization, simple_knn._C, lpips
from diff_gaussian_rasterization import GaussianRasterizationSettings
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
print("raster settings fields:", GaussianRasterizationSettings._fields)
lpips.LPIPS(net="vgg", verbose=False)  # pre-download weights into $TORCH_HOME
print("gs env OK")
EOF
