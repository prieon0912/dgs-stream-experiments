#!/usr/bin/env bash
# Third experiment: environments for the alternative first-frame initializers.
#   bash runpod/04_install_alt_init.sh            # all three (about 30-45 min, mostly CUDA builds + weights)
#   bash runpod/04_install_alt_init.sh taming     # or: edgs / zipsplat
#
# One conda env per method (taming / edgs / zipsplat). Nothing is installed into "gs" or "da3":
# Taming and EDGS each ship their own diff_gaussian_rasterization + simple_knn builds with the same
# Python package names as 3DGStream's fork, so they must never share an env with it.
#   taming   : python 3.10, torch 2.0.1+cu118 (same torch as env gs / our 3DGS 15k baseline -> fair timing)
#   edgs     : python 3.10, torch 2.4.1+cu118 (RoMa + current graphdeco code; same torch as env da3)
#   zipsplat : python 3.10, torch 2.4.1+cu118 + xformers 0.0.28.post1 (DA3-Giant backbone, as env da3)
# All builds use the pod's nvcc 11.8. Repos are checked out at the commits in alt_init/alt_env.sh.
# Re-running is safe: finished steps are skipped, a finished env is only smoke-tested again.
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$EXP/alt_init/alt_env.sh"
mkdir -p "$ALT" "$ALT_RUNS" "$TORCH_HOME" "$HF_HOME"
WHICH=${1:-all}

command -v nvcc >/dev/null || { echo "ERROR: nvcc not found"; exit 1; }
nvcc --version | grep -q "release 11.8" || { echo "ERROR: nvcc is not CUDA 11.8"; exit 1; }
export TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1)"
export MAX_JOBS=${MAX_JOBS:-8}
echo "building for sm $TORCH_CUDA_ARCH_LIST"

checkout() {   # repo_url dir commit
  local url=$1 dir=$2 rev=$3
  if [ ! -d "$dir/.git" ]; then git clone "$url" "$dir"; fi
  git -C "$dir" fetch -q origin "$rev" 2>/dev/null || git -C "$dir" fetch -q origin
  git -C "$dir" checkout -q --detach "$rev"
  echo "$dir @ $(git -C "$dir" rev-parse HEAD)"
}

new_env() {    # name torch_spec...
  local name=$1; shift
  if ! conda env list | grep -q "^$name "; then conda create -y -q -n "$name" python=3.10; fi
  conda activate "$name"
  python -c "import torch" 2>/dev/null || pip install -q "$@" --index-url https://download.pytorch.org/whl/cu118
  # torch 2.0.1 cpp_extension needs pkg_resources.packaging (setuptools < 70)
  pip install -q "setuptools<70" wheel ninja "numpy<2"
}

install_taming() {
  echo "=== taming"
  checkout https://github.com/humansensinglab/taming-3dgs.git "$TAMING_ROOT" "$TAMING_COMMIT"
  git -C "$TAMING_ROOT" submodule update --init --recursive
  new_env taming torch==2.0.1 torchvision==0.15.2
  pip install -q plyfile tqdm websockets "opencv-python-headless<4.11" pillow "scipy<1.14" tensorboard
  # torch 2.0.1 is built against numpy 1.x: anything that pulled numpy 2 breaks torch.from_numpy
  # ("RuntimeError: Numpy is not available" in the first run) -> pin it again after all other installs
  pip install -q "numpy<2"
  python -c "import numpy, torch; torch.from_numpy(numpy.zeros(2)); print('numpy', numpy.__version__, 'ok with torch', torch.__version__)"
  python -c "import diff_gaussian_rasterization" 2>/dev/null || pip install --no-build-isolation "$TAMING_ROOT/submodules/diff-gaussian-rasterization"
  python -c "import simple_knn._C" 2>/dev/null || pip install --no-build-isolation "$TAMING_ROOT/submodules/simple-knn"
  python -c "import fused_ssim" 2>/dev/null || pip install --no-build-isolation "$TAMING_ROOT/submodules/fused-ssim"
  (cd "$TAMING_ROOT" && python - <<'EOF'
import torch, diff_gaussian_rasterization, simple_knn._C, fused_ssim
import diff_gaussian_rasterization as d
print("taming: torch", torch.__version__, "| rasterizer from", d.__file__)
from lpipsPyTorch import lpips   # imported by train.py; vgg weights download on first use
EOF
  )
  conda deactivate
}

install_edgs() {
  echo "=== edgs"
  checkout https://github.com/CompVis/EDGS.git "$EDGS_ROOT" "$EDGS_COMMIT"
  # .gitmodules uses git@github.com (needs an ssh key) -> https; gitlab.inria.fr may be down -> github mirror
  git -C "$EDGS_ROOT" config submodule.submodules/RoMa.url https://github.com/Parskatt/RoMa.git
  git -C "$EDGS_ROOT" submodule update --init submodules/RoMa submodules/gaussian-splatting
  GSD=$EDGS_ROOT/submodules/gaussian-splatting
  git -C "$GSD" submodule update --init --recursive submodules/diff-gaussian-rasterization
  git -C "$GSD" submodule update --init submodules/simple-knn 2>/dev/null || {
    echo "gitlab.inria.fr unreachable -> github mirror for simple-knn (same as env gs)"
    rm -rf "$GSD/submodules/simple-knn"
    git clone -q https://github.com/camenduru/simple-knn "$GSD/submodules/simple-knn"; }
  echo "submodules:"; git -C "$EDGS_ROOT" submodule status
  new_env edgs torch==2.4.1 torchvision==0.19.1
  pip install -q hydra-core omegaconf wandb tqdm torchmetrics lpips matplotlib rich plyfile imageio imageio-ffmpeg \
    scipy scikit-learn plotly opencv-python-headless pycolmap \
    einops kornia albumentations loguru h5py timm poselib
  python -c "import diff_gaussian_rasterization" 2>/dev/null || pip install --no-build-isolation "$GSD/submodules/diff-gaussian-rasterization"
  python -c "import simple_knn._C" 2>/dev/null || pip install --no-build-isolation "$GSD/submodules/simple-knn"
  python -c "import romatch" 2>/dev/null || pip install --no-deps -e "$EDGS_ROOT/submodules/RoMa"
  (cd "$EDGS_ROOT" && python - <<'EOF'
import sys, torch
sys.path.insert(0, "./submodules/gaussian-splatting")
import diff_gaussian_rasterization as d, simple_knn._C, hydra, romatch
from romatch import roma_outdoor
from scene import Scene, GaussianModel                     # graphdeco code inside EDGS
from source.trainer import EDGSTrainer                     # pulls lpips, rich, wandb, corr_init
m = roma_outdoor(device="cuda")                            # downloads RoMa + DINOv2 weights into $TORCH_HOME
import lpips; lpips.LPIPS(net="vgg", verbose=False)       # EDGSTrainer builds this
print("edgs: torch", torch.__version__, "| rasterizer from", d.__file__, "| RoMa OK")
EOF
  )
  conda deactivate
}

install_zipsplat() {
  echo "=== zipsplat"
  checkout https://github.com/cvg/ZipSplat.git "$ZIPSPLAT_ROOT" "$ZIPSPLAT_COMMIT"
  new_env zipsplat torch==2.4.1 torchvision==0.19.1 xformers==0.0.28.post1
  # its requirements.txt is unpinned ("torch", "gsplat") -> install without deps so pip cannot swap torch
  pip install -q einops tqdm matplotlib plyfile pillow "imageio[ffmpeg]" "triton==3.0.0"
  # gsplat: prebuilt wheel for torch 2.4 + cu118 first; else build the commit env da3 uses
  python -c "import gsplat" 2>/dev/null || pip install -q gsplat==1.4.0 --index-url https://docs.gsplat.studio/whl/pt24cu118 \
    || pip install --no-build-isolation "git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70"
  python -c "import zipsplat" 2>/dev/null || pip install --no-deps -e "$ZIPSPLAT_ROOT"
  python - <<'EOF'
import torch, gsplat
from gsplat.rendering import rasterization
# gsplat compiles / loads its CUDA kernels on first use -> do it here, not inside the timed run
means = torch.zeros(1, 3, device="cuda"); means[0, 2] = 2
quats = torch.tensor([[1.0, 0, 0, 0]], device="cuda"); scales = torch.full((1, 3), 0.1, device="cuda")
K = torch.tensor([[[50.0, 0, 32], [0, 50.0, 32], [0, 0, 1]]], device="cuda")
img, _, _ = rasterization(means, quats, scales, torch.ones(1, device="cuda"), torch.ones(1, 3, device="cuda"),
                          torch.eye(4, device="cuda")[None], K, 64, 64)
from zipsplat import ZipSplat
m = ZipSplat(weights="zipsplat")          # downloads the 504 px checkpoint (DA3-Giant backbone) into $TORCH_HOME
print("zipsplat: torch", torch.__version__, "gsplat", gsplat.__version__, "| render", float(img.sum()) > 0, "| weights OK")
EOF
  conda deactivate
}

case $WHICH in
  taming) install_taming;;
  edgs) install_edgs;;
  zipsplat) install_zipsplat;;
  all) install_taming; install_edgs; install_zipsplat;;
  *) echo "usage: $0 [all|taming|edgs|zipsplat]"; exit 1;;
esac
(conda activate gs && python "$EXP/scripts/16_snapshot_environment.py" --out "$ALT_RUNS/environment_gs.json") || true
for e in taming edgs zipsplat; do
  conda env list | grep -q "^$e " && conda list -n "$e" --export > "$ALT_RUNS/conda_$e.txt" || true
done
echo "alt-init envs ready ($WHICH). next: bash jobs/alt_init_s0/10_alt_init_screen.sh"
