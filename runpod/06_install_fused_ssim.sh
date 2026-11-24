#!/usr/bin/env bash
# [env: gs] fused-ssim (rahul-goel/fused-ssim) for --fused_ssim on (stream/train_frames_select.py).
#   bash runpod/06_install_fused_ssim.sh
# Pinned commit (HEAD of 2026-02-02). Upstream lists torch >= 2.3.1 as tested; the gs env is torch 2.0.1 + cu118,
# so this script builds it and runs a forward/backward smoke test. If the build fails, report the log:
# Phase 1 (fused SSIM) stops there (SPEC_backward_accel.md). Not added to versions.lock.json until it is validated.
set -euo pipefail
source "$(dirname "$0")/env.sh"
conda activate gs

FUSED_SSIM_COMMIT=${FUSED_SSIM_COMMIT:-a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38}
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')   # A6000 -> 86
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
nvcc --version | tail -n2
echo "building fused-ssim $FUSED_SSIM_COMMIT for sm_$ARCH"
# setup.py reads CUDA_ARCHITECTURES ("86"); without it, it probes the current device
CUDA_ARCHITECTURES=$ARCH pip install --no-build-isolation -v \
  "git+https://github.com/rahul-goel/fused-ssim@$FUSED_SSIM_COMMIT" 2>&1 | tail -n 20

python - <<'EOF'
import torch
from fused_ssim import fused_ssim
a = torch.rand(1, 3, 64, 80, device="cuda", requires_grad=True)
b = torch.rand(1, 3, 64, 80, device="cuda")
v = fused_ssim(a, b, padding="same", train=True)
v.backward()
assert torch.isfinite(v) and a.grad is not None and torch.isfinite(a.grad).all()
print("fused_ssim OK", float(v))
EOF
echo "next: cd \$REPO_GS && python \$EXP/tools/ssim_check.py --scene \$SCENE   (identity check vs utils.loss_utils.ssim)"
