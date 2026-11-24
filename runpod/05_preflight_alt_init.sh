#!/usr/bin/env bash
# Checks before spending GPU time on the third experiment. Prints every problem, exits 1 if any is fatal.
#   bash runpod/05_preflight_alt_init.sh
set -uo pipefail
source "$(dirname "$0")/env.sh"
source "$EXP/alt_init/alt_env.sh"
source "$EXP/jobs/alt_init_s0/config.sh"
FATAL=0
bad()  { echo "FAIL: $*"; FATAL=1; }
warn() { echo "WARN: $*"; }
ok()   { echo "ok:   $*"; }

# GPU: a pod started "with CPUs only" (GPU taken while stopped) has no usable GPU
if nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >/tmp/gpu.txt 2>/dev/null && [ -s /tmp/gpu.txt ]; then
  ok "GPU $(head -n1 /tmp/gpu.txt)"
  others=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
  [ "${others:-0}" -gt 0 ] && warn "$others other GPU process(es) running -> timings would not be exclusive"
else
  bad "no GPU visible (pod started on CPU, or GPU taken) -> migrate / pick another GPU before running"
fi
command -v nvcc >/dev/null && nvcc --version | grep -q "release 11.8" && ok "nvcc 11.8" || bad "nvcc 11.8 not found"

# data from experiments 1/2 on this volume
for f in cameras.bin images.bin points3D.bin; do
  [ -f "$FRAME0/sparse/0/$f" ] || [ -f "$FRAME0/sparse/$f" ] || bad "missing $FRAME0/sparse/0/$f"
done
n_img=$(ls "$FRAME0/images_2" 2>/dev/null | grep -c . || true)
[ "$n_img" -ge 21 ] && ok "$n_img images in images_2 (cam00 = test, rest = train)" || bad "only $n_img images in $FRAME0/images_2 (expected 21)"
[ -f "$FRAME0/images_2/cam00.png" ] || ls "$FRAME0/images_2" | grep -q '^cam00\.' || bad "cam00 image missing (test view)"
[ -x "$CONDA_ROOT/envs/gs/bin/python" ] && ok "env gs present" || bad "env gs missing (run runpod/01_install_gs.sh)"
[ -f "$REF_DIR/static/static.json" ] && ok "reference DA3 opacity 1M + 1k found" \
  || warn "reference $REF_DIR/static/static.json missing -> no 0.1/0.3/0.5 dB verdicts"
[ -f "$REF_DIR/refined_timing.json" ] || warn "reference timing missing"
[ -f "$AABB_CONFIG" ] && ok "AABB $AABB_CONFIG" || bad "missing $AABB_CONFIG"

# disk: three envs (~6-8 GB each) + ZipSplat (DA3-Giant) / RoMa / DINOv2 weights + outputs
free=$(df -BG --output=avail "$WS" | tail -n1 | tr -dc '0-9')
[ "${free:-0}" -ge 40 ] && ok "${free} GB free on $WS" || bad "only ${free} GB free on $WS (need about 40 GB)"

# network: package indexes, GitHub, Hugging Face (weights), PyTorch wheels
for url in https://github.com https://huggingface.co https://download.pytorch.org/whl/cu118 https://pypi.org/simple/ https://docs.gsplat.studio/whl/pt24cu118; do
  curl -fsSI --max-time 15 "$url" >/dev/null 2>&1 && ok "reach $url" || warn "cannot reach $url"
done

# old envs with clashing CUDA extension names must not be touched
for e in taming edgs zipsplat; do conda env list 2>/dev/null | grep -q "^$e " && ok "env $e already exists (will be reused)"; done
[ $FATAL = 0 ] && echo "preflight OK" || echo "preflight FAILED"
exit $FATAL
