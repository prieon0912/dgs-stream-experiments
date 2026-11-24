#!/usr/bin/env bash
# Run on EVERY pod start (apt packages live on the container disk and vanish on restart).
# Idempotent: miniconda is installed into /workspace only once.
set -euo pipefail
source "$(dirname "$0")/env.sh"

echo "== apt (tmux, ffmpeg, colmap, misc)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq tmux ffmpeg colmap unzip wget git htop nvtop > /dev/null || \
  apt-get install -y -qq tmux ffmpeg unzip wget git htop > /dev/null
command -v colmap >/dev/null && echo "colmap: $(command -v colmap)" || echo "WARN: colmap not installed -> prep will fall back to pycolmap"

echo "== conda (Miniforge / conda-forge) in $CONDA_ROOT"
if [ ! -x "$CONDA_ROOT/bin/conda" ]; then
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
  bash /tmp/miniforge.sh -b -p "$CONDA_ROOT"
  "$CONDA_ROOT/bin/conda" config --set auto_activate_base false
fi

echo "== GPU / CUDA / disk"
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv
if command -v nvcc >/dev/null; then nvcc --version | tail -n1; else
  echo "ERROR: nvcc not found. Use a *devel* template, e.g. runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"; fi
df -h "$WS" | tail -n1
FREE_GB=$(df -BG --output=avail "$WS" | tail -n1 | tr -dc '0-9')
if [ "${FREE_GB:-0}" -lt 60 ]; then
  echo "WARN: /workspace free ${FREE_GB}GB < 60GB. Use N_FRAMES=150 or enlarge the volume."; fi

echo "done. next: bash runpod/01_install_gs.sh  &  bash runpod/02_install_da3.sh  (in separate tmux windows)"
