#!/usr/bin/env bash
# One command after the zip is in /workspace on a fresh pod:
#   python3 -m zipfile -e /workspace/dgs_exp_v5_prune_update.zip /workspace/ && bash /workspace/dgs_exp_v5/runpod/bootstrap_new_pod.sh
# Installs tmux, then starts the driver detached (survives closing the browser). Log: /workspace/prune_update.log
set -euo pipefail
cd /workspace
command -v tmux >/dev/null || { export DEBIAN_FRONTEND=noninteractive; apt-get update -qq; apt-get install -y -qq tmux > /dev/null; }
nvcc --version | grep -q "release 11.8" || { echo "ERROR: nvcc is not CUDA 11.8 -> wrong container image"; exit 1; }
if tmux has-session -t prune 2>/dev/null; then echo "tmux session 'prune' already running"; exit 0; fi
tmux new-session -d -s prune \
  "bash /workspace/dgs_exp_v5/runpod/run_prune_update.sh 2>&1 | tee -a /workspace/prune_update.log; exec bash"
sleep 2
tmux ls
echo "started; follow with: tail -f /workspace/prune_update.log   stage: cat /workspace/prune_update.stage"
