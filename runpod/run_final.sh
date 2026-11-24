#!/usr/bin/env bash
# Final experiment pods (A6000, each with its own /workspace disk).
#   each pod (own disk): bash runpod/run_final.sh setup <label> <BE|AC|ALL>
#   (join <label> <role> = only wait for env.ready then start the worker; for pods sharing a volume)
# Expects on the volume (moved there beforehand):
#   /workspace/final/inits/{opacity1m,redundancy600k,reproj600k,zipsplat05}.ply (+ _timing.json)
#   /workspace/data/flame_steak/frame000000..frame000299/{images_2,sparse}
#   /workspace/final/suite/flame_steak_suite  (copied into the 3DGStream checkout after cloning)
# Progress: /workspace/final/worker_<label>.log, state/, results/. The pod is never stopped by this script.
set -uo pipefail
ROLE=${1:?setup|join}; LABEL=${2:-$(hostname)}; WROLE=${3:-ALL}
EXPDIR=$(cd "$(dirname "$0")/.." && pwd)
export WS=/workspace
mkdir -p $WS/final
L=$WS/final/setup_$LABEL.log
exec > >(tee -a "$L") 2>&1
echo "== $(date -Is) $ROLE $LABEL exp=$EXPDIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

apt_part() {
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq rsync tmux unzip wget git > /dev/null || echo "WARN apt"
}

if [ "$ROLE" = setup ]; then
  apt_part
  bash "$EXPDIR/runpod/00_setup_pod.sh" || { echo "setup_pod failed"; exit 1; }
  if [ ! -f $WS/final/env.ready ]; then
    bash "$EXPDIR/runpod/01_install_gs.sh" || { echo "install_gs failed"; touch $WS/final/env.FAILED; exit 1; }
  fi
  source "$EXPDIR/runpod/env.sh"
  if [ ! -f "$SUITE/cfg_args.json" ]; then
    mkdir -p "$(dirname "$SUITE")" && cp -r $WS/final/suite/flame_steak_suite "$SUITE"
  fi
  # data check (fail loudly here rather than inside 16 runs)
  python - <<'PYEOF' || { touch $WS/final/env.FAILED; exit 1; }
import os, glob, sys
ws = "/workspace"
need = [f"{ws}/final/inits/{c}.ply" for c in ("opacity1m", "redundancy600k", "reproj600k", "zipsplat05")]
miss = [p for p in need if not os.path.isfile(p)]
frames = sorted(glob.glob(f"{ws}/data/flame_steak/frame0*"))
bad = [f for f in frames if len(glob.glob(f + "/images_2/*.png")) < 21]
suite = os.environ.get("SUITE", f"{ws}/3DGStream/test/flame_steak_suite")
print("inits missing:", miss, "| frames:", len(frames), "| frames with <21 images:", len(bad),
      "| suite cfg:", os.path.isfile(f"{suite}/cfg_args.json"))
sys.exit(1 if miss or len(frames) < 300 or bad or not os.path.isfile(f"{suite}/cfg_args.json") else 0)
PYEOF
  date -Is > $WS/final/env.ready
else
  echo "waiting for env.ready"
  while [ ! -f $WS/final/env.ready ]; do
    [ -f $WS/final/env.FAILED ] && { echo "setup pod failed"; exit 1; }
    sleep 30
  done
  apt_part
  bash "$EXPDIR/runpod/00_setup_pod.sh" || true
fi
echo "== $(date -Is) starting worker $LABEL"
setsid nohup bash "$EXPDIR/jobs/final_s0/worker.sh" "$LABEL" "$WROLE" > $WS/final/worker_${LABEL}.out 2>&1 &
echo "worker pid $!"
