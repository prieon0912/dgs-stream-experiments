#!/usr/bin/env bash
# [env: gs]  One 3DGStream run with a selective inside-AABB update mode.   RESUMABLE.
#
#   bash scripts/08b_run_stream_select.sh <name> <init_ply> <A|B|C> [selection options...]
#
#   Copy of 08_run_stream.sh (unchanged: AABB binding, NTC warm-up, resume, GPU log, pruning of old
#   frame PLYs). The only differences: train_frames.py is replaced by stream/train_frames_select.py
#   and the selection log / state live in the run dir:
#     $R/selection.jsonl        one line per streamed frame (counts, timings, motion quantiles)
#     $R/selection_state.pt     mode B motion history (for resume)
#     $R/selection_args.json    the selection options of this run (resume refuses different ones)
#   Options after the mode are passed to train_frames_select.py, e.g.
#     B: --b_tau_xyz 0.01 [--b_tau_rot_deg 1.0 --b_window 10 --b_warmup 10 --b_recheck 30]
#     C: [--c_kappa 2 --c_tau 25 --c_radius 48]
set -euo pipefail
source "$(dirname "$0")/../runpod/env.sh"
conda activate gs

NAME=$1; INIT_PLY=$(realpath "$2"); MODE=$3; shift 3; SEL_ARGS=("$@"); NTC_MODE=warmup
case "$MODE" in A|B|C|E|EB) ;; *) echo "mode must be A, B, C, E or EB"; exit 1;; esac
if [[ ! "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then echo "Invalid run name"; exit 1; fi
if [ "$NTC_MODE" != "official" ] && [ "$NTC_MODE" != "warmup" ]; then
  echo "NTC mode must be warmup or official"; exit 1
fi
R=$RUNS/stream/$NAME
mkdir -p "$R"
python - "$R/selection_args.json" "$MODE" "${SEL_ARGS[@]}" <<'PYEOF'
import json, os, sys
path, mode, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
want = {"mode": mode, "args": rest}
if os.path.exists(path):
    have = json.load(open(path))
    if have != want:
        sys.exit(f"selection options differ from the existing run dir: {have} vs {want}")
else:
    json.dump(want, open(path, "w"), indent=1)
PYEOF
python "$EXP/scripts/16_snapshot_environment.py" --out "$R/environment_current.json"
# Freeze bounds and input identity before reusing an NTC or pruning old snapshots.
NTC_CONF=$(python - "$SUITE/cfg_args.json" "$REPO_GS" <<'EOF'
import json, os, sys
conf = json.load(open(sys.argv[1]))["ntc_conf_path"]
print(os.path.join(sys.argv[2], conf))
EOF
)
python "$EXP/scripts/13_aabb.py" bind-run --run-dir "$R" --aabb "$AABB_CONFIG" \
  --ply "$INIT_PLY" --ntc-config "$NTC_CONF" --ntc-mode "$NTC_MODE" --environment "$R/environment_current.json"
cp "$R/environment_current.json" "$R/environment_gs.json"
mkdir -p "$R/init/point_cloud/iteration_15000"
cp -n "$INIT_PLY" "$R/init/point_cloud/iteration_15000/point_cloud.ply" || true
cmp -s "$INIT_PLY" "$R/init/point_cloud/iteration_15000/point_cloud.ply" || { echo "init differs from existing run dir $R"; exit 1; }
# init-time record travels with the run (read by 09/10)
TJ="${INIT_PLY%.ply}_timing.json"
if [ -f "$TJ" ]; then cp "$TJ" "$R/init_timing.json"; else echo '{"steps":{},"init_time_sec":null,"notes":["no timing (e.g. suite-provided init)"]}' > "$R/init_timing.json"; fi

# --- NTC
if [ "$NTC_MODE" = "official" ]; then
  NTC=$REPO_GS/ntc/flame_steak_ntc_params_F_4.pth
elif [ -f "$R/ntc.pth" ]; then
  NTC=$R/ntc.pth
else
  python "$EXP/scripts/07_ntc_warmup.py" --ply "$INIT_PLY" --out "$R/ntc.pth" \
    --conf "$NTC_CONF" --aabb "$R/aabb.json" 2>&1 | tee "$R/ntc_warmup.log"
  NTC=$R/ntc.pth
fi
python "$EXP/scripts/13_aabb.py" inspect-ntc --ntc "$NTC" --expect "$R/aabb.json" \
  --out "$R/ntc_bounds.json"

# --- disk: a previous segment may have died on a full disk -> drop a cut-off newest frame first
python "$EXP/scripts/prune_stream.py" "$R" --repair --keep 3

# --- resume: last frame that finished stage 2
LAST=$(ls -d "$R"/out/frame* 2>/dev/null | while read d; do [ -f "$d/0_rendering2.png" ] && basename "$d"; done | sort | tail -n1 || true)
if [ -z "$LAST" ]; then
  FRAME_START=1; MODEL_PATH=$R/init
else
  N=$((10#${LAST#frame})); FRAME_START=$((N + 1)); MODEL_PATH=$R/out/$LAST
  echo "resuming after $LAST (frame_start=$FRAME_START)"
fi
if [ "$FRAME_START" -ge "$N_FRAMES" ]; then
  python "$EXP/scripts/19_execution_status.py" audit --run-dir "$R" --expected-frames "$N_FRAMES"
  python "$EXP/scripts/19_execution_status.py" record --run-dir "$R" --stage streaming --state complete --exit-code 0
  echo "already complete: $R"; exit 0
fi

python - "$SUITE/cfg_args.json" "$R/cfg_run.json" "$FRAME_START" "$N_FRAMES" "$NTC" "$REPO_GS" <<'EOF'
import json, sys, os
src, dst, fs, fe, ntc, repo = sys.argv[1:]
c = json.load(open(src))
c.update(frame_start=int(fs), frame_end=int(fe), ntc_path=ntc,
         ntc_conf_path=os.path.join(repo, c["ntc_conf_path"]), quiet=False)
c["test_iterations"] = [c["iterations"], c["iterations"] + c["iterations_s2"]]
c["save_iterations"] = [c["iterations"]]
json.dump(c, open(dst, "w"), indent=1)
EOF

# exclusivity check (streaming time is only comparable when nothing else runs on the GPU)
EXCL=$(python -c "from common.timing import gpu_exclusive as g; print(g())")
echo "{\"segment_start_frame\": $FRAME_START, \"gpu_exclusive_at_start\": \"$EXCL\", \"date\": \"$(date -Is)\"}" >> "$R/segments.jsonl"
[ "$EXCL" != "True" ] && echo "WARNING: GPU busy at start -> per-frame times of this segment are not reportable"

# GPU memory log every 2 s: "memory.used [MiB], utilization [%]" (peak memory for the table)
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -l 2 >> "$R/gpu.csv" &
SMI=$!
# full PLY per frame is ~100 MB per 1M gaussians -> keep only the newest 3 while training (CPU only, cheap)
( while sleep 120; do python "$EXP/scripts/prune_stream.py" "$R" --keep 3 > /dev/null 2>&1; done ) &
PRUNE=$!
trap "kill $SMI $PRUNE 2>/dev/null || true" EXIT

cd "$REPO_GS"
echo "=== $(date) start $NAME mode=$MODE ${SEL_ARGS[*]} frames [$FRAME_START,$N_FRAMES) model=$MODEL_PATH" | tee -a "$R/train.log"
PYTHONPATH="$REPO_GS:$EXP/stream:$PYTHONPATH" python "$EXP/stream/train_frames_select.py" \
  --update_mode "$MODE" --selection_log "$R/selection.jsonl" --selection_state "$R/selection_state.pt" \
  "${SEL_ARGS[@]}" --read_config --config_path "$R/cfg_run.json" \
  -o "$R/out" -m "$MODEL_PATH" -v "$SCENE" --image images_2 \
  --first_load_iteration 15000 2>&1 | tee -a "$R/train.log"
python "$EXP/scripts/19_execution_status.py" audit --run-dir "$R" --expected-frames "$N_FRAMES"
python "$EXP/scripts/19_execution_status.py" record --run-dir "$R" --stage streaming --state complete --exit-code 0
echo "=== $(date) done $NAME" | tee -a "$R/train.log"
