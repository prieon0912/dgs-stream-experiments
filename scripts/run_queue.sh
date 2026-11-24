#!/usr/bin/env bash
# Sequential streams with explicit per-run status and a final success/failure CSV.
set -uo pipefail
SCRIPT=$(realpath "${BASH_SOURCE[0]}")
source "$(dirname "$SCRIPT")/../runpod/env.sh"
Q=$(realpath "$1")

if [ -z "${INSIDE_QUEUE:-}" ]; then
  printf -v launch '%q ' env INSIDE_QUEUE=1 "AABB_CONFIG=$AABB_CONFIG" "N_FRAMES=$N_FRAMES" \
    bash "$SCRIPT" "$Q"
  tmux new-session -d -s "${QUEUE_SESSION:-q}" "$launch" || exit 1
  echo "Queue started. Watch while running: tmux attach -t ${QUEUE_SESSION:-q}"
  exit 0
fi

failed=0
while read -r NAME PLY MODE; do
  [[ -z "${NAME:-}" || "$NAME" == \#* ]] && continue
  if [[ ! "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then echo "Invalid run name"; failed=1; continue; fi
  if [ ! -s "$PLY" ]; then
    # Initialization was skipped (e.g. DA3 alignment failed); keep its recorded reason.
    prior=$(python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("state",""))' \
      "$RUNS/stream/$NAME/execution_status.json" 2>/dev/null || true)
    case "$prior" in
      alignment_failed|failed) ;;
      *) python "$EXP/scripts/19_execution_status.py" record --run-dir "$RUNS/stream/$NAME" \
           --stage initialization --state skipped_no_init --exit-code 1 ;;
    esac
    echo "SKIPPED: $NAME (no initialization: ${prior:-missing})"
    failed=1
    continue
  fi
  python "$EXP/scripts/19_execution_status.py" record --run-dir "$RUNS/stream/$NAME" \
    --stage streaming --state running
  if bash "$EXP/scripts/08_run_stream.sh" "$NAME" "$PLY" "${MODE:-warmup}" < /dev/null; then
    echo "COMPLETE: $NAME"
  else
    code=$?
    python "$EXP/scripts/19_execution_status.py" record --run-dir "$RUNS/stream/$NAME" \
      --stage streaming --state failed --exit-code "$code"
    echo "FAILED: $NAME (exit $code)"
    failed=1
  fi
done < "$Q"
python "$EXP/scripts/19_execution_status.py" queue --queue "$Q" --runs-dir "$RUNS/stream" \
  --out "${Q%.txt}_status.csv" || failed=1
echo "Queue finished; failures=$failed. Status: ${Q%.txt}_status.csv"
exit "$failed"
