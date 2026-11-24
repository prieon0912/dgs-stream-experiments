"""Rewrite a generated run_initializations.sh so one failed condition does not stop the others.

Each condition block (a paragraph that starts with ``CURRENT_RUN=<name>``) is wrapped in a
function and executed in its own ``set -e`` subshell. A failure is recorded in
``$RUNS/stream/<name>/execution_status.json`` with state ``alignment_failed`` when the block's
DA3 alignment report says ``alignment_pass: false``, otherwise ``failed``. Conditions that
succeed are unaffected (their stage markers make reruns skip finished work).

The rewritten script exits 0 when at least one condition is ready, so the driver continues to
streaming; it exits 1 only when every condition failed. A summary is written to
``$JOB_DIR/initialization_failures.tsv``.

Usage: python scripts/22_isolate_initializations.py jobs/<job>/run_initializations.sh
"""
import re
import sys
from pathlib import Path

MARK = "# isolated-initializations: v1"


def transform(text):
    if MARK in text:
        raise SystemExit("Script is already isolated")
    paragraphs = text.rstrip("\n").split("\n\n")
    header, blocks = [], []
    for paragraph in paragraphs:
        first = paragraph.lstrip("\n").split("\n", 1)[0]
        if re.fullmatch(r"CURRENT_RUN=[A-Za-z0-9][A-Za-z0-9_-]*", first):
            blocks.append(paragraph.strip("\n"))
        elif blocks:
            raise SystemExit("Unexpected non-condition paragraph after the first condition")
        else:
            header.append(paragraph)
    if not blocks:
        raise SystemExit("No condition blocks found")
    head = "\n\n".join(header)
    if "trap on_failure ERR" not in head:
        raise SystemExit("Expected header ERR trap not found")
    head = head.replace("trap on_failure ERR", "# per-condition failures are handled by run_condition below")
    out = [head.replace("set -eEuo pipefail", "set -eEuo pipefail\n" + MARK, 1), "", r'''INIT_OK=0
INIT_FAILED=0
FAIL_LOG="$JOB_DIR/initialization_failures.tsv"
printf 'run\tstate\texit_code\talignment_report\n' > "$FAIL_LOG"
run_condition() {
  local name=$1 func=$2 alignment=${3:-} code state
  set +e
  ( set -eEuo pipefail; trap 'exit $?' ERR; "$func" )
  code=$?
  set -e
  if [ "$code" -eq 0 ]; then
    INIT_OK=$((INIT_OK + 1))
    return 0
  fi
  state=failed
  if [ -n "$alignment" ] && [ -f "$alignment" ] && \
     python -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("alignment_pass") is False else 1)' "$alignment"; then
    state=alignment_failed
  fi
  python "$EXP/scripts/19_execution_status.py" record --run-dir "$RUNS/stream/$name" \
    --stage initialization --state "$state" --exit-code "$code" || true
  printf '%s\t%s\t%s\t%s\n' "$name" "$state" "$code" "$alignment" >> "$FAIL_LOG"
  INIT_FAILED=$((INIT_FAILED + 1))
  echo "INIT_SKIPPED: $name ($state, exit $code); continuing with the next condition."
}
''']
    calls = []
    for index, block in enumerate(blocks, 1):
        name = block.split("\n", 1)[0].split("=", 1)[1]
        func = f"init_condition_{index:02d}"
        match = re.search(r"03_convert_da3\.py\" --raw \S+ --out (\S+)\.ply", block)
        if match:
            alignment = match.group(1) + "_alignment.json"
        else:
            # Conditions that reuse another condition's DA3 output (e.g. DA3+1k on the shared views).
            shared = re.search(r"--ply (\S+)/limited\.ply", block)
            alignment = shared.group(1) + "/converted_alignment.json" if shared else ""
        body = "\n".join("  " + line if line else line for line in block.split("\n"))
        out.append(f"{func}() {{\n{body}\n}}\n")
        calls.append(f"run_condition {name} {func} {alignment}".rstrip())
    out.append("\n".join(calls))
    out.append(r'''
echo "Initializations finished: ready=$INIT_OK skipped=$INIT_FAILED (details: $FAIL_LOG)"
if [ "$INIT_OK" -eq 0 ]; then
  echo "No condition produced an initialization."
  exit 1
fi
exit 0''')
    return "\n".join(out) + "\n"


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    path = Path(sys.argv[1])
    path.write_text(transform(path.read_text()), newline="\n")
    print(f"Isolated condition blocks in {path}")


if __name__ == "__main__":
    main()
