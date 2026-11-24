"""Read a validated revision from a lock produced after a working installation."""
import argparse
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--key", choices=["gs_commit", "da3_commit", "tcnn_commit"], required=True)
    args = parser.parse_args()
    value = json.loads(Path(args.lock).read_text(encoding="utf-8-sig"))[args.key]
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", value):
        raise SystemExit("Invalid locked commit ID")
    print(value)


if __name__ == "__main__":
    main()
