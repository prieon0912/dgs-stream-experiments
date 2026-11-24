"""GPU metrics for audited streams. See common/evaluate_stream.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.evaluate_stream import main

if __name__ == "__main__":
    main()
