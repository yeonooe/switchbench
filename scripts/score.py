"""Score an episodes.jsonl file (e.g. a participant submission or a partial run).

    python scripts/score.py results/ellsa_dev/episodes.jsonl
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from switchbench import metrics  # noqa: E402

for path in sys.argv[1:]:
    print(metrics.to_markdown(metrics.summarize(metrics.load_records(path)), path), "\n")
