"""Fail unless enough eval cases passed.

Run with: uv run python scripts/eval_gate.py evals/promptfoo/output/results.json [--threshold 0.9]
A case passes only when every one of its assertions passes. Exits 1 below the threshold.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_THRESHOLD = 0.9


def case_results(results: dict[str, Any]) -> list[dict[str, Any]]:
    return list(results["results"]["results"])


def summarize(cases: list[dict[str, Any]]) -> tuple[float, list[str]]:
    """(pass rate, descriptions of failed cases). An empty run is a failure, not a pass."""
    if not cases:
        return 0.0, ["no cases ran"]
    failed = [c.get("testCase", {}).get("description", "?") for c in cases if not c["success"]]
    return (len(cases) - len(failed)) / len(cases), failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args(argv)
    rate, failed = summarize(case_results(json.loads(args.results.read_text())))
    print(f"pass rate: {rate:.1%} (threshold {args.threshold:.0%})")
    for description in failed:
        print(f"  FAILED: {description}")
    return 0 if rate >= args.threshold else 1


if __name__ == "__main__":
    sys.exit(main())
