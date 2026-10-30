"""Fail when any deterministic eval assertion fails; report the rest.

Run with: uv run python scripts/eval_gate.py evals/promptfoo/output/results.json
Hard gates, which must all pass in every run of every case: claims re-verified by verify.py,
the insufficient_data flag, the latest-year check, and refusal plus guardrails_include.
Report-only, shown but never failing the build: the llm-rubric (a model's judgement, so
noisy), latency (it depends on the network and the provider, not on the code), and the
guardrails a case marks guardrails_report_only because they fire only sometimes.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HARD_GATES = ("claims", "insufficient_data", "latest_year", "guardrails")
REPORT_ONLY = ("rubric", "latency", "guardrails_report")


def _components(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-assertion results; promptfoo sets gradingResult to null when the provider errored."""
    return list((result.get("gradingResult") or {}).get("componentResults", []))


def case_results(results: dict[str, Any]) -> list[dict[str, Any]]:
    return list(results["results"]["results"])


def failed_metrics(result: dict[str, Any], metrics: tuple[str, ...]) -> list[str]:
    """The named assertions (by metric) that failed in one run of one case."""
    return [
        c["assertion"]["metric"]
        for c in _components(result)
        if not c["pass"] and c.get("assertion", {}).get("metric") in metrics
    ]


def _count(results: list[dict[str, Any]], metric: str) -> tuple[int, int]:
    """(passed, total) for one metric over every run that has that assertion."""
    seen = [
        c for r in results for c in _components(r) if c.get("assertion", {}).get("metric") == metric
    ]
    return sum(bool(c["pass"]) for c in seen), len(seen)


def hard_failures(results: list[dict[str, Any]]) -> dict[str, list[list[str]]]:
    """Case description -> the failed hard gates of each failing run. Empty when all pass.

    A run that errored before producing an answer has no assertion results at all, so it
    counts as failing every hard gate rather than as a pass.
    """
    failures: dict[str, list[list[str]]] = defaultdict(list)
    for r in results:
        name = r.get("testCase", {}).get("description", "?")
        failed = failed_metrics(r, HARD_GATES)
        if not _components(r):
            failed = list(HARD_GATES)
        if failed:
            failures[name].append(failed)
    return dict(failures)


def report(results: list[dict[str, Any]]) -> tuple[list[str], bool]:
    """The report lines, and whether every hard gate passed."""
    if not results:
        return ["no cases ran"], False
    failures = hard_failures(results)
    bad_runs = sum(len(runs) for runs in failures.values())
    lines = [f"hard gates: {len(results) - bad_runs}/{len(results)} runs passed all of them"]
    for name, runs in failures.items():
        metrics = sorted({m for run in runs for m in run})
        lines.append(f"  FAILED {len(runs)} run(s): {name}  [{', '.join(metrics)}]")
    for metric in REPORT_ONLY:
        passed, total = _count(results, metric)
        lines.append(f"report only: {metric} {passed}/{total} passed")
        for r in results:
            if failed_metrics(r, (metric,)):
                lines.append(f"  {metric}: {r.get('testCase', {}).get('description', '?')}")
    return lines, not failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args(argv)
    lines, ok = report(case_results(json.loads(args.results.read_text())))
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
