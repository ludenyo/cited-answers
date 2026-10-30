"""Send every eval question to a running /ask endpoint several times.

Purpose: put enough traces into Langfuse to make the dashboard worth looking at, check
the answers against the eval file, and save results so two runs can be compared.

    uv run --env-file .env uvicorn cited_answers.api:app          # in one terminal
    uv run --env-file .env python scripts/load_test.py            # in another
    uv run --env-file .env python scripts/load_test.py --save evals/results/after.json
    uv run python scripts/load_test.py --compare evals/results/before.json evals/results/after.json
    uv run python scripts/load_test.py --repeats 1 --only easy-01 trap-02

This makes real model calls. 20 questions x 5 repeats is 100 requests, roughly 5 to 15
minutes depending on concurrency and a few US cents on Haiku.

Two checks run on every answer. The insufficient_data flag must match the eval file. For
questions with a latest_year_check, the newest year the answer cites for that country
and indicator must equal the newest year in the database, which is why this script reads
DATABASE_URL. Without it that check is skipped and said so.
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx2
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = ROOT / "evals" / "questions_draft.yaml"

Pair = tuple[str, str]  # (country_code, indicator_code)
LatestYears = dict[Pair, int]


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    text: str
    expect_insufficient: bool
    latest_checks: tuple[Pair, ...] = ()


@dataclass(frozen=True)
class Job:
    question: Question
    run: int  # 1-based repeat number


@dataclass(frozen=True)
class Result:
    job: Job
    ok: bool
    seconds: float
    error: str | None = None
    insufficient_data: bool | None = None
    pass_rate: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    prompt_version: str | None = None
    figure_ids: tuple[str, ...] = ()
    latest_ok: bool | None = None  # None: no check for this question, or it was skipped

    @property
    def as_expected(self) -> bool | None:
        """Did the insufficient_data flag match the eval file? None when there was no answer."""
        if self.insufficient_data is None:
            return None
        return self.insufficient_data == self.job.question.expect_insufficient


# ---- questions and jobs ----------------------------------------------------------------


def _question_from_yaml(raw: dict[str, Any]) -> Question:
    checks = tuple(
        (c["country"], c["indicator"]) for c in raw["expect"].get("latest_year_check", [])
    )
    return Question(
        raw["id"],
        raw["category"],
        raw["question"],
        bool(raw["expect"]["insufficient_data"]),
        checks,
    )


def load_questions(path: Path, only: list[str] | None = None) -> list[Question]:
    questions = [_question_from_yaml(q) for q in yaml.safe_load(path.read_text())["questions"]]
    if only:
        unknown = set(only) - {q.id for q in questions}
        if unknown:
            raise ValueError(f"Unknown question ids: {sorted(unknown)}")
        questions = [q for q in questions if q.id in only]
    return questions


def build_jobs(questions: list[Question], repeats: int) -> list[Job]:
    """Round by round, so each question has run once before any runs a second time.

    That spreads each question's traces over the whole run instead of bunching them.
    """
    return [Job(q, run) for run in range(1, repeats + 1) for q in questions]


# ---- the latest-year check -------------------------------------------------------------


def cited_years(figure_ids: tuple[str, ...], pair: Pair) -> list[int]:
    """Years of the cited figures for one country and indicator. Ids look like KEN:X.Y:2023."""
    country, indicator = pair
    pattern = re.compile(rf"^{re.escape(country)}:{re.escape(indicator)}:(\d{{4}})$")
    return [int(m.group(1)) for fid in figure_ids if (m := pattern.match(fid))]


def latest_year_ok(
    question: Question, figure_ids: tuple[str, ...], latest_years: LatestYears
) -> bool | None:
    """True when, for every checked pair, the newest cited year is the newest in the database.

    Citing older years as well is fine (a trend). Citing only older years, which is what
    reporting an old value as the latest looks like, fails. None when there is nothing to check.
    """
    if not question.latest_checks or any(p not in latest_years for p in question.latest_checks):
        return None
    for pair in question.latest_checks:
        years = cited_years(figure_ids, pair)
        if not years or max(years) != latest_years[pair]:
            return False
    return True


def fetch_latest_years(questions: list[Question], database_url: str) -> LatestYears:
    """Newest year in the database for each country and indicator the questions check."""
    import psycopg

    pairs = {p for q in questions for p in q.latest_checks}
    found: LatestYears = {}
    with psycopg.connect(database_url) as conn:
        for country, indicator in sorted(pairs):
            row = conn.execute(
                "SELECT max(year) FROM figures WHERE country_code = %s AND indicator_code = %s",
                (country, indicator),
            ).fetchone()
            if row and row[0] is not None:
                found[(country, indicator)] = int(row[0])
    return found


# ---- one request -------------------------------------------------------------------------


def _figure_ids(body: dict[str, Any]) -> tuple[str, ...]:
    """Every figure the answer cites, whether directly or as a source of a derived claim."""
    ids: list[str] = []
    for claim in body.get("claims", []):
        if claim.get("figure_id"):
            ids.append(claim["figure_id"])
        ids.extend(claim.get("source_figure_ids") or [])
    return tuple(ids)


def run_job(client: httpx2.Client, job: Job) -> Result:
    """One request. Any failure becomes a failed Result; this never raises."""
    started = time.perf_counter()
    try:
        response = client.post("/ask", json={"question": job.question.text})
        elapsed = time.perf_counter() - started
        if response.status_code != 200:
            return Result(job, False, elapsed, error=f"HTTP {response.status_code}")
        body: dict[str, Any] = response.json()
        return Result(
            job,
            True,
            elapsed,
            insufficient_data=bool(body["insufficient_data"]),
            pass_rate=float(body["citation_pass_rate"]),
            input_tokens=int(body["input_tokens"]),
            output_tokens=int(body["output_tokens"]),
            prompt_version=str(body["prompt_version"]),
            figure_ids=_figure_ids(body),
        )
    except Exception as exc:  # noqa: BLE001 - a bad response must not stop the whole run
        return Result(job, False, time.perf_counter() - started, error=type(exc).__name__)


def with_latest_check(result: Result, latest_years: LatestYears) -> Result:
    if not result.ok:
        return result
    return replace(
        result, latest_ok=latest_year_ok(result.job.question, result.figure_ids, latest_years)
    )


def progress_line(done: int, total: int, result: Result) -> str:
    label = f"[{done:>{len(str(total))}}/{total}] {result.job.question.id:<12} run {result.job.run}"
    if not result.ok:
        return f"{label}  FAILED ({result.error}) after {result.seconds:.1f}s"
    flag = "as expected" if result.as_expected else "DIFFERS from eval file"
    line = (
        f"{label}  {result.seconds:5.1f}s  pass_rate={result.pass_rate:.2f}  "
        f"insufficient_data={result.insufficient_data} ({flag})"
    )
    if result.latest_ok is not None:
        line += "  latest_year=" + ("ok" if result.latest_ok else "STALE")
    return line


# ---- saving and summarising --------------------------------------------------------------


def result_to_dict(result: Result) -> dict[str, Any]:
    question = result.job.question
    return {
        "id": question.id,
        "category": question.category,
        "run": result.job.run,
        "expect_insufficient": question.expect_insufficient,
        "ok": result.ok,
        "error": result.error,
        "seconds": round(result.seconds, 2),
        "insufficient_data": result.insufficient_data,
        "as_expected": result.as_expected,
        "pass_rate": result.pass_rate,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "prompt_version": result.prompt_version,
        "figure_ids": list(result.figure_ids),
        "latest_ok": result.latest_ok,
    }


def git_commit() -> str | None:
    """Short hash of the checked-out commit, so a saved run says what code produced it."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def save_results(path: Path, results: list[Result], meta: dict[str, Any]) -> None:
    payload = {"meta": meta, "results": [result_to_dict(r) for r in results]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def _latest_cell(results: list[Result]) -> str:
    checked = [r for r in results if r.latest_ok is not None]
    return f"{sum(bool(r.latest_ok) for r in checked)}/{len(checked)}" if checked else "-"


def _row(question: Question, results: list[Result]) -> str:
    ok = [r for r in results if r.ok]
    matched = [r for r in ok if r.as_expected]
    return (
        f"{question.id:<12} {len(ok):>2}/{len(results):<2}  "
        f"{_mean([r.pass_rate or 0.0 for r in ok]):>9.2f}  "
        f"{len(matched):>2}/{len(ok):<2}  "
        f"{_latest_cell(ok):>6}  "
        f"{_mean([r.seconds for r in ok]):>8.1f}s  "
        f"{_mean([float(r.input_tokens) for r in ok]):>9.0f}  "
        f"{_mean([float(r.output_tokens) for r in ok]):>9.0f}"
    )


def summarize(questions: list[Question], results: list[Result]) -> str:
    """A per-question table, then totals."""
    lines = [
        f"{'question':<12} {'ok':<5}  {'pass_rate':>9}  {'flag ok':<5}  {'latest':>6}  "
        f"{'mean time':>9}  {'tokens in':>9}  {'tokens out':>9}"
    ]
    for question in questions:
        lines.append(_row(question, [r for r in results if r.job.question.id == question.id]))
    ok = [r for r in results if r.ok]
    versions = sorted({r.prompt_version for r in ok if r.prompt_version})
    matched = sum(bool(r.as_expected) for r in ok)
    lines += [
        "",
        f"requests: {len(results)}  ok: {len(ok)}  failed: {len(results) - len(ok)}",
        f"mean citation_pass_rate: {_mean([r.pass_rate or 0.0 for r in ok]):.3f}",
        f"insufficient_data matched the eval file: {matched}/{len(ok)}",
        f"latest-year check passed: {_latest_cell(ok)}",
        f"tokens: {sum(r.input_tokens for r in ok)} in, {sum(r.output_tokens for r in ok)} out",
        f"prompt versions seen: {', '.join(versions) or 'none'}",
    ]
    return "\n".join(lines)


# ---- comparing two saved runs ------------------------------------------------------------


def _stats(rows: list[dict[str, Any]]) -> dict[str, float]:
    ok = [r for r in rows if r["ok"]]
    checked = [r for r in ok if r["latest_ok"] is not None]
    return {
        "ok": len(ok),
        "n": len(rows),
        "flag": sum(r["as_expected"] is True for r in ok),
        "latest": sum(r["latest_ok"] is True for r in checked),
        "latest_n": len(checked),
        "pass_rate": _mean([r["pass_rate"] or 0.0 for r in ok]),
        "tokens_in": _mean([float(r["input_tokens"]) for r in ok]),
        "seconds": _mean([r["seconds"] for r in ok]),
    }


def _pair(before: float, after: float, fmt: str) -> str:
    return f"{before:{fmt}} -> {after:{fmt}}"


def _compare_row(label: str, b: dict[str, float], a: dict[str, float]) -> str:
    latest = (
        "-"
        if not (b["latest_n"] or a["latest_n"])
        else (f"{int(b['latest'])}/{int(b['latest_n'])} -> {int(a['latest'])}/{int(a['latest_n'])}")
    )
    return (
        f"{label:<12} {int(b['flag'])}/{int(b['ok'])} -> {int(a['flag'])}/{int(a['ok'])}".ljust(34)
        + f"{latest:<16}"
        + f"{_pair(b['pass_rate'], a['pass_rate'], '.2f'):<14}"
        + f"{_pair(b['tokens_in'], a['tokens_in'], '.0f'):<17}"
        + _pair(b["seconds"], a["seconds"], ".1f")
    )


def _rows_for(rows: list[dict[str, Any]], question_id: str) -> list[dict[str, Any]]:
    return [r for r in rows if r["id"] == question_id]


def _shared_ids(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> list[str]:
    """Question ids present in both runs, in the order of the first run."""
    in_after = {r["id"] for r in after}
    return [i for i in dict.fromkeys(r["id"] for r in before) if i in in_after]


def _describe(run: dict[str, Any], label: str) -> str:
    meta = run["meta"]
    return f"{label} commit {meta.get('commit')}, prompts {meta.get('prompt_versions')}"


def compare_runs(before: dict[str, Any], after: dict[str, Any]) -> str:
    """Side-by-side figures for two saved runs, per question and overall."""
    ids = _shared_ids(before["results"], after["results"])
    header = (
        f"{'question':<12} {'flag ok':<22}{'latest year':<16}"
        f"{'pass_rate':<14}{'tokens in':<17}seconds"
    )
    lines = [_describe(before, "before:"), _describe(after, "after: "), "", header]
    for question_id in ids:
        b = _stats(_rows_for(before["results"], question_id))
        a = _stats(_rows_for(after["results"], question_id))
        lines.append(_compare_row(question_id, b, a))
    both = [r for r in before["results"] if r["id"] in ids]
    both_after = [r for r in after["results"] if r["id"] in ids]
    lines += ["", _compare_row("ALL", _stats(both), _stats(both_after))]
    return "\n".join(lines)


# ---- running ------------------------------------------------------------------------------


def check_server(client: httpx2.Client) -> None:
    """Fail fast with a clear message instead of 100 connection errors."""
    try:
        client.get("/health").raise_for_status()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Cannot reach the server at {client.base_url} ({type(exc).__name__}). "
            "Start it first: uv run --env-file .env uvicorn cited_answers.api:app"
        ) from exc


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://localhost:8000", help="base URL of the server")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--repeats", type=int, default=5, help="times each question is sent")
    parser.add_argument("--concurrency", type=int, default=2, help="requests in flight at once")
    parser.add_argument("--timeout", type=float, default=180.0, help="seconds per request")
    parser.add_argument("--only", nargs="+", metavar="ID", help="run just these question ids")
    parser.add_argument("--save", type=Path, help="write the results to this JSON file")
    parser.add_argument(
        "--compare", nargs=2, type=Path, metavar=("BEFORE", "AFTER"),
        help="print a comparison of two saved runs and exit; sends no requests",
    )  # fmt: skip
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.concurrency < 1:
        parser.error("--repeats and --concurrency must be at least 1")
    return args


def resolve_latest_years(questions: list[Question]) -> LatestYears:
    """Newest year per checked pair from the database, or nothing (with a warning)."""
    if not any(q.latest_checks for q in questions):
        return {}
    url = os.environ.get("DATABASE_URL")
    if not url:
        print(
            "WARNING: DATABASE_URL not set, so the latest-year check is skipped. "
            "Run with: uv run --env-file .env python scripts/load_test.py",
            file=sys.stderr,
        )
        return {}
    return fetch_latest_years(questions, url)


def run_all(
    args: argparse.Namespace,
    questions: list[Question],
    latest_years: LatestYears,
    transport: httpx2.BaseTransport | None,
) -> list[Result]:
    jobs = build_jobs(questions, args.repeats)
    print(f"Sending {len(jobs)} requests ({len(questions)} questions x {args.repeats}) "
          f"to {args.url}, {args.concurrency} at a time")  # fmt: skip
    results: list[Result] = []
    with httpx2.Client(base_url=args.url, timeout=args.timeout, transport=transport) as client:
        check_server(client)
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for result in pool.map(lambda job: run_job(client, job), jobs):
                result = with_latest_check(result, latest_years)
                results.append(result)
                print(progress_line(len(results), len(jobs), result), flush=True)
    return results


def main(
    argv: list[str] | None = None,
    transport: httpx2.BaseTransport | None = None,
    latest_years: LatestYears | None = None,
) -> int:
    """transport and latest_years let tests run the whole script without a server or database."""
    args = parse_args(argv)
    if args.compare:
        before, after = (json.loads(p.read_text()) for p in args.compare)
        print(compare_runs(before, after))
        return 0
    questions = load_questions(args.questions, args.only)
    if latest_years is None:
        latest_years = resolve_latest_years(questions)
    results = run_all(args, questions, latest_years, transport)
    print("\n" + summarize(questions, results))
    if args.save:
        meta = {
            "commit": git_commit(),
            "saved_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "url": args.url,
            "repeats": args.repeats,
            "prompt_versions": sorted({r.prompt_version for r in results if r.prompt_version}),
            "latest_years": {f"{c}:{i}": y for (c, i), y in sorted(latest_years.items())},
        }
        save_results(args.save, results, meta)
        print(f"\nSaved to {args.save}")
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
