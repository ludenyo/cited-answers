"""Send every eval question to a running /ask endpoint several times.

Purpose: put enough traces into Langfuse to make the dashboard worth looking at, check
the answers against the eval file, and save results so two runs can be compared.

    uv run --env-file .env uvicorn cited_answers.api:app          # in one terminal
    uv run --env-file .env python scripts/load_test.py            # in another
    uv run --env-file .env python scripts/load_test.py --save evals/results/after.json
    uv run python scripts/load_test.py --compare evals/results/a.txt evals/results/b.json
    uv run python scripts/load_test.py --repeats 1 --only easy-01 trap-02

This makes real model calls. 30 questions x 5 repeats is 150 requests, roughly 5 to 15
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
DEFAULT_QUESTIONS = ROOT / "evals" / "questions.yaml"

Pair = tuple[str, str]  # (country_code, indicator_code)
LatestYears = dict[Pair, int]


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    text: str
    expect_insufficient: bool
    latest_checks: tuple[Pair, ...] = ()
    expect_refused: bool = False
    guardrails_include: tuple[str, ...] = ()


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
    refused: bool = False
    guardrails: tuple[str, ...] = ()

    @property
    def as_expected(self) -> bool | None:
        """Did the insufficient_data flag match the eval file? None when there was no answer."""
        if self.insufficient_data is None:
            return None
        return self.insufficient_data == self.job.question.expect_insufficient

    @property
    def guardrails_ok(self) -> bool | None:
        """Refused only when the eval file says so, and every expected guardrail fired.

        None when there was no answer.
        """
        question = self.job.question
        if not self.ok:
            return None
        return self.refused == question.expect_refused and set(question.guardrails_include) <= set(
            self.guardrails
        )


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
        bool(raw["expect"].get("refused", False)),
        tuple(raw["expect"].get("guardrails_include", [])),
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
            refused=bool(body.get("refused", False)),
            guardrails=tuple(body.get("guardrails", [])),
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
    if result.guardrails_ok is False:
        line += "  GUARDRAILS DIFFER"
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
        "refused": result.refused,
        "guardrails": list(result.guardrails),
        "guardrails_ok": result.guardrails_ok,
    }


def _git(*args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)  # noqa: S603, S607
    except OSError:
        return None


def git_commit() -> str | None:
    """Short hash of the checked-out commit, so a saved run says what code produced it."""
    out = _git("rev-parse", "--short", "HEAD")
    return out.stdout.strip() or None if out and out.returncode == 0 else None


def git_dirty() -> bool | None:
    """True when tracked files differ from HEAD, so the run used code not yet committed.

    Results saved while this is true belong to the commit that is about to be made.
    """
    out = _git("diff", "--quiet", "HEAD")
    return None if out is None or out.returncode not in (0, 1) else out.returncode == 1


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
        f"refusals and guardrails matched the eval file: "
        f"{sum(bool(r.guardrails_ok) for r in ok)}/{len(ok)}",
        f"tokens: {sum(r.input_tokens for r in ok)} in, {sum(r.output_tokens for r in ok)} out",
        f"prompt versions seen: {', '.join(versions) or 'none'}",
    ]
    return "\n".join(lines)


# ---- loading and comparing saved runs ---------------------------------------------------

_PROGRESS = re.compile(
    r"^\[\s*\d+/\d+\]\s+(?P<id>\S+)\s+run (?P<run>\d+)\s+(?P<seconds>[\d.]+)s\s+"
    r"pass_rate=(?P<pass_rate>[\d.]+)\s+insufficient_data=(?P<flag>True|False)\s+"
    r"\((?P<match>as expected|DIFFERS from eval file)\)(?:\s+latest_year=(?P<latest>ok|STALE))?"
)
_FAILED = re.compile(r"^\[\s*\d+/\d+\]\s+(?P<id>\S+)\s+run (?P<run>\d+)\s+FAILED")
_VERSIONS = re.compile(r"^prompt versions seen: (?P<versions>.+)$", re.MULTILINE)


def _row_from_progress(match: re.Match[str]) -> dict[str, Any]:
    return {
        "id": match["id"],
        "run": int(match["run"]),
        "ok": True,
        "seconds": float(match["seconds"]),
        "insufficient_data": match["flag"] == "True",
        "as_expected": match["match"] == "as expected",
        "pass_rate": float(match["pass_rate"]),
        "latest_ok": None if match["latest"] is None else match["latest"] == "ok",
    }


def parse_text_run(text: str) -> dict[str, Any]:
    """Rebuild a run from this script's own printed output (for runs saved only as text)."""
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if matched := _PROGRESS.match(line):
            rows.append(_row_from_progress(matched))
        elif failed := _FAILED.match(line):
            rows.append({"id": failed["id"], "run": int(failed["run"]), "ok": False,
                         "as_expected": None, "pass_rate": None, "latest_ok": None})  # fmt: skip
    versions = _VERSIONS.search(text)
    prompts = [v.strip() for v in versions["versions"].split(",")] if versions else []
    return {"meta": {"commit": None, "prompt_versions": prompts, "source": "text"}, "results": rows}


def load_run(path: Path) -> dict[str, Any]:
    """A saved run from a .json file (--save) or a .txt file (the script's printed output)."""
    if path.suffix == ".json":
        return json.loads(path.read_text())
    return parse_text_run(path.read_text())


def _stats(rows: list[dict[str, Any]]) -> dict[str, float]:
    ok = [r for r in rows if r["ok"]]
    checked = [r for r in ok if r.get("latest_ok") is not None]
    return {
        "ok": len(ok),
        "flag": sum(r["as_expected"] is True for r in ok),
        "latest": sum(r["latest_ok"] is True for r in checked),
        "latest_n": len(checked),
        "pass_rate": _mean([r["pass_rate"] or 0.0 for r in ok]),
    }


def _flag_cell(s: dict[str, float]) -> str:
    return f"{int(s['flag'])}/{int(s['ok'])}" if s["ok"] else "-"


def _rate_cell(s: dict[str, float]) -> str:
    return f"{s['pass_rate']:.2f}" if s["ok"] else "-"


def _latest_year_cell(s: dict[str, float]) -> str:
    return f"{int(s['latest'])}/{int(s['latest_n'])}" if s["latest_n"] else "-"


CELL = 8  # wide enough for "100/100"


def _group(stats: list[dict[str, float]], cell: Any, width: int = CELL) -> str:
    return "".join(f"{cell(s):<{width}}" for s in stats)


def _compare_line(label: str, stats: list[dict[str, float]]) -> str:
    return (
        f"{label:<13}{_group(stats, _flag_cell)}  {_group(stats, _rate_cell)}  "
        f"{_group(stats, _latest_year_cell)}"
    )


def _describe_run(letter: str, name: str, run: dict[str, Any]) -> str:
    meta = run["meta"]
    commit = meta.get("commit") or "-"
    if meta.get("dirty"):
        commit += " + uncommitted changes"
    prompts = ", ".join(meta.get("prompt_versions") or []) or "-"
    requests = len(run["results"])
    return f"  {letter}  {name:<30} prompt {prompts:<10} commit {commit:<34} {requests} requests"


def compare_runs(runs: list[tuple[str, dict[str, Any]]]) -> str:
    """Per-question flag agreement, pass rate and latest-year results for two or more runs."""
    letters = [chr(ord("A") + i) for i in range(len(runs))]
    ids = list(dict.fromkeys(r["id"] for _, run in runs for r in run["results"]))
    shared = [i for i in ids if all(any(r["id"] == i for r in run["results"]) for _, run in runs)]
    head = "".join(f"{letter:<{CELL}}" for letter in letters)
    width = CELL * len(runs) + 2
    lines = ["runs:"] + [
        _describe_run(L, n, run) for L, (n, run) in zip(letters, runs, strict=True)
    ]
    lines += [
        "",
        "flag agreement = answers whose insufficient_data matched the eval file (matched/answered)",
        "latest year = answers whose newest cited year was the newest in the database",
        "              (- when the question has no such check)",
        "",
        f"{'question':<13}{'flag agreement':<{width}}{'pass rate':<{width}}latest year",
        f"{'':<13}{head:<{width}}{head:<{width}}{head}",
    ]
    for question_id in ids:
        stats = [_stats([r for r in run["results"] if r["id"] == question_id]) for _, run in runs]
        lines.append(_compare_line(question_id, stats))
    totals = [_stats([r for r in run["results"] if r["id"] in shared]) for _, run in runs]
    lines += ["", _compare_line(f"ALL ({len(shared)})", totals)]
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
        "--compare", nargs="+", type=Path, metavar="RUN",
        help="compare two or more saved runs (.json or .txt), oldest first; sends no requests",
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
        if len(args.compare) < 2:
            raise SystemExit("--compare needs at least two saved runs")
        print(compare_runs([(p.stem, load_run(p)) for p in args.compare]))
        return 0
    questions = load_questions(args.questions, args.only)
    if latest_years is None:
        latest_years = resolve_latest_years(questions)
    results = run_all(args, questions, latest_years, transport)
    print("\n" + summarize(questions, results))
    if args.save:
        meta = {
            "commit": git_commit(),
            "dirty": git_dirty(),
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
