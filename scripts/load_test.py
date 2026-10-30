"""Send every eval question to a running /ask endpoint several times.

Purpose: put enough traces into Langfuse to make the dashboard worth looking at, and
print a per-question summary so repeatability is visible straight away.

    uv run --env-file .env uvicorn cited_answers.api:app          # in one terminal
    uv run python scripts/load_test.py                            # in another
    uv run python scripts/load_test.py --repeats 1 --only easy-01 trap-02

This makes real model calls. The default 18 questions x 5 repeats is 90 requests,
roughly 10 to 15 minutes at the default concurrency and a few US cents on Haiku.
"""

import argparse
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx2
import yaml

DEFAULT_QUESTIONS = Path(__file__).resolve().parents[1] / "evals" / "questions_draft.yaml"


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    text: str
    expect_insufficient: bool


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

    @property
    def as_expected(self) -> bool | None:
        """Did the insufficient_data flag match the eval file? None when there was no answer."""
        if self.insufficient_data is None:
            return None
        return self.insufficient_data == self.job.question.expect_insufficient


def load_questions(path: Path, only: list[str] | None = None) -> list[Question]:
    raw = yaml.safe_load(path.read_text())["questions"]
    questions = [
        Question(q["id"], q["category"], q["question"], bool(q["expect"]["insufficient_data"]))
        for q in raw
    ]
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
        )
    except Exception as exc:  # noqa: BLE001 - a bad response must not stop the whole run
        return Result(job, False, time.perf_counter() - started, error=type(exc).__name__)


def progress_line(done: int, total: int, result: Result) -> str:
    label = f"[{done:>{len(str(total))}}/{total}] {result.job.question.id:<12} run {result.job.run}"
    if not result.ok:
        return f"{label}  FAILED ({result.error}) after {result.seconds:.1f}s"
    flag = "as expected" if result.as_expected else "DIFFERS from eval file"
    return (
        f"{label}  {result.seconds:5.1f}s  pass_rate={result.pass_rate:.2f}  "
        f"insufficient_data={result.insufficient_data} ({flag})"
    )


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def _row(question: Question, results: list[Result]) -> str:
    ok = [r for r in results if r.ok]
    matched = [r for r in ok if r.as_expected]
    return (
        f"{question.id:<12} {len(ok):>2}/{len(results):<2}  "
        f"{_mean([r.pass_rate or 0.0 for r in ok]):>9.2f}  "
        f"{len(matched):>2}/{len(ok):<2}  "
        f"{_mean([r.seconds for r in ok]):>8.1f}s  "
        f"{_mean([float(r.input_tokens) for r in ok]):>9.0f}  "
        f"{_mean([float(r.output_tokens) for r in ok]):>9.0f}"
    )


def summarize(questions: list[Question], results: list[Result]) -> str:
    """A per-question table, then totals."""
    lines = [
        f"{'question':<12} {'ok':<5}  {'pass_rate':>9}  {'flag ok':<5}  "
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
        f"tokens: {sum(r.input_tokens for r in ok)} in, {sum(r.output_tokens for r in ok)} out",
        f"prompt versions seen: {', '.join(versions) or 'none'}",
    ]
    return "\n".join(lines)


def check_server(client: httpx2.Client) -> None:
    """Fail fast with a clear message instead of 90 connection errors."""
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
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.concurrency < 1:
        parser.error("--repeats and --concurrency must be at least 1")
    return args


def main(argv: list[str] | None = None, transport: httpx2.BaseTransport | None = None) -> int:
    """transport lets tests run the whole script against a fake server."""
    args = parse_args(argv)
    questions = load_questions(args.questions, args.only)
    jobs = build_jobs(questions, args.repeats)
    print(
        f"Sending {len(jobs)} requests ({len(questions)} questions x {args.repeats}) "
        f"to {args.url}, {args.concurrency} at a time"
    )
    results: list[Result] = []
    with httpx2.Client(base_url=args.url, timeout=args.timeout, transport=transport) as client:
        check_server(client)
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for result in pool.map(lambda job: run_job(client, job), jobs):
                results.append(result)
                print(progress_line(len(results), len(jobs), result), flush=True)
    print("\n" + summarize(questions, results))
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
