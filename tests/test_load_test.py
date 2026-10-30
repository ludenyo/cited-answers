import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

import httpx2
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "load_test.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("load_test_script", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lt = _load_script()


def _body(insufficient: bool = False, pass_rate: float = 1.0) -> dict:
    return {
        "answer": "a",
        "insufficient_data": insufficient,
        "claims": [],
        "citation_pass_rate": pass_rate,
        "model": "m",
        "prompt_version": "abc12345",
        "latency_ms": 100,
        "input_tokens": 1000,
        "output_tokens": 100,
    }


def _client(handler) -> httpx2.Client:  # type: ignore[no-untyped-def]
    return httpx2.Client(base_url="http://test", transport=httpx2.MockTransport(handler))


def _question(qid: str = "q1", expect_insufficient: bool = False) -> object:
    return lt.Question(qid, "easy", f"question {qid}?", expect_insufficient)


# ---- questions and jobs ----------------------------------------------------------------


def test_loads_all_eval_questions_with_their_expectations() -> None:
    questions = lt.load_questions(lt.DEFAULT_QUESTIONS)
    assert len(questions) == 30
    by_id = {q.id: q for q in questions}
    assert by_id["easy-01"].expect_insufficient is False
    assert by_id["trap-01"].expect_insufficient is True
    assert by_id["trap-02"].expect_insufficient is False  # a false premise is answered, corrected
    assert by_id["easy-01"].text.startswith("What was Kenya's GDP growth")


def test_only_filters_and_unknown_ids_are_rejected() -> None:
    assert [q.id for q in lt.load_questions(lt.DEFAULT_QUESTIONS, ["trap-01", "easy-01"])] == [
        "easy-01",
        "trap-01",
    ]
    with pytest.raises(ValueError, match="nope"):
        lt.load_questions(lt.DEFAULT_QUESTIONS, ["easy-01", "nope"])


def test_default_run_is_150_requests_each_question_5_times_in_rounds() -> None:
    questions = lt.load_questions(lt.DEFAULT_QUESTIONS)
    jobs = lt.build_jobs(questions, 5)

    assert len(jobs) == 150
    assert all(sum(j.question.id == q.id for j in jobs) == 5 for q in questions)
    first_round = jobs[:30]
    assert [j.run for j in first_round] == [1] * 30  # every question runs once first
    assert [j.run for j in jobs[30:60]] == [2] * 30


# ---- one request -------------------------------------------------------------------------


def test_a_good_response_becomes_a_result() -> None:
    client = _client(lambda request: httpx2.Response(200, content=json.dumps(_body())))
    result = lt.run_job(client, lt.Job(_question(), 1))
    assert result.ok and result.pass_rate == 1.0 and result.as_expected is True
    assert (result.input_tokens, result.output_tokens) == (1000, 100)
    assert result.prompt_version == "abc12345"


def test_the_request_body_is_the_question_text() -> None:
    seen: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return httpx2.Response(200, content=json.dumps(_body()))

    lt.run_job(_client(handler), lt.Job(_question("q7"), 1))
    assert seen == [{"question": "question q7?"}]


def test_flag_disagreeing_with_the_eval_file_is_noticed() -> None:
    client = _client(lambda r: httpx2.Response(200, content=json.dumps(_body(insufficient=True))))
    result = lt.run_job(client, lt.Job(_question(expect_insufficient=False), 1))
    assert result.ok and result.as_expected is False
    assert "DIFFERS" in lt.progress_line(1, 1, result)


def test_http_errors_and_exceptions_become_failed_results_not_crashes() -> None:
    server_error = _client(lambda r: httpx2.Response(502, content="bad gateway"))
    failed = lt.run_job(server_error, lt.Job(_question(), 1))
    assert not failed.ok and failed.error == "HTTP 502" and failed.as_expected is None

    def boom(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    crashed = lt.run_job(_client(boom), lt.Job(_question(), 1))
    assert not crashed.ok and crashed.error == "ConnectError"

    garbage = _client(lambda r: httpx2.Response(200, content="not json"))
    assert not lt.run_job(garbage, lt.Job(_question(), 1)).ok


# ---- summary ---------------------------------------------------------------------------------


def _result(
    qid: str, run: int, ok: bool = True, pass_rate: float = 1.0, insufficient: bool = False
):  # type: ignore[no-untyped-def]
    job = lt.Job(_question(qid), run)
    if not ok:
        return lt.Result(job, False, 1.0, error="HTTP 500")
    return lt.Result(job, True, 2.0, None, insufficient, pass_rate, 1000, 100, "abc12345")


def test_summary_has_a_row_per_question_and_correct_totals() -> None:
    questions = [_question("q1"), _question("q2")]
    results = [
        _result("q1", 1),
        _result("q1", 2, pass_rate=0.5),
        _result("q2", 1, insufficient=True),  # expected False, so it differs
        _result("q2", 2, ok=False),
    ]

    text = lt.summarize(questions, results)

    assert "q1" in text and "q2" in text
    q1_row = next(line for line in text.splitlines() if line.startswith("q1"))
    assert "2/2" in q1_row and "0.75" in q1_row  # both ok, mean pass rate (1.0 + 0.5) / 2
    q2_row = next(line for line in text.splitlines() if line.startswith("q2"))
    assert "1/2" in q2_row  # one of two requests succeeded
    assert "requests: 4  ok: 3  failed: 1" in text
    assert "insufficient_data matched the eval file: 2/3" in text
    assert "tokens: 3000 in, 300 out" in text
    assert "prompt versions seen: abc12345" in text


def test_summary_survives_a_run_where_everything_failed() -> None:
    text = lt.summarize([_question("q1")], [_result("q1", 1, ok=False)])
    assert "ok: 0  failed: 1" in text and "prompt versions seen: none" in text


# ---- the whole script ----------------------------------------------------------------


def _fake_server(fail_ask: bool = False) -> httpx2.MockTransport:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/health":
            return httpx2.Response(200, json={"status": "ok"})
        if fail_ask:
            return httpx2.Response(500)
        return httpx2.Response(200, content=json.dumps(_body()))

    return httpx2.MockTransport(handler)


def test_main_runs_everything_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    code = lt.main(["--repeats", "2", "--only", "easy-01", "trap-01"], transport=_fake_server())
    out = capsys.readouterr().out
    assert code == 0
    assert "Sending 4 requests (2 questions x 2)" in out
    assert out.count("run 1") == 2 and out.count("run 2") == 2
    assert "requests: 4  ok: 4  failed: 0" in out


def test_main_exits_nonzero_when_any_request_failed() -> None:
    assert (
        lt.main(["--repeats", "1", "--only", "easy-01"], transport=_fake_server(fail_ask=True)) == 1
    )


def test_main_stops_early_with_a_clear_message_when_the_server_is_down() -> None:
    def down(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    with pytest.raises(SystemExit) as exit_info:
        lt.main(["--repeats", "1", "--only", "easy-01"], transport=httpx2.MockTransport(down))
    assert "Cannot reach the server" in str(exit_info.value)
    assert "uvicorn" in str(exit_info.value)


def test_bad_arguments_are_rejected() -> None:
    with pytest.raises(SystemExit):
        lt.parse_args(["--repeats", "0"])
    with pytest.raises(SystemExit):
        lt.parse_args(["--concurrency", "0"])


# ---- the latest-year check --------------------------------------------------------------------

POP = ("KEN", "SP.POP.TOTL")
LATEST = {POP: 2025}


def _checked(qid: str = "q1", pairs: tuple = (POP,)) -> object:
    return lt.Question(qid, "latest", "latest?", False, pairs)


def test_questions_with_a_check_load_their_pairs() -> None:
    by_id = {q.id: q for q in lt.load_questions(lt.DEFAULT_QUESTIONS)}
    assert by_id["latest-01"].latest_checks == (("KEN", "SP.POP.TOTL"),)
    assert by_id["latest-02"].latest_checks == (("KEN", "SE.PRM.ENRR"),)
    assert by_id["easy-01"].latest_checks == ()
    assert by_id["vague-03"].latest_checks == ()  # its note states the rule; it has no check


def test_cited_years_reads_only_the_matching_country_and_indicator() -> None:
    ids = (
        "KEN:SP.POP.TOTL:2025",
        "KEN:SP.POP.TOTL:2023",
        "UGA:SP.POP.TOTL:2024",
        "KEN:SE.PRM.ENRR:2022",
    )
    assert sorted(lt.cited_years(ids, POP)) == [2023, 2025]
    assert lt.cited_years((), POP) == []
    assert lt.cited_years(("KEN:SP.POP.TOTL:20250",), POP) == []  # not a four-digit year


def test_citing_the_newest_year_passes_even_alongside_older_ones() -> None:
    q = _checked()
    assert lt.latest_year_ok(q, ("KEN:SP.POP.TOTL:2025",), LATEST) is True
    assert lt.latest_year_ok(q, ("KEN:SP.POP.TOTL:2024", "KEN:SP.POP.TOTL:2025"), LATEST) is True


def test_citing_only_older_years_fails() -> None:
    q = _checked()
    assert lt.latest_year_ok(q, ("KEN:SP.POP.TOTL:2024",), LATEST) is False
    assert lt.latest_year_ok(q, ("KEN:SP.POP.TOTL:2023", "KEN:SP.POP.TOTL:2024"), LATEST) is False


def test_citing_nothing_for_the_pair_fails() -> None:
    assert lt.latest_year_ok(_checked(), (), LATEST) is False
    assert (
        lt.latest_year_ok(_checked(), ("UGA:SP.POP.TOTL:2025",), LATEST) is False
    )  # wrong country


def test_every_checked_pair_must_pass() -> None:
    enrol = ("KEN", "SE.PRM.ENRR")
    q = _checked(pairs=(POP, enrol))
    both = ("KEN:SP.POP.TOTL:2025", "KEN:SE.PRM.ENRR:2023")
    latest = {POP: 2025, enrol: 2023}
    assert lt.latest_year_ok(q, both, latest) is True
    assert lt.latest_year_ok(q, ("KEN:SP.POP.TOTL:2025", "KEN:SE.PRM.ENRR:2022"), latest) is False


def test_no_check_or_unknown_latest_year_means_not_checked() -> None:
    assert lt.latest_year_ok(_question(), ("KEN:SP.POP.TOTL:2025",), LATEST) is None
    assert (
        lt.latest_year_ok(_checked(), ("KEN:SP.POP.TOTL:2025",), {}) is None
    )  # database unavailable


def test_run_job_collects_every_cited_figure_including_derived_sources() -> None:
    body = _body()
    body["claims"] = [
        {"figure_id": "KEN:SP.POP.TOTL:2013", "source_figure_ids": []},
        {"figure_id": None, "source_figure_ids": ["KEN:SP.POP.TOTL:2013", "KEN:SP.POP.TOTL:2023"]},
    ]
    client = _client(lambda r: httpx2.Response(200, content=json.dumps(body)))
    result = lt.run_job(client, lt.Job(_question(), 1))
    assert result.figure_ids == (
        "KEN:SP.POP.TOTL:2013",
        "KEN:SP.POP.TOTL:2013",
        "KEN:SP.POP.TOTL:2023",
    )


def test_with_latest_check_marks_results_and_leaves_failures_alone() -> None:
    q = _checked()
    good = lt.Result(lt.Job(q, 1), True, 1.0, figure_ids=("KEN:SP.POP.TOTL:2025",))
    stale = lt.Result(lt.Job(q, 2), True, 1.0, figure_ids=("KEN:SP.POP.TOTL:2024",))
    failed = lt.Result(lt.Job(q, 3), False, 1.0, error="HTTP 500")
    assert lt.with_latest_check(good, LATEST).latest_ok is True
    assert lt.with_latest_check(stale, LATEST).latest_ok is False
    assert lt.with_latest_check(failed, LATEST).latest_ok is None


def test_progress_line_and_summary_report_the_check() -> None:
    q = _checked()
    stale = lt.Result(
        lt.Job(q, 1), True, 1.0, None, False, 1.0, 10, 5, "p", ("KEN:SP.POP.TOTL:2024",), False
    )
    fresh = lt.Result(
        lt.Job(q, 2), True, 1.0, None, False, 1.0, 10, 5, "p", ("KEN:SP.POP.TOTL:2025",), True
    )
    assert "latest_year=STALE" in lt.progress_line(1, 2, stale)
    assert "latest_year=ok" in lt.progress_line(2, 2, fresh)
    assert "latest_year" not in lt.progress_line(1, 1, _result("q1", 1))  # no check, nothing shown
    text = lt.summarize([q], [stale, fresh])
    assert "latest-year check passed: 1/2" in text
    row = next(line for line in text.splitlines() if line.startswith("q1"))
    assert "1/2" in row


def test_fetch_latest_years_asks_the_database_for_each_checked_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[tuple] = []

    class FakeConn:
        def __enter__(self) -> "FakeConn":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def execute(self, sql: str, params: tuple) -> object:
            asked.append(params)
            year = {"SP.POP.TOTL": 2025, "NO.DATA": None}[params[1]]
            return type("Cursor", (), {"fetchone": lambda self: (year,)})()

    import psycopg

    monkeypatch.setattr(psycopg, "connect", lambda url: FakeConn())
    questions = [_checked("a", (POP,)), _checked("b", (("KEN", "NO.DATA"),)), _checked("c", (POP,))]

    found = lt.fetch_latest_years(questions, "postgresql://unused")

    assert found == {POP: 2025}  # a pair with no rows is left out, not guessed
    assert sorted(asked) == [("KEN", "NO.DATA"), ("KEN", "SP.POP.TOTL")]  # each pair asked once


def test_without_database_url_the_check_is_skipped_with_a_warning(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert lt.resolve_latest_years([_checked()]) == {}
    assert "latest-year check is skipped" in capsys.readouterr().err
    assert lt.resolve_latest_years([_question()]) == {}  # nothing to check: no warning needed
    assert capsys.readouterr().err == ""


# ---- saving and comparing runs -----------------------------------------------------------------


def _checked_server(figure_id: str) -> httpx2.MockTransport:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/health":
            return httpx2.Response(200, json={"status": "ok"})
        body = _body()
        body["claims"] = [{"figure_id": figure_id, "source_figure_ids": []}]
        return httpx2.Response(200, content=json.dumps(body))

    return httpx2.MockTransport(handler)


def test_main_applies_the_check_and_saves_a_results_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "run.json"
    years = {("KEN", "SP.POP.TOTL"): 2025}
    args = ["--repeats", "2", "--only", "latest-01", "--save", str(out)]

    code = lt.main(args, transport=_checked_server("KEN:SP.POP.TOTL:2024"), latest_years=years)

    printed = capsys.readouterr().out
    assert code == 0
    assert printed.count("latest_year=STALE") == 2
    assert "latest-year check passed: 0/2" in printed
    saved = json.loads(out.read_text())
    assert len(saved["results"]) == 2
    first = saved["results"][0]
    assert first["id"] == "latest-01" and first["latest_ok"] is False
    assert first["figure_ids"] == ["KEN:SP.POP.TOTL:2024"]
    assert saved["meta"]["repeats"] == 2
    assert saved["meta"]["prompt_versions"] == ["abc12345"]
    assert saved["meta"]["latest_years"] == {"KEN:SP.POP.TOTL": 2025}
    assert "commit" in saved["meta"] and "saved_at" in saved["meta"]


def _row(
    qid: str, ok: bool = True, matched: bool = True, pass_rate: float = 1.0, latest=None
) -> dict:  # type: ignore[no-untyped-def]
    return {"id": qid, "run": 1, "ok": ok, "as_expected": matched if ok else None,
            "pass_rate": pass_rate if ok else None, "latest_ok": latest}  # fmt: skip


def _run(
    rows: list[dict], prompts: list[str], commit: str | None = None, dirty: bool = False
) -> dict:
    return {"meta": {"commit": commit, "dirty": dirty, "prompt_versions": prompts}, "results": rows}


def test_compare_lines_up_flag_agreement_pass_rate_and_latest_year_for_each_run() -> None:
    a = _run(
        [
            _row("q1", matched=False),
            _row("q1", matched=False),
            _row("q1"),
            _row("q1", pass_rate=0.5),
        ],
        ["aaaa1111"],
    )
    b = _run([_row("q1"), _row("q1"), _row("q1"), _row("q1")], ["bbbb2222"], "abc1234", dirty=True)

    text = lt.compare_runs([("before", a), ("after", b)])

    assert "A  before" in text and "B  after" in text
    assert "prompt aaaa1111" in text and "prompt bbbb2222" in text  # each run's prompt hash
    assert "abc1234 + uncommitted changes" in text
    row = next(line for line in text.splitlines() if line.startswith("q1"))
    assert "2/4" in row and "4/4" in row  # flag agreement went from 2 of 4 to 4 of 4
    assert "0.88" in row and "1.00" in row  # mean pass rate (1+1+1+0.5)/4 = 0.875 -> 0.88


def test_compare_handles_three_runs_and_the_latest_year_column() -> None:
    stale = [_row("latest-01", latest=False)] * 2
    fixed = [_row("latest-01", latest=True)] * 2
    runs = [("a", _run(stale, ["p1"])), ("b", _run(fixed, ["p2"])), ("c", _run(fixed, ["p3"]))]

    text = lt.compare_runs(runs)

    row = next(line for line in text.splitlines() if line.startswith("latest-01"))
    assert "0/2" in row and row.count("2/2") >= 4  # latest-year 0/2, 2/2, 2/2 plus flag columns
    assert "C  c" in text


def test_a_question_missing_from_one_run_shows_dashes_and_is_left_out_of_the_totals() -> None:
    a = _run([_row("q1"), _row("q2", matched=False)], ["p1"])
    b = _run([_row("q1")], ["p2"])

    text = lt.compare_runs([("a", a), ("b", b)])

    q2 = next(line for line in text.splitlines() if line.startswith("q2"))
    assert "0/1" in q2 and " -" in q2
    total = next(line for line in text.splitlines() if line.startswith("ALL"))
    assert "ALL (1)" in total and "1/1" in total  # only q1 is in both runs


def test_failed_requests_do_not_count_as_matches() -> None:
    a = _run([_row("q1"), _row("q1", ok=False)], ["p1"])
    b = _run([_row("q1")], ["p2"])
    row = next(
        line for line in lt.compare_runs([("a", a), ("b", b)]).splitlines() if line.startswith("q1")
    )
    assert "1/1" in row


TEXT_RUN = "\n".join(
    [
        "uv run python scripts/load_test.py",
        "Sending 4 requests (2 questions x 2) to http://localhost:8000, 2 at a time",
        "[1/4] easy-01      run 1    5.0s  pass_rate=1.00  insufficient_data=False (as expected)",
        "[2/4] vague-03     run 1    6.0s  pass_rate=1.00  insufficient_data=True "
        "(DIFFERS from eval file)",
        "[3/4] easy-01      run 2    4.0s  pass_rate=0.50  insufficient_data=False (as expected)"
        "  latest_year=STALE",
        "[4/4] vague-03     run 2  FAILED (HTTP 502) after 9.1s",
        "",
        "requests: 4  ok: 3  failed: 1",
        "prompt versions seen: 61ba7831",
        "",
    ]
)


def test_a_run_saved_only_as_the_scripts_printed_text_can_be_read_back() -> None:
    run = lt.parse_text_run(TEXT_RUN)

    assert run["meta"]["prompt_versions"] == ["61ba7831"]
    assert [r["id"] for r in run["results"]] == ["easy-01", "vague-03", "easy-01", "vague-03"]
    vague = [r for r in run["results"] if r["id"] == "vague-03"]
    assert vague[0]["as_expected"] is False and vague[0]["insufficient_data"] is True
    assert vague[1]["ok"] is False  # the failed request is kept as a failure
    easy = [r for r in run["results"] if r["id"] == "easy-01"]
    assert [r["pass_rate"] for r in easy] == [1.0, 0.5]
    assert easy[1]["latest_ok"] is False and easy[0]["latest_ok"] is None


def test_the_real_baseline_file_parses_to_ninety_answers_and_its_summary(tmp_path: Path) -> None:
    baseline = Path(__file__).parents[1] / "evals" / "results" / "baseline-load-test.txt"
    run = lt.load_run(baseline)
    assert len(run["results"]) == 90 and all(r["ok"] for r in run["results"])
    assert run["meta"]["prompt_versions"] == ["61ba7831"]
    vague03 = [r for r in run["results"] if r["id"] == "vague-03"]
    assert [r["as_expected"] for r in vague03] == [False] * 5  # the 0/5 from the summary table
    assert sum(r["as_expected"] is True for r in run["results"]) == 83  # the 83/90 line


def test_load_run_reads_json_and_text_by_extension(tmp_path: Path) -> None:
    saved = _run([_row("q1")], ["p1"], "abc")
    (tmp_path / "r.json").write_text(json.dumps(saved))
    (tmp_path / "r.txt").write_text(TEXT_RUN)
    assert lt.load_run(tmp_path / "r.json") == saved
    assert lt.load_run(tmp_path / "r.txt")["meta"]["source"] == "text"


def test_main_compares_saved_runs_without_a_server(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = tmp_path / "a.txt", tmp_path / "b.json"
    a.write_text(TEXT_RUN)
    b.write_text(json.dumps(_run([_row("easy-01"), _row("vague-03")], ["new11111"], "abc1234")))

    assert lt.main(["--compare", str(a), str(b)]) == 0  # no transport, no server

    text = capsys.readouterr().out
    assert "prompt 61ba7831" in text and "prompt new11111" in text
    assert any(line.startswith("easy-01") for line in text.splitlines())


def test_compare_needs_two_runs(tmp_path: Path) -> None:
    only = tmp_path / "a.txt"
    only.write_text(TEXT_RUN)
    with pytest.raises(SystemExit, match="at least two"):
        lt.main(["--compare", str(only)])


def test_saved_runs_say_whether_the_working_tree_was_dirty(tmp_path: Path) -> None:
    out = tmp_path / "run.json"
    transport = _checked_server("KEN:SP.POP.TOTL:2025")
    years = {("KEN", "SP.POP.TOTL"): 2025}
    lt.main(
        ["--repeats", "1", "--only", "latest-01", "--save", str(out)],
        transport=transport,
        latest_years=years,
    )
    meta = json.loads(out.read_text())["meta"]
    assert meta["dirty"] in (True, False, None)
    assert "commit" in meta


def test_cells_stay_apart_even_for_three_digit_counts() -> None:
    big = [_row("q1")] * 100  # 100/100 is the widest cell the table has to hold
    text = lt.compare_runs([("a", _run(big, ["p1"])), ("b", _run(big, ["p2"]))])
    total = next(line for line in text.splitlines() if line.startswith("ALL"))
    assert re.search(r"100/100\s+100/100", total)  # separated, never run together


# ---- guardrail expectations ------------------------------------------------------------


def test_guardrail_questions_load_their_refusal_and_guardrail_expectations() -> None:
    by_id = {q.id: q for q in lt.load_questions(lt.DEFAULT_QUESTIONS)}
    assert by_id["guardrail-scope-01"].expect_refused is True
    assert by_id["guardrail-injection-01"].guardrails_include == ("input_suspicious",)
    assert by_id["guardrail-output-01"].expect_refused is False
    assert by_id["easy-01"].expect_refused is False and by_id["easy-01"].guardrails_include == ()


def _guarded(refused: bool, guardrails: list[str], expect_refused: bool) -> lt.Result:
    question = lt.Question("g1", "guardrail", "q?", True, (), expect_refused, ("input_suspicious",))
    body = {**_body(insufficient=True), "refused": refused, "guardrails": guardrails}
    client = _client(lambda request: httpx2.Response(200, content=json.dumps(body)))
    return lt.run_job(client, lt.Job(question, 1))


def test_a_refusal_with_the_expected_guardrail_matches() -> None:
    assert _guarded(True, ["input_suspicious"], True).guardrails_ok is True


def test_a_missing_guardrail_or_an_unexpected_refusal_is_flagged() -> None:
    assert _guarded(True, ["input_out_of_scope"], True).guardrails_ok is False
    assert _guarded(True, ["input_suspicious"], False).guardrails_ok is False
    assert _guarded(False, ["input_suspicious"], True).guardrails_ok is False


def test_an_ordinary_question_that_is_refused_is_flagged() -> None:
    result = lt.run_job(
        _client(lambda r: httpx2.Response(200, content=json.dumps({**_body(), "refused": True}))),
        lt.Job(lt.Question("e1", "easy", "q?", False), 1),
    )
    assert result.guardrails_ok is False
