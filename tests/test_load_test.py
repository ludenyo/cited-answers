import importlib.util
import json
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
    assert len(questions) == 20
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


def test_default_run_is_100_requests_each_question_5_times_in_rounds() -> None:
    questions = lt.load_questions(lt.DEFAULT_QUESTIONS)
    jobs = lt.build_jobs(questions, 5)

    assert len(jobs) == 100
    assert all(sum(j.question.id == q.id for j in jobs) == 5 for q in questions)
    first_round = jobs[:20]
    assert [j.run for j in first_round] == [1] * 20  # every question runs once first
    assert [j.run for j in jobs[20:40]] == [2] * 20


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


def test_compare_shows_before_and_after_side_by_side(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    years = {("KEN", "SP.POP.TOTL"): 2025}
    before, after = tmp_path / "before.json", tmp_path / "after.json"
    base = ["--repeats", "3", "--only", "latest-01"]
    lt.main(
        [*base, "--save", str(before)],
        transport=_checked_server("KEN:SP.POP.TOTL:2024"),
        latest_years=years,
    )
    lt.main(
        [*base, "--save", str(after)],
        transport=_checked_server("KEN:SP.POP.TOTL:2025"),
        latest_years=years,
    )
    capsys.readouterr()

    assert lt.main(["--compare", str(before), str(after)]) == 0

    text = capsys.readouterr().out
    row = next(line for line in text.splitlines() if line.startswith("latest-01"))
    assert "0/3 -> 3/3" in row  # the stale-year check went from 0 of 3 to 3 of 3
    assert "ALL" in text and "before: commit" in text and "after:  commit" in text


def test_compare_sends_no_requests_and_only_covers_shared_questions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = {
        "meta": {},
        "results": [
            {**lt.result_to_dict(_result("q1", 1)), "id": "q1"},
            {**lt.result_to_dict(_result("q2", 1)), "id": "q2"},
        ],
    }
    b = {"meta": {}, "results": [{**lt.result_to_dict(_result("q1", 1)), "id": "q1"}]}
    (tmp_path / "a.json").write_text(json.dumps(a))
    (tmp_path / "b.json").write_text(json.dumps(b))

    lt.main(["--compare", str(tmp_path / "a.json"), str(tmp_path / "b.json")])  # no transport given

    text = capsys.readouterr().out
    assert "q1" in text and "q2" not in text
