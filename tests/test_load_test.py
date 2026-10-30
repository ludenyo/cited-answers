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
    assert len(questions) == 18
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


def test_default_run_is_90_requests_each_question_5_times_in_rounds() -> None:
    questions = lt.load_questions(lt.DEFAULT_QUESTIONS)
    jobs = lt.build_jobs(questions, 5)

    assert len(jobs) == 90
    assert all(sum(j.question.id == q.id for j in jobs) == 5 for q in questions)
    first_round = jobs[:18]
    assert [j.run for j in first_round] == [1] * 18  # every question runs once first
    assert [j.run for j in jobs[18:36]] == [2] * 18


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
