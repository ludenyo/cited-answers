"""Tests for the promptfoo eval suite: its checks, its case generator and its helper scripts."""

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import psycopg
import pytest
import yaml

from cited_answers.models import FigureResult
from cited_answers.settings import VerificationConfig

ROOT = Path(__file__).parents[1]
QUESTIONS = yaml.safe_load((ROOT / "evals" / "questions.yaml").read_text())["questions"]
FIXTURE = ROOT / "tests" / "fixtures" / "eval_figures.sql"


def _load(path: Path, name: str) -> ModuleType:
    sys.path.insert(0, str(path.parent))  # the promptfoo modules import each other by name
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checks = _load(ROOT / "evals" / "promptfoo" / "checks.py", "eval_checks")
generator = _load(ROOT / "evals" / "promptfoo" / "tests_from_yaml.py", "eval_generator")
gate = _load(ROOT / "scripts" / "eval_gate.py", "eval_gate")
export = _load(ROOT / "scripts" / "export_eval_fixture.py", "eval_export")
eval_db = _load(ROOT / "scripts" / "eval_db.py", "eval_db")

TOLERANCES = VerificationConfig(abs_tolerance=0.05, rel_tolerance=0.001)


def _figure(figure_id: str, value: float) -> FigureResult:
    country, indicator, year = figure_id.split(":")
    return FigureResult(
        figure_id=figure_id,
        country_code=country,
        country_name=country,
        indicator_code=indicator,
        indicator_name=indicator,
        year=int(year),
        value=value,
        unit=None,
        source_url=f"https://example.test/{figure_id}",
    )


FIGURES = {f.figure_id: f for f in (_figure("KEN:A.B:2022", 10.0), _figure("KEN:A.B:2023", 12.0))}


def _claim(**overrides: Any) -> dict[str, Any]:
    claim = {
        "statement": "x",
        "value": 12.0,
        "kind": "figure",
        "figure_id": "KEN:A.B:2023",
        "source_figure_ids": [],
        "operation": None,
        "passed": True,
        "actual_value": 12.0,
    }
    return claim | overrides


def _response(claims: list[dict[str, Any]], **overrides: Any) -> dict[str, Any]:
    base = {"answer": "text", "claims": claims, "insufficient_data": False, "refused": False}
    return base | overrides


# ---- assertions ------------------------------------------------------------------------


def test_parse_response_accepts_json_text_or_a_parsed_object() -> None:
    assert checks.parse_response('{"a": 1}') == {"a": 1}
    assert checks.parse_response({"a": 1}) == {"a": 1}
    with pytest.raises(ValueError):
        checks.parse_response("[1]")


def test_claims_pass_when_they_match_the_database() -> None:
    derived = _claim(
        kind="derived",
        figure_id=None,
        source_figure_ids=["KEN:A.B:2022", "KEN:A.B:2023"],
        operation="difference",
        value=2.0,
    )
    passed, _ = checks.check_claims(_response([_claim(), derived]), FIGURES.get, TOLERANCES)
    assert passed


def test_a_claim_is_rechecked_even_if_the_response_says_it_passed() -> None:
    wrong = _claim(value=99.0, passed=True)
    passed, reason = checks.check_claims(_response([wrong]), FIGURES.get, TOLERANCES)
    assert not passed and "x" in reason


def test_a_derived_claim_with_the_wrong_sign_fails() -> None:
    fall = _claim(
        kind="derived",
        figure_id=None,
        source_figure_ids=["KEN:A.B:2023", "KEN:A.B:2022"],
        operation="difference",
        value=2.0,  # to - from is -2.0
    )
    assert not checks.check_claims(_response([fall]), FIGURES.get, TOLERANCES)[0]


def test_no_claims_is_a_pass_for_the_claims_check() -> None:
    assert checks.check_claims(_response([]), FIGURES.get, TOLERANCES)[0]


def test_flag_and_refusal_checks() -> None:
    assert checks.check_flag(_response([], insufficient_data=True), True)[0]
    assert not checks.check_flag(_response([]), True)[0]
    refused = _response([], refused=True, guardrails=["input_suspicious"])
    assert checks.check_refusal(refused, True, ["input_suspicious"])[0]
    assert not checks.check_refusal(refused, True, ["input_out_of_scope"])[0]
    assert not checks.check_refusal(refused, False, [])[0]
    assert not checks.check_refusal(_response([]), True, [])[0]


def test_latest_year_needs_the_newest_cited_year_to_be_the_database_newest() -> None:
    pair = ("KEN", "A.B")
    assert checks.check_latest_year(_response([_claim()]), {pair: 2023})[0]
    older_too = _response([_claim(figure_id="KEN:A.B:2022"), _claim()])
    assert checks.check_latest_year(older_too, {pair: 2023})[0]
    only_old = _response([_claim(figure_id="KEN:A.B:2022")])
    assert not checks.check_latest_year(only_old, {pair: 2023})[0]
    assert not checks.check_latest_year(_response([]), {pair: 2023})[0]


def test_latest_year_counts_the_sources_of_a_derived_claim() -> None:
    derived = _claim(
        kind="derived",
        figure_id=None,
        source_figure_ids=["KEN:A.B:2022", "KEN:A.B:2023"],
        operation="difference",
    )
    assert checks.check_latest_year(_response([derived]), {("KEN", "A.B"): 2023})[0]


def test_the_rubric_view_gives_each_claim_one_explicit_citation() -> None:
    derived = _claim(
        kind="derived",
        figure_id=None,
        source_figure_ids=["KEN:A.B:2022", "KEN:A.B:2023"],
        operation="difference",
    )
    view = json.loads(checks.rubric_view(_response([_claim(), derived])))
    assert view["answer"] == "text"
    assert [c["cited"] for c in view["claims"]] == [
        "KEN:A.B:2023",
        ["KEN:A.B:2022", "KEN:A.B:2023"],
    ]
    assert all("source_figure_ids" not in c for c in view["claims"])


# ---- test case generator ---------------------------------------------------------------


def test_one_case_per_eval_question_with_the_expected_values_untouched() -> None:
    cases = generator.generate_tests()
    assert [c["metadata"]["id"] for c in cases] == [q["id"] for q in QUESTIONS]
    for case, q in zip(cases, QUESTIONS, strict=True):
        assert case["vars"]["question"] == q["question"]
        flag = next(a for a in case["assert"] if a["metric"] == "insufficient_data")
        assert flag["config"]["insufficient_data"] is q["expect"]["insufficient_data"]


def test_every_case_checks_claims_flag_guardrails_and_latency_under_20_seconds() -> None:
    for case in generator.generate_tests():
        by_metric = {a["metric"]: a for a in case["assert"]}
        assert {"claims", "insufficient_data", "guardrails", "latency"} <= set(by_metric)
        assert by_metric["latency"]["threshold"] == 20_000


def test_latest_year_and_rubric_apply_only_where_they_make_sense() -> None:
    for case, q in zip(generator.generate_tests(), QUESTIONS, strict=True):
        metrics = {a["metric"] for a in case["assert"]}
        assert ("latest_year" in metrics) == bool(q["expect"].get("latest_year_check"))
        assert ("rubric" in metrics) == (not q["expect"].get("refused", False))


def test_the_rubric_text_is_the_one_asked_for() -> None:
    assert generator.RUBRIC == "Answers the question directly; no number without a citation."


# ---- the gate ------------------------------------------------------------------------


def _run(name: str, **metrics: bool) -> dict[str, Any]:
    """One promptfoo result: metric name -> did that assertion pass."""
    components = [{"pass": ok, "assertion": {"metric": m}} for m, ok in metrics.items()]
    return {
        "success": all(metrics.values()),
        "testCase": {"description": name},
        "gradingResult": {"componentResults": components},
    }


ALL_PASS = dict(claims=True, insufficient_data=True, latest_year=True, guardrails=True)


def test_all_hard_gates_passing_passes_the_gate() -> None:
    lines, ok = gate.report([_run("a", **ALL_PASS, rubric=True, latency=True)])
    assert ok and lines[0].startswith("hard gates: 1/1")


@pytest.mark.parametrize("metric", gate.HARD_GATES)
def test_any_failed_hard_gate_fails_the_gate_and_is_named(metric: str) -> None:
    lines, ok = gate.report([_run("case-x", **{**ALL_PASS, metric: False})])
    assert not ok
    assert any("case-x" in line and metric in line for line in lines)


@pytest.mark.parametrize("metric", gate.REPORT_ONLY)
def test_a_failed_rubric_or_latency_is_reported_but_never_fails_the_gate(metric: str) -> None:
    lines, ok = gate.report([_run("a", **ALL_PASS, **{metric: False})])
    assert ok
    assert f"report only: {metric} 0/1 passed" in lines
    assert any(line.strip() == f"{metric}: a" for line in lines)


def test_with_repeats_one_failing_run_fails_and_the_count_is_shown() -> None:
    runs = [_run("flaky", **ALL_PASS), _run("flaky", **{**ALL_PASS, "guardrails": False})]
    lines, ok = gate.report(runs + [_run("flaky", **ALL_PASS)])
    assert not ok
    assert lines[0] == "hard gates: 2/3 runs passed all of them"
    assert "  FAILED 1 run(s): flaky  [guardrails]" in lines


def test_a_run_with_no_assertion_results_is_a_failure_not_a_pass() -> None:
    errored = {"success": False, "testCase": {"description": "boom"}, "gradingResult": None}
    assert not gate.report([errored])[1]


def test_an_empty_run_fails_the_gate() -> None:
    assert gate.report([]) == (["no cases ran"], False)


def test_main_exit_code_follows_the_hard_gates(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"results": {"results": [_run("a", **ALL_PASS, rubric=False)]}}))
    assert gate.main([str(path)]) == 0
    bad = _run("a", **{**ALL_PASS, "claims": False})
    path.write_text(json.dumps({"results": {"results": [bad]}}))
    assert gate.main([str(path)]) == 1


def test_the_metric_names_match_what_the_case_generator_emits() -> None:
    emitted = {a["metric"] for c in generator.generate_tests() for a in c["assert"]}
    assert emitted == set(gate.HARD_GATES) | set(gate.REPORT_ONLY)


# ---- the fixture and its tooling -------------------------------------------------------


def test_the_fixture_has_every_figure_the_eval_cases_expect() -> None:
    sql = FIXTURE.read_text()
    ids = {i for q in QUESTIONS for i in q["expect"].get("figure_ids", [])}
    assert ids
    for figure_id in ids:
        assert "INSERT INTO figures (id, " in sql and f"VALUES ('{figure_id}'," in sql, figure_id


def test_the_fixture_has_a_chunk_for_every_figure() -> None:
    sql = FIXTURE.read_text()
    figures = len(re.findall(r"^INSERT INTO figures ", sql, re.MULTILINE))
    chunks = len(re.findall(r"^INSERT INTO figure_chunks ", sql, re.MULTILINE))
    assert figures == chunks > 0


def test_expected_series_covers_figure_ids_and_latest_year_pairs() -> None:
    questions = [
        {"id": "a", "expect": {"figure_ids": ["KEN:A.B:2023"]}},
        {"id": "b", "expect": {"latest_year_check": [{"country": "UGA", "indicator": "X.Y"}]}},
        {"id": "c", "expect": {}},
    ]
    assert export.expected_series(questions) == [("KEN", "A.B"), ("UGA", "X.Y")]
    with pytest.raises(ValueError, match="bad figure id"):
        export.expected_series([{"id": "d", "expect": {"figure_ids": ["nope"]}}])


def test_sql_literals_escape_quotes_and_keep_numbers_bare() -> None:
    assert export.literal(None) == "NULL"
    assert export.literal(3.5) == "3.5"
    assert export.literal("it's") == "'it''s'"
    row = export.insert("t", "a, b", ("x", 1))
    assert row == "INSERT INTO t (a, b) VALUES ('x', 1) ON CONFLICT DO NOTHING;"


def test_eval_database_url_prefers_the_override_then_derives_a_separate_database() -> None:
    assert eval_db.eval_url({"EVAL_DATABASE_URL": "postgresql://u@h/ci"}) == "postgresql://u@h/ci"
    derived = eval_db.eval_url({"DATABASE_URL": "postgresql://u:p@localhost:5435/cited_answers"})
    assert derived == "postgresql://u:p@localhost:5435/cited_answers_eval"


def test_loading_the_fixture_replaces_whatever_figures_were_there(db: psycopg.Connection) -> None:
    db.execute(
        "INSERT INTO figures (id, country_code, country_name, indicator_code, indicator_name, "
        "year, value, unit, source_url) VALUES ('X:Y:2000','X','X','Y','Y',2000,1,NULL,'u')"
    )
    count = eval_db.replace_figures(db, FIXTURE)
    assert count == len(re.findall(r"^INSERT INTO figures ", FIXTURE.read_text(), re.MULTILINE))
    assert db.execute("SELECT count(*) FROM figures WHERE id = 'X:Y:2000'").fetchone() == (0,)
    chunks = db.execute("SELECT count(*) FROM figure_chunks").fetchone()
    assert chunks == (count,)
