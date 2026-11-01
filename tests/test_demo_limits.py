"""Demo protections: the daily cap, environment-configured limits, and the index page."""

import threading
from collections.abc import Callable
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from cited_answers import api as api_module
from cited_answers.api import DAILY_CAP_MESSAGE, create_app, get_ask_fn
from cited_answers.guardrails import DailyCap, RateLimiter
from cited_answers.models import AskResponse
from cited_answers.settings import (
    RateLimitConfig,
    SettingsError,
    daily_ask_cap_from_env,
    rate_limit_from_env,
    trusted_proxy_hops,
)
from tests.test_ask_endpoint import CANNED

DAY = 86_400
NOON_DAY_ONE = DAY + 12 * 3600  # an arbitrary UTC day, at 12:00


class FakeClock:
    def __init__(self, now: float = NOON_DAY_ONE) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _client(
    calls: list[str],
    *,
    rate_limiter: RateLimiter | None = None,
    daily_cap: DailyCap | None = None,
) -> TestClient:
    def ask_fn(question: str) -> AskResponse:
        calls.append(question)
        return CANNED

    app = create_app(rate_limiter=rate_limiter, daily_cap=daily_cap)
    app.dependency_overrides[get_ask_fn] = lambda: ask_fn
    return TestClient(app)


def _post(client: TestClient, question: str = "q", ip: str = "1.1.1.1") -> int:
    return client.post("/ask", json={"question": question}, headers={"x-ip": ip}).status_code


@pytest.fixture(autouse=True)
def _by_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tell clients apart by a test header, since every TestClient shares one address."""
    monkeypatch.setattr(api_module, "client_key", lambda request: request.headers["x-ip"])
    for name in ("RATE_LIMIT_REQUESTS", "RATE_LIMIT_WINDOW_SECONDS", "DAILY_ASK_CAP"):
        monkeypatch.delenv(name, raising=False)


# ---- the daily cap ------------------------------------------------------------------------


def test_daily_cap_allows_the_limit_then_blocks_until_midnight_utc() -> None:
    clock = FakeClock()
    cap = DailyCap(2, clock=clock)

    assert [cap.take(), cap.take()] == [None, None]
    wait = cap.take()

    assert wait == 12 * 3600  # noon to the next UTC midnight
    clock.now += 3600
    assert cap.take() == 11 * 3600  # still blocked, and the wait shrinks


def test_daily_cap_resets_on_the_next_utc_day() -> None:
    clock = FakeClock(2 * DAY - 1)  # one second before midnight
    cap = DailyCap(1, clock=clock)
    assert cap.take() is None
    assert cap.take() == 1

    clock.now += 1  # midnight

    assert cap.take() is None
    assert cap.take() == DAY


def test_daily_cap_is_exact_under_concurrent_calls() -> None:
    cap = DailyCap(100)
    allowed: list[bool] = []

    def worker() -> None:
        for _ in range(50):
            allowed.append(cap.take() is None)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    assert len(allowed) == 400 and sum(allowed) == 100


def test_ask_after_the_daily_cap_gets_a_friendly_429_and_never_reaches_the_pipeline() -> None:
    calls: list[str] = []
    client = _client(calls, daily_cap=DailyCap(2, clock=FakeClock()))

    codes = [_post(client, f"q{i}", ip=f"10.0.0.{i}") for i in range(4)]

    assert codes == [200, 200, 429, 429]  # different clients: this is the global cap, not per IP
    assert calls == ["q0", "q1"]
    blocked = client.post("/ask", json={"question": "again"}, headers={"x-ip": "9.9.9.9"})
    assert blocked.json()["detail"] == DAILY_CAP_MESSAGE
    assert "Demo limit reached" in blocked.json()["detail"]
    assert int(blocked.headers["Retry-After"]) == 12 * 3600


def test_the_cap_lifts_the_next_day() -> None:
    clock = FakeClock()
    client = _client([], daily_cap=DailyCap(1, clock=clock))
    assert [_post(client), _post(client, ip="2.2.2.2")] == [200, 429]

    clock.now += DAY

    assert _post(client, ip="3.3.3.3") == 200


def test_the_health_check_and_index_page_do_not_use_the_daily_cap() -> None:
    client = _client([], daily_cap=DailyCap(1, clock=FakeClock()))
    assert _post(client) == 200  # the cap is now used up

    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200


def test_a_client_blocked_by_its_own_rate_limit_does_not_use_the_daily_cap() -> None:
    calls: list[str] = []
    client = _client(
        calls,
        rate_limiter=RateLimiter(1, 3600),
        daily_cap=DailyCap(2, clock=FakeClock()),
    )

    spammer = [_post(client, f"s{i}", ip="6.6.6.6") for i in range(5)]
    other = _post(client, "other", ip="7.7.7.7")

    assert spammer == [200, 429, 429, 429, 429]
    assert other == 200  # one cap slot was left, so the spam did not drain it
    assert calls == ["s0", "other"]


def test_there_is_no_cap_unless_one_is_configured() -> None:
    client = _client([])
    assert client.app.state.daily_cap is None  # type: ignore[attr-defined]
    assert [_post(client, ip=f"8.8.8.{i}") for i in range(20)] == [200] * 20


# ---- limits from the environment -----------------------------------------------------------


def test_the_daily_cap_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAILY_ASK_CAP", "300")
    cap = create_app().state.daily_cap
    assert cap is not None and cap._limit == 300


@pytest.mark.parametrize("raw", ["", "0"])
def test_an_empty_or_zero_daily_cap_means_no_cap(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("DAILY_ASK_CAP", raw)
    assert daily_ask_cap_from_env() is None


def test_the_per_ip_limit_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "10")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_SECONDS", "3600")

    limiter = create_app().state.rate_limiter

    assert (limiter._max, limiter._window) == (10, 3600)


def test_each_per_ip_setting_falls_back_to_the_config_on_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = RateLimitConfig(requests=60, window_seconds=60)
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "5")
    assert rate_limit_from_env(base) == RateLimitConfig(requests=5, window_seconds=60)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("DAILY_ASK_CAP", "lots"),
        ("DAILY_ASK_CAP", "-1"),
        ("RATE_LIMIT_REQUESTS", "0"),
        ("RATE_LIMIT_WINDOW_SECONDS", "soon"),
        ("TRUST_PROXY_HOPS", "-2"),
    ],
)
def test_a_bad_limit_fails_loudly_instead_of_being_ignored(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(SettingsError, match=name):
        create_app()


# ---- who is asking behind a proxy ----------------------------------------------------------


def _request(forwarded: str | None, peer: str = "169.254.1.1") -> SimpleNamespace:
    headers = {"x-forwarded-for": forwarded} if forwarded is not None else {}
    return SimpleNamespace(client=SimpleNamespace(host=peer), headers=headers)


@pytest.fixture
def real_client_key(monkeypatch: pytest.MonkeyPatch) -> Callable[[SimpleNamespace], str]:
    monkeypatch.undo()  # drop the header-based stand-in from the autouse fixture
    return api_module.client_key  # type: ignore[return-value]


def test_without_trusted_proxies_the_forwarded_header_is_ignored(
    monkeypatch: pytest.MonkeyPatch, real_client_key: Callable[[SimpleNamespace], str]
) -> None:
    monkeypatch.delenv("TRUST_PROXY_HOPS", raising=False)
    assert trusted_proxy_hops() == 0
    assert real_client_key(_request("6.6.6.6")) == "169.254.1.1"  # a forged header changes nothing


def test_behind_one_proxy_the_address_the_proxy_appended_is_used(
    monkeypatch: pytest.MonkeyPatch, real_client_key: Callable[[SimpleNamespace], str]
) -> None:
    monkeypatch.setenv("TRUST_PROXY_HOPS", "1")
    # The client claimed to be 1.2.3.4; the proxy appended the address it really saw.
    assert real_client_key(_request("1.2.3.4, 203.0.113.9")) == "203.0.113.9"


def test_a_missing_or_short_forwarded_header_falls_back_to_the_peer(
    monkeypatch: pytest.MonkeyPatch, real_client_key: Callable[[SimpleNamespace], str]
) -> None:
    monkeypatch.setenv("TRUST_PROXY_HOPS", "2")
    assert real_client_key(_request(None)) == "169.254.1.1"
    assert real_client_key(_request("203.0.113.9")) == "169.254.1.1"


# ---- the index page -------------------------------------------------------------------------


def test_the_index_page_has_the_question_box_examples_and_the_length_limit() -> None:
    page = _client([]).get("/")

    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert 'maxlength="500"' in page.text
    assert "EXAMPLES" in page.text and 'fetch("/ask"' in page.text


def test_questions_over_500_characters_are_still_rejected() -> None:
    client = _client([])
    assert _post(client, "x" * 501) == 422
    assert _post(client, "x" * 500) == 200
