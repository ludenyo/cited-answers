"""FastAPI application and HTTP routes."""

import logging
import math
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Annotated

import anthropic
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

from cited_answers import embeddings, guardrails, retrieval, tracing
from cited_answers.answer import ask, load_system_prompt, run_tool
from cited_answers.db import connect
from cited_answers.guardrails import DailyCap, RateLimiter
from cited_answers.models import AskRequest, AskResponse
from cited_answers.settings import (
    Settings,
    daily_ask_cap_from_env,
    get_app_config,
    rate_limit_from_env,
    trusted_proxy_hops,
)

logger = logging.getLogger(__name__)

AskFn = Callable[[str], AskResponse]

INDEX_PAGE = Path(__file__).parent / "static" / "index.html"
DAILY_CAP_MESSAGE = (
    "Demo limit reached: this demo answers a limited number of questions per day to keep "
    "its costs down. Please come back tomorrow."
)


def get_ask_fn() -> AskFn:
    """Production wiring: one DB connection and one model client per request.

    Kept as a dependency so tests can replace the whole pipeline.
    """

    def _ask(question: str) -> AskResponse:
        with connect() as conn:
            return ask(
                question,
                # Reads ANTHROPIC_API_KEY from the environment. No automatic retries: the SDK would
                # re-send a timed-out call with a fresh timeout, which breaks the request deadline.
                client=anthropic.Anthropic(max_retries=0),
                run=partial(run_tool, conn),
                lookup=partial(retrieval.get_figure_by_id, conn),
                settings=Settings.from_env(),
                config=get_app_config(),
            )

    return _ask


def configure_logging() -> None:
    """Make this package's INFO logs visible under uvicorn.

    Uvicorn configures only its own loggers, so without this the per-request
    latency and token line would be dropped. Only our package logger is touched,
    so third-party libraries keep their default (quiet) levels.
    """
    package_logger = logging.getLogger("cited_answers")
    package_logger.setLevel(logging.INFO)
    if not package_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        package_logger.addHandler(handler)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Fail at startup, with a clear message, if configuration is missing or broken."""
    configure_logging()
    Settings.from_env()  # raises SettingsError when ANTHROPIC_MODEL is unset
    config = get_app_config()
    load_system_prompt(config.answer.prompt_path)  # raises if the file is gone
    guardrails.load_classifier_prompt(config.guardrails.classifier.prompt_path)
    tracing.init_tracing()  # never raises: a tracing problem must not stop the app
    if os.environ.get("WARM_EMBEDDING_MODEL", "").strip().lower() == "true":
        embeddings.warm_up()  # load the model now so the first question is not slow
    yield
    tracing.shutdown_tracing()  # send queued traces before exiting


def client_key(request: Request) -> str:
    """Who is asking, for rate limiting: the address the connection came from.

    Behind a reverse proxy that is the proxy's address, so every user would share one
    limit. With TRUST_PROXY_HOPS=N the Nth entry from the right of X-Forwarded-For is
    used instead: each proxy appends the address it saw, so the entries on the left can
    be forged by the client and only the ones the trusted proxies added can be believed.
    """
    peer = request.client.host if request.client else "unknown"
    hops = trusted_proxy_hops()
    if hops == 0:
        return peer
    entries = [
        e.strip() for e in request.headers.get("x-forwarded-for", "").split(",") if e.strip()
    ]
    return entries[-hops] if len(entries) >= hops else peer


def enforce_rate_limit(request: Request) -> None:
    """Reject the request with 429 if this client has used up its allowance."""
    key = client_key(request)
    wait = request.app.state.rate_limiter.check(key)
    if wait is None:
        return
    retry_after = max(1, math.ceil(wait))
    # The address is hashed: events and logs need to tell clients apart, not to store IPs.
    guardrails.record_trigger(
        guardrails.RATE_LIMITED,
        client=guardrails.client_fingerprint(key),
        retry_after_seconds=retry_after,
    )
    raise HTTPException(
        status_code=429,
        detail=f"Too many requests. Please try again in {retry_after} seconds.",
        headers={"Retry-After": str(retry_after)},
    )


def enforce_daily_cap(request: Request) -> None:
    """Reject the request with 429 once the day's global allowance of /ask calls is used."""
    cap: DailyCap | None = request.app.state.daily_cap
    wait = cap.take() if cap else None
    if wait is None:
        return
    retry_after = max(1, math.ceil(wait))
    guardrails.record_trigger(guardrails.DAILY_CAP_REACHED, retry_after_seconds=retry_after)
    raise HTTPException(
        status_code=429,
        detail=DAILY_CAP_MESSAGE,
        headers={"Retry-After": str(retry_after)},
    )


def create_app(
    rate_limiter: RateLimiter | None = None, daily_cap: DailyCap | None = None
) -> FastAPI:
    """Build the FastAPI app.

    A factory (rather than a module-level app) keeps tests free to build a
    fresh instance with their own settings, such as a tighter rate limit. Without
    arguments the limits come from config/app.yaml and the RATE_LIMIT_*, DAILY_ASK_CAP
    environment variables.
    """
    app = FastAPI(title="cited-answers", lifespan=lifespan)
    limits = rate_limit_from_env(get_app_config().guardrails.rate_limit)
    app.state.rate_limiter = rate_limiter or RateLimiter(limits.requests, limits.window_seconds)
    trusted_proxy_hops()  # read per request, but a bad value should stop startup, not a request
    cap = daily_ask_cap_from_env()
    app.state.daily_cap = daily_cap or (DailyCap(cap) if cap else None)
    index_html = INDEX_PAGE.read_text(encoding="utf-8")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return index_html

    @app.get("/health")
    def health() -> dict[str, str]:
        # Liveness only for now; a database ping will be added once the DB layer exists.
        return {"status": "ok"}

    # The per-IP limit runs first, so a client already blocked does not use up the daily cap.
    @app.post("/ask", dependencies=[Depends(enforce_rate_limit), Depends(enforce_daily_cap)])
    def post_ask(request: AskRequest, ask_fn: Annotated[AskFn, Depends(get_ask_fn)]) -> AskResponse:
        # A plain def so FastAPI runs the blocking model and DB calls in a thread pool.
        try:
            return ask_fn(request.question)
        except anthropic.APIError as exc:
            logger.error("Model API error: %s", type(exc).__name__)  # never log keys or bodies
            raise HTTPException(status_code=502, detail="The language model call failed.") from exc

    return app


app = create_app()
