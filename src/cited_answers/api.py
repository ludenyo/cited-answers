"""FastAPI application and HTTP routes."""

import logging
import math
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from functools import partial
from typing import Annotated

import anthropic
from fastapi import Depends, FastAPI, HTTPException, Request

from cited_answers import guardrails, retrieval, tracing
from cited_answers.answer import ask, load_system_prompt, run_tool
from cited_answers.db import connect
from cited_answers.guardrails import RateLimiter
from cited_answers.models import AskRequest, AskResponse
from cited_answers.settings import Settings, get_app_config

logger = logging.getLogger(__name__)

AskFn = Callable[[str], AskResponse]


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
    yield
    tracing.shutdown_tracing()  # send queued traces before exiting


def client_key(request: Request) -> str:
    """Who is asking, for rate limiting: the address the connection came from.

    Behind a reverse proxy this is the proxy's address, so every user would share one
    limit. If this is deployed behind one, read the forwarded address it sets instead,
    and only from a proxy you trust, because clients can forge that header.
    """
    return request.client.host if request.client else "unknown"


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


def create_app(rate_limiter: RateLimiter | None = None) -> FastAPI:
    """Build the FastAPI app.

    A factory (rather than a module-level app) keeps tests free to build a
    fresh instance with their own settings, such as a tighter rate limit.
    """
    app = FastAPI(title="cited-answers", lifespan=lifespan)
    limits = get_app_config().guardrails.rate_limit
    app.state.rate_limiter = rate_limiter or RateLimiter(limits.requests, limits.window_seconds)

    @app.get("/health")
    def health() -> dict[str, str]:
        # Liveness only for now; a database ping will be added once the DB layer exists.
        return {"status": "ok"}

    @app.post("/ask", dependencies=[Depends(enforce_rate_limit)])
    def post_ask(request: AskRequest, ask_fn: Annotated[AskFn, Depends(get_ask_fn)]) -> AskResponse:
        # A plain def so FastAPI runs the blocking model and DB calls in a thread pool.
        try:
            return ask_fn(request.question)
        except anthropic.APIError as exc:
            logger.error("Model API error: %s", type(exc).__name__)  # never log keys or bodies
            raise HTTPException(status_code=502, detail="The language model call failed.") from exc

    return app


app = create_app()
