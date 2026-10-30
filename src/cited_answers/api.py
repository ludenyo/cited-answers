"""FastAPI application and HTTP routes."""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from functools import partial
from typing import Annotated

import anthropic
from fastapi import Depends, FastAPI, HTTPException

from cited_answers import retrieval
from cited_answers.answer import ask, load_system_prompt, run_tool
from cited_answers.db import connect
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
                client=anthropic.Anthropic(),  # reads ANTHROPIC_API_KEY from the environment
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
    load_system_prompt(get_app_config().answer.prompt_path)  # raises if the file is gone
    yield


def create_app() -> FastAPI:
    """Build the FastAPI app.

    A factory (rather than a module-level app) keeps tests free to build a
    fresh instance with their own settings later on.
    """
    app = FastAPI(title="cited-answers", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        # Liveness only for now; a database ping will be added once the DB layer exists.
        return {"status": "ok"}

    @app.post("/ask")
    def post_ask(request: AskRequest, ask_fn: Annotated[AskFn, Depends(get_ask_fn)]) -> AskResponse:
        # A plain def so FastAPI runs the blocking model and DB calls in a thread pool.
        try:
            return ask_fn(request.question)
        except anthropic.APIError as exc:
            logger.error("Model API error: %s", type(exc).__name__)  # never log keys or bodies
            raise HTTPException(status_code=502, detail="The language model call failed.") from exc

    return app


app = create_app()
