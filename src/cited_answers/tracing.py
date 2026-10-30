"""Langfuse tracing that can never break a request.

Every call into the Langfuse SDK goes through this module and is wrapped so that a
failure (Langfuse down, bad keys, an SDK bug) is logged and then ignored. Exceptions
raised by the code being traced are never swallowed or altered.

Tracing is off when the keys are missing or LANGFUSE_TRACING_ENABLED=false, in which
case every function here is a cheap no-op.
"""

import logging
import os
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

# The Langfuse client once init_tracing() has succeeded, else None (tracing off).
_client: Any | None = None

# The Anthropic SDK 2.x emits its own OpenTelemetry span for every messages.create call,
# and Langfuse exports it as a second generation with the same tokens. The call-model
# span in answer.py already records that call (with the messages, which the SDK span
# lacks), so exporting both would double every token and cost total.
_NATIVE_ANTHROPIC_SCOPE_PREFIX = "com.anthropic"

_SECRET = re.compile(r"(?:sk-ant|sk-lf|pk-lf)-[A-Za-z0-9_\-]{8,}")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_MAX_STATUS_MESSAGE = 200


class _NullObservation:
    """Stands in for an observation when tracing is off or failed to start."""

    def update(self, **kwargs: Any) -> None:
        return None


_NULL = _NullObservation()


class _SafeObservation:
    """Wraps a real SDK observation so that update() can never raise."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def update(self, **kwargs: Any) -> None:
        try:
            self._inner.update(**kwargs)
        except Exception:
            logger.warning("Tracing: could not update observation", exc_info=True)


Observation = _SafeObservation | _NullObservation


def tracing_enabled() -> bool:
    """True when both Langfuse keys are set and tracing has not been switched off."""
    if os.environ.get("LANGFUSE_TRACING_ENABLED", "true").strip().lower() == "false":
        return False
    return bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
        and os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    )


def mask_sensitive(*, data: Any, **kwargs: Any) -> Any:
    """Redact API keys and email addresses from anything sent to Langfuse.

    Questions are free text typed by users, so they can contain personal data;
    nothing in this app needs it in a trace. The World Bank figures are public.
    """
    if isinstance(data, str):
        return _EMAIL.sub("[email]", _SECRET.sub("[secret]", data))
    if isinstance(data, dict):
        return {key: mask_sensitive(data=value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [mask_sensitive(data=item) for item in data]
    return data


def should_export_span(span: Any) -> bool:
    """Langfuse's default export rule, minus the SDK's duplicate Anthropic spans."""
    from langfuse import is_default_export_span

    scope = getattr(span, "instrumentation_scope", None)
    if scope is not None and (scope.name or "").startswith(_NATIVE_ANTHROPIC_SCOPE_PREFIX):
        return False
    return bool(is_default_export_span(span))


def create_client(**overrides: Any) -> Any:
    """Build the Langfuse client exactly as production does. Tests add an exporter.

    Host, environment and timeout come from LANGFUSE_HOST, LANGFUSE_TRACING_ENVIRONMENT
    and LANGFUSE_TIMEOUT. Export happens on a background thread, so a slow or
    unreachable Langfuse never blocks a request.

    The mask covers input and output but NOT metadata (checked against SDK 5.1), so
    user text and secrets must never be put in metadata.
    """
    from langfuse import Langfuse  # imported here so a broken install cannot stop startup

    return Langfuse(mask=mask_sensitive, should_export_span=should_export_span, **overrides)


def init_tracing() -> bool:
    """Create the Langfuse client. Returns whether tracing is on. Never raises."""
    global _client
    if _client is not None:
        return True
    if not tracing_enabled():
        logger.info("Tracing is off: Langfuse keys not set or LANGFUSE_TRACING_ENABLED=false")
        return False
    try:
        _client = create_client()
    except Exception:
        logger.warning("Tracing is off: could not start the Langfuse client", exc_info=True)
        _client = None
        return False
    logger.info("Tracing is on")
    return True


def shutdown_tracing() -> None:
    """Send anything still queued, then stop the background exporter. Never raises."""
    global _client
    client, _client = _client, None
    if client is None:
        return
    try:
        client.shutdown()
    except Exception:
        logger.warning("Tracing: could not flush on shutdown", exc_info=True)


def _exit(manager: Any, error: BaseException | None) -> None:
    if manager is None:
        return
    try:
        if error is None:
            manager.__exit__(None, None, None)
        else:
            manager.__exit__(type(error), error, error.__traceback__)
    except Exception:
        logger.warning("Tracing: could not close a span", exc_info=True)


@contextmanager
def _safely(open_manager: Callable[[], Any], what: str) -> Iterator[Any]:
    """Enter an SDK context manager, yielding its value, or None if that failed.

    The body's own exceptions pass straight through (and are re-raised after the SDK
    has seen them); only failures of the SDK itself are caught.
    """
    manager = None
    value = None
    try:
        manager = open_manager()
        value = manager.__enter__()
    except Exception:
        logger.warning("Tracing: could not start %s", what, exc_info=True)
        manager = None
    try:
        yield value
    except BaseException as error:
        _exit(manager, error)
        raise
    _exit(manager, None)


@contextmanager
def observation(name: str, as_type: str = "span", **kwargs: Any) -> Iterator[Observation]:
    """A span of the given type, nested under the current one.

    kwargs go to Langfuse: input, output, metadata, model, model_parameters,
    usage_details. Yields an object whose update() is always safe to call.
    """
    client = _client
    if client is None:
        yield _NULL
        return
    with _safely(
        lambda: client.start_as_current_observation(name=name, as_type=as_type, **kwargs), name
    ) as inner:
        obs: Observation = _SafeObservation(inner) if inner is not None else _NULL
        try:
            yield obs
        except BaseException as error:
            message = f"{type(error).__name__}: {error}"[:_MAX_STATUS_MESSAGE]
            obs.update(level="ERROR", status_message=message)
            raise


@contextmanager
def trace_attributes(**kwargs: Any) -> Iterator[None]:
    """Attributes (trace_name, tags, metadata, ...) copied onto every span inside.

    Tags cannot be changed once a span exists, so set them here at the start.
    """
    if _client is None:
        yield
        return
    from langfuse import propagate_attributes  # same reason as in init_tracing

    with _safely(lambda: propagate_attributes(**kwargs), "trace attributes"):
        yield


def event(name: str, **kwargs: Any) -> None:
    """Record a point-in-time event under the current span, or as its own trace if there is
    none. kwargs go to Langfuse: input, output, metadata, level, status_message. Never raises.
    """
    if _client is None:
        return
    try:
        _client.create_event(name=name, **kwargs)
    except Exception:
        logger.warning("Tracing: could not record event %s", name, exc_info=True)


def score_current_trace(name: str, value: float, data_type: str) -> None:
    """Attach a score to the trace being recorded. Never raises."""
    if _client is None:
        return
    try:
        _client.score_current_trace(name=name, value=value, data_type=data_type)
    except Exception:
        logger.warning("Tracing: could not record score %s", name, exc_info=True)
