"""promptfoo provider: calls the ask pipeline directly, with no HTTP server in between.

Wired the same way as api.get_ask_fn: one DB connection and one model client per call.
Tracing is left to the environment; CI sets LANGFUSE_TRACING_ENABLED=false.
"""

import json
import sys
from functools import partial
from pathlib import Path
from typing import Any

import anthropic

sys.path.insert(0, str(Path(__file__).parent))  # so the assertion modules sit next to this file

from cited_answers import retrieval  # noqa: E402
from cited_answers.answer import ask, run_tool  # noqa: E402
from cited_answers.db import connect  # noqa: E402
from cited_answers.settings import Settings, get_app_config  # noqa: E402


def call_api(prompt: str, options: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """prompt is the question text. Failures come back as an error, not an exception."""
    try:
        with connect() as conn:
            response = ask(
                prompt,
                client=anthropic.Anthropic(max_retries=0),  # same as the API: keeps the deadline
                run=partial(run_tool, conn),
                lookup=partial(retrieval.get_figure_by_id, conn),
                settings=Settings.from_env(),
                config=get_app_config(),
            )
    except Exception as exc:  # noqa: BLE001 - one bad case must not stop the whole suite
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "output": response.model_dump_json(),
        "tokenUsage": {
            "prompt": response.input_tokens,
            "completion": response.output_tokens,
            "total": response.input_tokens + response.output_tokens,
        },
    }


if __name__ == "__main__":  # a quick manual check: python provider.py "a question"
    print(json.dumps(call_api(sys.argv[1], {}, {}), indent=2))
