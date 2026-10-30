"""Rubric grader for llm-rubric assertions, run on the same Anthropic key as the pipeline.

promptfoo's own Anthropic provider sends a temperature, which Haiku 5.5 rejects with a 400
(see CLAUDE.md), so the grader calls the SDK directly and sends none.
"""

from typing import Any

import anthropic

from cited_answers.settings import Settings

MAX_TOKENS = 1024  # thinking tokens count against this


def call_api(prompt: str, options: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    try:
        message = anthropic.Anthropic(max_retries=2).messages.create(
            model=Settings.from_env().guard_model,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:  # noqa: BLE001 - a grader failure must fail the case, not crash
        return {"error": f"{type(exc).__name__}: {exc}"}
    text = "".join(block.text for block in message.content if block.type == "text")
    return {"output": text}
