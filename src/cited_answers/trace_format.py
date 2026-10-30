"""Turn Anthropic messages into the role-labelled format Langfuse renders well.

Langfuse shows a generation's input and output as a chat when messages are in the
OpenAI shape, and shows tool calls as cards only when they sit in an assistant
message's `tool_calls` list with JSON-string arguments. Anthropic uses content
blocks instead, so they are converted here. Pure functions, no SDK calls.
"""

import json
from typing import Any


def _get(block: Any, key: str, default: Any = None) -> Any:
    """Read a field from a dict block or an SDK object block."""
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def assistant_message(content: Any) -> dict[str, Any]:
    """One assistant turn: text, any tool calls, and thinking text if it was returned."""
    if isinstance(content, str):
        return {"role": "assistant", "content": content}
    texts: list[str] = []
    thoughts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in content:
        kind = _get(block, "type")
        if kind == "text":
            texts.append(_get(block, "text", ""))
        elif kind == "thinking" and _get(block, "thinking"):
            thoughts.append(_get(block, "thinking"))
        elif kind == "tool_use":
            tool_calls.append(
                {
                    "id": _get(block, "id"),
                    "type": "function",
                    "function": {
                        "name": _get(block, "name"),
                        "arguments": json.dumps(_get(block, "input", {})),
                    },
                }
            )
    message: dict[str, Any] = {"role": "assistant", "content": "\n".join(texts)}
    if thoughts:
        message["thinking"] = "\n".join(thoughts)
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _user_messages(content: Any) -> list[dict[str, Any]]:
    """A user turn becomes user text plus one `tool` message per tool result."""
    if isinstance(content, str):
        return [{"role": "user", "content": content}]
    converted: list[dict[str, Any]] = []
    for block in content:
        if _get(block, "type") == "tool_result":
            message: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": _get(block, "tool_use_id"),
                "content": _get(block, "content"),
            }
            if _get(block, "is_error"):
                message["is_error"] = True
            converted.append(message)
        elif _get(block, "type") == "text":
            converted.append({"role": "user", "content": _get(block, "text", "")})
    return converted


def messages_for_trace(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The full conversation exactly as the model saw it, system prompt first."""
    converted: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for message in messages:
        if message["role"] == "assistant":
            converted.append(assistant_message(message["content"]))
        else:
            converted.extend(_user_messages(message["content"]))
    return converted


def response_for_trace(response: Any) -> dict[str, Any]:
    return assistant_message(response.content)


def usage_for_trace(response: Any) -> dict[str, int]:
    """Token counts under the keys Langfuse uses to work out cost."""
    return {"input": response.usage.input_tokens, "output": response.usage.output_tokens}
