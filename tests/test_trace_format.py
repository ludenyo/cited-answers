import json
from types import SimpleNamespace

from cited_answers.trace_format import (
    assistant_message,
    messages_for_trace,
    response_for_trace,
    usage_for_trace,
)


def _tool_use(call_id: str, name: str, tool_input: dict) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=tool_input)


def _text(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def test_assistant_tool_use_becomes_openai_style_tool_calls() -> None:
    message = assistant_message(
        [_text("Let me look."), _tool_use("t1", "get_figure", {"year": 2023})]
    )

    assert message["role"] == "assistant"
    assert message["content"] == "Let me look."
    [call] = message["tool_calls"]
    assert call["id"] == "t1" and call["type"] == "function"
    assert call["function"]["name"] == "get_figure"
    assert json.loads(call["function"]["arguments"]) == {"year": 2023}  # a JSON string


def test_assistant_without_tool_calls_has_no_tool_calls_key() -> None:
    assert assistant_message([_text("done")]) == {"role": "assistant", "content": "done"}
    assert assistant_message("plain string") == {"role": "assistant", "content": "plain string"}


def test_thinking_is_kept_only_when_it_has_text() -> None:
    empty = SimpleNamespace(type="thinking", thinking="", signature="x")  # the default: omitted
    shown = SimpleNamespace(type="thinking", thinking="check 2022 first", signature="x")
    assert "thinking" not in assistant_message([empty, _text("a")])
    assert assistant_message([shown, _text("a")])["thinking"] == "check 2022 first"


def test_conversation_is_converted_with_system_first_and_tool_results_as_tool_role() -> None:
    messages = [
        {"role": "user", "content": "What was Kenya's GDP growth in 2023?"},
        {"role": "assistant", "content": [_tool_use("t1", "search_figures", {"question": "x"})]},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": '{"figures": []}'},
                {"type": "tool_result", "tool_use_id": "t2", "content": "boom", "is_error": True},
            ],
        },
    ]

    converted = messages_for_trace("SYSTEM TEXT", messages)

    assert [m["role"] for m in converted] == ["system", "user", "assistant", "tool", "tool"]
    assert converted[0]["content"] == "SYSTEM TEXT"
    assert converted[2]["tool_calls"][0]["id"] == "t1"
    assert converted[3] == {"role": "tool", "tool_call_id": "t1", "content": '{"figures": []}'}
    assert converted[4]["is_error"] is True and converted[4]["tool_call_id"] == "t2"


def test_it_accepts_dict_blocks_as_well_as_sdk_objects() -> None:
    message = assistant_message([{"type": "text", "text": "hi"}])
    assert message["content"] == "hi"


def test_conversion_is_a_snapshot_not_a_live_view() -> None:
    messages: list[dict] = [{"role": "user", "content": "q"}]
    converted = messages_for_trace("s", messages)
    messages.append({"role": "assistant", "content": "later"})
    assert len(converted) == 2


def test_response_and_usage() -> None:
    response = SimpleNamespace(
        content=[_text("ok")], usage=SimpleNamespace(input_tokens=120, output_tokens=30)
    )
    assert response_for_trace(response) == {"role": "assistant", "content": "ok"}
    assert usage_for_trace(response) == {"input": 120, "output": 30}
