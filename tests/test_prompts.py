from ftpipe.metrics import parse_calls
from ftpipe.prompts import SYSTEM_PROMPT, build_messages, format_target


def test_build_messages_shape():
    tools = [{"name": "get_time", "parameters": {"city": {"type": "str"}}}]
    msgs = build_messages("what time in Paris", tools)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == SYSTEM_PROMPT
    assert "what time in Paris" in msgs[1]["content"]
    assert "get_time" in msgs[1]["content"]


def test_system_prompt_states_the_contract():
    assert "JSON array" in SYSTEM_PROMPT
    assert "[]" in SYSTEM_PROMPT


def test_format_target_round_trips():
    calls = [{"name": "a", "arguments": {"x": 1}}, {"name": "b", "arguments": {}}]
    text = format_target(calls)
    assert parse_calls(text) == calls


def test_format_target_empty():
    assert format_target([]) == "[]"
    assert parse_calls(format_target([])) == []
