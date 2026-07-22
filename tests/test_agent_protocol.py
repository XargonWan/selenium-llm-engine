"""Unit tests for the engine-agnostic agent protocol harness.

These tests validate detection, prompt building, and the robust parser without
touching Selenium or the HTTP layer.
"""

import json

from core.agent_protocol import (
    ParsedAgentResponse,
    ToolCall,
    build_agent_system_prompt,
    build_agent_turn_reminder,
    build_reformulation_prompt,
    detect_agent_context,
    needs_reformulation,
    parse_agent_response,
    to_openai_tool_calls,
)


_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


# ---------------------------------------------------------------------------
# detect_agent_context
# ---------------------------------------------------------------------------


def test_detect_agent_context_with_tools():
    ctx = detect_agent_context({"tools": [_WEATHER_TOOL]})
    assert ctx is not None
    assert ctx["tools"] == [_WEATHER_TOOL]


def test_detect_agent_context_with_tool_choice():
    ctx = detect_agent_context({"tool_choice": "auto"})
    assert ctx is not None


def test_detect_agent_context_with_response_format():
    ctx = detect_agent_context({"response_format": {"type": "json_object"}})
    assert ctx is not None
    assert ctx["response_format"] == {"type": "json_object"}


def test_detect_agent_context_explicit_mode():
    ctx = detect_agent_context({"mode": "agent"})
    assert ctx is not None


def test_detect_agent_context_plain_request_returns_none():
    assert detect_agent_context({"messages": [{"role": "user", "content": "hi"}]}) is None


def test_detect_agent_context_empty_tools_returns_none():
    assert detect_agent_context({"tools": []}) is None


def test_detect_agent_context_non_dict_returns_none():
    assert detect_agent_context("not a dict") is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# build_agent_system_prompt
# ---------------------------------------------------------------------------


def test_build_agent_system_prompt_lists_tools():
    prompt = build_agent_system_prompt({"tools": [_WEATHER_TOOL]})
    assert "[AGENT MODE" in prompt
    assert "get_weather" in prompt
    assert "tool_calls" in prompt


def test_build_agent_system_prompt_asserts_tools_are_executable():
    """The harness must tell the model the tools are real and will be executed,
    so it does not refuse with 'I can't / no access' like a plain chat."""
    prompt = build_agent_system_prompt({"tools": [_WEATHER_TOOL]}).lower()
    assert "execute" in prompt
    assert "runtime" in prompt
    # Must explicitly forbid the "I can't / no access" style refusals.
    assert "can't" in prompt or "cannot" in prompt or "access" in prompt


def test_build_agent_system_prompt_uses_roleplay_framing():
    """The harness frames the task as a roleplay so guardrail-heavy web UIs
    (even unlogged) stay in character and emit tool calls instead of refusing."""
    prompt = build_agent_system_prompt({"tools": [_WEATHER_TOOL]}).lower()
    assert "roleplay" in prompt
    assert "character" in prompt
    assert "never refuses" in prompt or "never refuse" in prompt


def test_build_agent_turn_reminder_with_tools():
    """The trailing reminder must reinforce the roleplay and demand JSON so the
    contract stays in context at the END of long multi-turn prompts."""
    reminder = build_agent_turn_reminder(has_tools=True).lower()
    assert "character" in reminder
    assert "json" in reminder
    assert "tool_calls" in reminder


def test_build_agent_turn_reminder_without_tools():
    reminder = build_agent_turn_reminder(has_tools=False).lower()
    assert "json" in reminder
    assert "content" in reminder


def test_build_agent_system_prompt_without_tools():
    prompt = build_agent_system_prompt({"tools": []})
    assert "[AGENT MODE]" in prompt
    assert "content" in prompt


def test_build_agent_system_prompt_includes_response_format():
    prompt = build_agent_system_prompt(
        {"tools": [], "response_format": {"type": "json_object"}}
    )
    assert "response_format" in prompt


def test_build_reformulation_prompt_mentions_json():
    assert "JSON" in build_reformulation_prompt()


def test_build_reformulation_prompt_is_self_contained():
    """When given the original harness, the retry must re-include it verbatim
    so the stateless engine keeps the tool list and task in context."""
    harness = build_agent_system_prompt(
        {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
        }
    ) + "What is the weather in Rome?"
    retry = build_reformulation_prompt(harness)
    assert harness in retry
    assert "get_weather" in retry
    assert "What is the weather in Rome?" in retry
    assert "JSON" in retry


# ---------------------------------------------------------------------------
# parse_agent_response
# ---------------------------------------------------------------------------


def test_parse_fenced_tool_calls():
    text = (
        '```json\n{"tool_calls": [{"name": "get_weather", '
        '"arguments": {"city": "Rome"}}]}\n```'
    )
    parsed = parse_agent_response(text)
    assert parsed.parsed_ok is True
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0].name == "get_weather"
    assert parsed.tool_calls[0].arguments == {"city": "Rome"}


def test_parse_bare_json_object_content():
    parsed = parse_agent_response('{"content": "hello"}')
    assert parsed.parsed_ok is True
    assert parsed.content == "hello"
    assert parsed.tool_calls == []


def test_parse_dirty_json_with_trailing_commas():
    text = '```json\n{"content": "ok",}\n```'
    parsed = parse_agent_response(text)
    assert parsed.parsed_ok is True
    assert parsed.content == "ok"


def test_parse_openai_function_shape():
    text = (
        '```json\n{"tool_calls": [{"id": "call_x", "type": "function", '
        '"function": {"name": "get_weather", '
        '"arguments": "{\\"city\\": \\"Rome\\"}"}}]}\n```'
    )
    parsed = parse_agent_response(text)
    assert parsed.parsed_ok is True
    assert parsed.tool_calls[0].name == "get_weather"
    assert parsed.tool_calls[0].arguments == {"city": "Rome"}
    assert parsed.tool_calls[0].id == "call_x"


def test_parse_json_embedded_in_prose():
    text = 'Sure! Here you go:\n{"content": "done"}\nHope that helps.'
    parsed = parse_agent_response(text)
    assert parsed.parsed_ok is True
    assert parsed.content == "done"


def test_parse_pure_prose_not_ok():
    parsed = parse_agent_response("I would call get_weather for Rome.")
    assert parsed.parsed_ok is False
    assert parsed.tool_calls == []


def test_parse_empty_text_not_ok():
    parsed = parse_agent_response("")
    assert parsed.parsed_ok is False


def test_parse_actions_alias_key():
    text = '```json\n{"actions": [{"name": "do_it", "arguments": {}}]}\n```'
    parsed = parse_agent_response(text)
    assert parsed.parsed_ok is True
    assert parsed.tool_calls[0].name == "do_it"


def test_parse_json_object_without_content_becomes_content():
    text = '```json\n{"ok": true, "value": 42}\n```'
    parsed = parse_agent_response(text)
    assert parsed.parsed_ok is True
    assert json.loads(parsed.content) == {"ok": True, "value": 42}


# ---------------------------------------------------------------------------
# needs_reformulation / to_openai_tool_calls
# ---------------------------------------------------------------------------


def test_needs_reformulation():
    assert needs_reformulation(ParsedAgentResponse(parsed_ok=False)) is True
    assert needs_reformulation(ParsedAgentResponse(parsed_ok=True, content="x")) is False


def test_to_openai_tool_calls_shape():
    calls = [ToolCall(id="call_0", name="get_weather", arguments={"city": "Rome"})]
    out = to_openai_tool_calls(calls)
    assert out[0]["id"] == "call_0"
    assert out[0]["type"] == "function"
    assert out[0]["function"]["name"] == "get_weather"
    assert json.loads(out[0]["function"]["arguments"]) == {"city": "Rome"}
