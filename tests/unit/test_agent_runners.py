"""Tests for agent_runners module: _extract_clarifying_question and _extract_final_agent_text."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from parking_agent.agent_runners import (
    _extract_clarifying_question_from_tool_call,
    _extract_final_agent_text,
)


def test_extract_clarifying_question_from_tool_call() -> None:
    """Extract question from ask_clarifying_question tool call dict."""
    tool_call = {
        "name": "ask_clarifying_question",
        "args": {"question": "Which facility?"},
    }
    assert _extract_clarifying_question_from_tool_call(tool_call) == "Which facility?"


def test_extract_clarifying_question_wrong_tool_returns_none() -> None:
    """Non-ask_clarifying_question tool call returns None."""
    tool_call = {"name": "retrieve_static_parking_info", "args": {"query": "x"}}
    assert _extract_clarifying_question_from_tool_call(tool_call) is None


def test_extract_final_agent_text() -> None:
    """Extract final text from agent output with AIMessage."""
    agent_output = {
        "messages": [
            AIMessage(content="Here is the parking information."),
        ],
    }
    assert _extract_final_agent_text(agent_output) == "Here is the parking information."


def test_extract_final_agent_text_prefers_latest_ai_message() -> None:
    """Extract text from the most recent AI message when multiple exist."""
    agent_output = {
        "messages": [
            AIMessage(content="First response"),
            AIMessage(content="Final response"),
        ],
    }
    assert _extract_final_agent_text(agent_output) == "Final response"
