"""Tests for message_reducer module: trim_to_last_n_messages."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from parking_agent.message_reducer import trim_to_last_n_messages


def test_trim_to_last_n_messages() -> None:
    """Keep only the last N messages in chronological order."""
    messages = [
        HumanMessage(content="msg 1"),
        AIMessage(content="msg 2"),
        HumanMessage(content="msg 3"),
        AIMessage(content="msg 4"),
        HumanMessage(content="msg 5"),
    ]

    result = trim_to_last_n_messages(messages, n=3)

    assert len(result) == 3
    assert result[0].content == "msg 3"
    assert result[1].content == "msg 4"
    assert result[2].content == "msg 5"


def test_trim_empty_list() -> None:
    """Empty input returns empty list."""
    result = trim_to_last_n_messages([])
    assert result == []
