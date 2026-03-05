"""Tests for chat history file-backed storage."""

from __future__ import annotations

from parking_agent.chat_history_store import ChatHistoryStore


def test_append_and_read_recent_messages(tmp_path) -> None:
    """Store user/assistant messages and read only latest N items."""
    store = ChatHistoryStore(
        history_dir=tmp_path / "history",
        summary_dir=tmp_path / "summary",
    )
    chat_id = "tg:100/200"

    store.append_user_message(chat_id, "hello")
    store.append_ai_message(chat_id, "hi there")
    store.append_user_message(chat_id, "   ")
    store.append_ai_message(chat_id, "how can I help?")

    recent = store.get_recent_messages(chat_id, limit=2)
    assert len(recent) == 2
    assert str(recent[0].content) == "hi there"
    assert str(recent[1].content) == "how can I help?"


def test_summary_roundtrip_and_safe_chat_id(tmp_path) -> None:
    """Summary is persisted and slash-containing chat ids are sanitized."""
    store = ChatHistoryStore(
        history_dir=tmp_path / "history",
        summary_dir=tmp_path / "summary",
    )
    chat_id = "tg:100/200"

    assert store.get_summary(chat_id) == ""
    store.set_summary(chat_id, "  Short summary.  ")
    assert store.get_summary(chat_id) == "Short summary."

    summary_path = (tmp_path / "summary" / "tg:100_200.txt")
    assert summary_path.exists()
