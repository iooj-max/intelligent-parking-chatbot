"""Shared chat history and summary storage backed by local files."""

from __future__ import annotations

from pathlib import Path

from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class ChatHistoryStore:
    """Persist conversation history per chat_id using LangChain file history."""

    def __init__(
        self,
        history_dir: Path | None = None,
        summary_dir: Path | None = None,
    ) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self._history_dir = history_dir or project_root / "runtime" / "chat_history"
        self._summary_dir = summary_dir or project_root / "runtime" / "chat_history_summary"
        self._history_dir.mkdir(parents=True, exist_ok=True)
        self._summary_dir.mkdir(parents=True, exist_ok=True)

    def get_recent_messages(self, chat_id: str, limit: int) -> list[BaseMessage]:
        """Return the most recent messages for chat_id."""
        if limit <= 0:
            return []
        history = self._history(chat_id)
        messages = history.messages
        if len(messages) <= limit:
            return messages
        return messages[-limit:]

    def append_user_message(self, chat_id: str, user_text: str) -> None:
        """Append a user message to history."""
        text = user_text.strip()
        if not text:
            return
        self._history(chat_id).add_message(HumanMessage(content=text))

    def append_ai_message(self, chat_id: str, ai_text: str) -> None:
        """Append an assistant message to history."""
        text = ai_text.strip()
        if not text:
            return
        self._history(chat_id).add_message(AIMessage(content=text))

    def get_summary(self, chat_id: str) -> str:
        """Read summary text for chat_id."""
        path = self._summary_path(chat_id)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8").strip()

    def set_summary(self, chat_id: str, summary: str) -> None:
        """Persist summary text for chat_id."""
        path = self._summary_path(chat_id)
        path.write_text((summary or "").strip(), encoding="utf-8")

    def _history(self, chat_id: str) -> FileChatMessageHistory:
        return FileChatMessageHistory(
            file_path=str(self._history_path(chat_id)),
            encoding="utf-8",
            ensure_ascii=False,
        )

    def _history_path(self, chat_id: str) -> Path:
        return self._history_dir / f"{self._safe_chat_id(chat_id)}.json"

    def _summary_path(self, chat_id: str) -> Path:
        return self._summary_dir / f"{self._safe_chat_id(chat_id)}.txt"

    @staticmethod
    def _safe_chat_id(chat_id: str) -> str:
        value = chat_id.strip()
        if not value:
            return "unknown_chat"
        return value.replace("/", "_")
