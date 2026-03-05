"""Message trimming utilities for minimal conversation state.

Keeps only the last N messages for context in the next turn.
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage

RECENT_MESSAGES_TO_KEEP = 5


def trim_to_last_n_messages(
    messages: list[BaseMessage], n: int = RECENT_MESSAGES_TO_KEEP
) -> list[BaseMessage]:
    """Keep only the last n messages in chronological order."""
    if not messages:
        return []
    return messages[-n:]
