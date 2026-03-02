"""Message content extraction utilities."""

from __future__ import annotations

from typing import Any


def message_content_to_text(content: Any) -> str:
    """Extract plain text from message content (str or list of content blocks)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_chunks: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_value = str(part.get("text", "")).strip()
                if text_value:
                    text_chunks.append(text_value)
        return "\n".join(text_chunks).strip()
    return ""
