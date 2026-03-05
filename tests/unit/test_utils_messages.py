"""Tests for utils.messages module: message_content_to_text."""

from __future__ import annotations

from parking_agent.utils.messages import message_content_to_text


def test_message_content_to_text_str() -> None:
    """String input returns stripped string."""
    assert message_content_to_text("  hello world  ") == "hello world"
    assert message_content_to_text("single") == "single"


def test_message_content_to_text_list() -> None:
    """List of dicts with type 'text' extracts text blocks."""
    content = [
        {"type": "text", "text": "First paragraph"},
        {"type": "image", "url": "http://example.com/img.png"},
        {"type": "text", "text": "Second paragraph"},
    ]
    result = message_content_to_text(content)
    assert result == "First paragraph\nSecond paragraph"


def test_message_content_to_text_empty_list() -> None:
    """Empty list or list without text blocks returns empty string."""
    assert message_content_to_text([]) == ""
    assert message_content_to_text([{"type": "image"}]) == ""


def test_message_content_to_text_other_types() -> None:
    """Non-str and non-list return empty string."""
    assert message_content_to_text(None) == ""
    assert message_content_to_text(123) == ""
