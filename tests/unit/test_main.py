"""Tests for main module: _format_admin_reservation_message and _chat_id_from_update."""

from __future__ import annotations

from unittest.mock import MagicMock

from parking_agent.main import (
    _chat_id_from_update,
    _format_admin_reservation_message,
)


def test_format_admin_reservation_message() -> None:
    """_format_admin_reservation_message formats reservation dict with expected fields."""
    reservation = {
        "customer_name": "John Doe",
        "facility": "airport_parking",
        "date": "2025-03-15",
        "start_time": "14:30",
        "duration_hours": 24,
        "vehicle_plate": "ABC-123",
    }

    result = _format_admin_reservation_message(reservation)

    assert "New parking reservation request" in result
    assert "Customer" in result
    assert "John Doe" in result
    assert "Facility" in result
    assert "airport_parking" in result
    assert "Date" in result
    assert "2025-03-15" in result
    assert "Please review and approve or reject" in result


def test_chat_id_from_update() -> None:
    """_chat_id_from_update returns chat id when effective_chat is present."""
    mock_update = MagicMock()
    mock_chat = MagicMock()
    mock_chat.id = 12345
    mock_update.effective_chat = mock_chat
    mock_update.effective_user = None

    result = _chat_id_from_update(mock_update)

    assert result == "12345"


def test_chat_id_from_update_fallback_to_user() -> None:
    """_chat_id_from_update falls back to user-{id} when chat is None."""
    mock_update = MagicMock()
    mock_update.effective_chat = None
    mock_user = MagicMock()
    mock_user.id = 67890
    mock_update.effective_user = mock_user

    result = _chat_id_from_update(mock_update)

    assert result == "user-67890"
