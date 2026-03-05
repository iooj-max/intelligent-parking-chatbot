"""Tests for MCP-backed reservation status storage helpers."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import parking_agent.mcp_reservation_status as status_mod


def test_normalize_status_accepts_known_values() -> None:
    """Known statuses are normalized to lowercase."""
    assert status_mod._normalize_status("PENDING") == "pending"
    assert status_mod._normalize_status(" approved ") == "approved"
    assert status_mod._normalize_status("rejected") == "rejected"


def test_normalize_status_rejects_unknown_value() -> None:
    """Unknown status raises ValueError."""
    with pytest.raises(ValueError, match="must be one of"):
        status_mod._normalize_status("processing")


def test_get_latest_reservation_status_reads_last_line(monkeypatch) -> None:
    """Latest status is extracted from the last valid log line."""

    async def fake_read_status_file(path: str) -> str:
        assert path.endswith("tg:1:reservation.txt")
        return (
            "john|airport|2026-03-10|10:00|2|ABC-123 | pending | 2026-03-01T10:00:00+00:00\n"
            "john|airport|2026-03-10|10:00|2|ABC-123 | approved | 2026-03-01T10:05:00+00:00\n"
        )

    monkeypatch.setattr(status_mod, "_read_status_file", fake_read_status_file)
    result = asyncio.run(status_mod.get_latest_reservation_status("tg:1:reservation"))
    assert result == "approved"


def test_append_reservation_status_writes_combined_log(monkeypatch) -> None:
    """Appending status writes previous content + new entry."""
    recorded_calls: list[tuple[str, dict]] = []

    async def fake_call_filesystem_tool(name: str, arguments: dict):
        recorded_calls.append((name, arguments))
        return SimpleNamespace(isError=False, content=[])

    async def fake_read_status_file(path: str) -> str:
        assert path.endswith("tg:9:reservation.txt")
        return "john|airport_parking|2026-03-11|09:00|3|ABC-111 | pending | 2026-03-01T10:00:00+00:00\n"

    monkeypatch.setattr(status_mod, "_call_filesystem_tool", fake_call_filesystem_tool)
    monkeypatch.setattr(status_mod, "_read_status_file", fake_read_status_file)
    monkeypatch.setattr(
        status_mod,
        "_current_timestamp_iso",
        lambda: "2026-03-01T11:00:00+00:00",
    )

    asyncio.run(status_mod.append_reservation_status("tg:9:reservation", "approved"))

    assert recorded_calls[0][0] == "create_directory"
    assert recorded_calls[1][0] == "write_file"
    payload = recorded_calls[1][1]["content"]
    assert "pending" in payload
    assert "approved" in payload
    assert payload.strip().endswith("approved | 2026-03-01T11:00:00+00:00")
