"""Tests for LangSmith trace export helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import UUID

import parking_agent.fetch_trace as fetch_trace_mod


def test_serialize_handles_datetime_uuid_and_nested_values() -> None:
    """_serialize converts nested SDK-like values to JSON-safe primitives."""
    payload = {
        "dt": datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc),
        "id": UUID("12345678-1234-5678-1234-567812345678"),
        "nested": [{"x": 1}, {"y": {1, 2}}],
    }
    result = fetch_trace_mod._serialize(payload)

    assert result["dt"] == "2026-03-01T12:00:00+00:00"
    assert result["id"] == "12345678-1234-5678-1234-567812345678"
    assert isinstance(result["nested"], list)


def test_fetch_trace_merges_and_sorts_runs(monkeypatch) -> None:
    """fetch_trace deduplicates root run and sorts by start time."""
    first = SimpleNamespace(
        id="a",
        trace_id="12345678-1234-5678-1234-567812345678",
        name="run-a",
        run_type="chain",
        status="success",
        error=None,
        start_time=datetime(2026, 3, 1, 12, 2, tzinfo=timezone.utc),
        end_time=None,
        inputs={},
        outputs={},
        metadata={},
        tags=[],
        parent_run_id=None,
        child_run_ids=[],
        events=[],
        extra={},
    )
    second = SimpleNamespace(
        id="b",
        trace_id="12345678-1234-5678-1234-567812345678",
        name="run-b",
        run_type="llm",
        status="success",
        error=None,
        start_time=datetime(2026, 3, 1, 12, 1, tzinfo=timezone.utc),
        end_time=None,
        inputs={},
        outputs={},
        metadata={},
        tags=[],
        parent_run_id=None,
        child_run_ids=[],
        events=[],
        extra={},
    )

    class FakeClient:
        def read_run(self, trace_id: str):
            assert trace_id == "12345678-1234-5678-1234-567812345678"
            return first

        def list_runs(self, trace_id: str):
            assert trace_id == "12345678-1234-5678-1234-567812345678"
            # Includes duplicated root id "a"
            return [second, first]

    monkeypatch.setattr(fetch_trace_mod, "Client", lambda: FakeClient())
    data = fetch_trace_mod.fetch_trace("12345678-1234-5678-1234-567812345678")

    assert data["run_count"] == 2
    assert [run["id"] for run in data["runs"]] == ["b", "a"]
