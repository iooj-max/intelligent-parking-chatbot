"""Pytest-based load smoke tests for core runtime components."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

import parking_agent.mcp_reservation_status as status_mod
from parking_agent.main import (
    _invoke_execution_graph_for_text,
    _resume_reservation_thread_with_admin_decision,
)


class _FakeExecutionGraph:
    def invoke(self, state_input, config=None):
        _ = state_input, config
        return {
            "messages": [AIMessage(content="Parking info response")],
            "__interrupt__": [],
            "awaiting_user_confirmation": False,
            "conversation_summary": "summary",
        }


class _FakeResumeGraph:
    def invoke(self, command, config=None):
        _ = command, config
        return {"messages": [AIMessage(content="approved")]}


@pytest.mark.load
@pytest.mark.system
def test_load_interactive_dialogue_mode_smoke() -> None:
    """Repeated execution graph invocations stay responsive."""
    execution_graph = _FakeExecutionGraph()
    start = time.perf_counter()
    runs = 120

    for _ in range(runs):
        text, interrupted, reservation, _, awaiting_confirmation = _invoke_execution_graph_for_text(
            execution_graph_app=execution_graph,
            shared_messages=[],
            shared_summary="",
            user_input="What are parking hours?",
            thread_id="tg:42:info",
            conversation_id="tg:42",
            intent="info_retrieval",
        )
        assert text == "Parking info response"
        assert interrupted is False
        assert reservation is None
        assert awaiting_confirmation is False

    elapsed = time.perf_counter() - start
    assert elapsed < 2.0


@pytest.mark.load
@pytest.mark.system
def test_load_admin_confirmation_resume_smoke() -> None:
    """Admin decision resume operation handles repeated calls."""
    graph_app = _FakeResumeGraph()
    start = time.perf_counter()
    runs = 150

    for _ in range(runs):
        output = _resume_reservation_thread_with_admin_decision(
            graph_app=graph_app,
            thread_id="tg:42:reservation",
            conversation_id="tg:42",
            admin_decision="approved",
        )
        assert output == "approved"

    elapsed = time.perf_counter() - start
    assert elapsed < 2.0


@pytest.mark.load
@pytest.mark.system
def test_load_mcp_recording_storage_smoke(monkeypatch) -> None:
    """MCP status append flow handles repeated writes with mocked I/O."""
    storage: dict[str, str] = {}

    async def fake_call_filesystem_tool(name: str, arguments: dict):
        if name == "write_file":
            storage[arguments["path"]] = arguments["content"]
        return SimpleNamespace(isError=False, content=[])

    async def fake_read_status_file(path: str) -> str:
        return storage.get(path, "")

    monkeypatch.setattr(status_mod, "_call_filesystem_tool", fake_call_filesystem_tool)
    monkeypatch.setattr(status_mod, "_read_status_file", fake_read_status_file)
    monkeypatch.setattr(status_mod, "_current_timestamp_iso", lambda: "2026-03-01T12:00:00+00:00")

    async def _run() -> None:
        await status_mod.append_reservation_status(
            thread_id="tg:42:reservation",
            status="pending",
            reservation={
                "customer_name": "John",
                "facility": "airport_parking",
                "date": "2026-03-20",
                "start_time": "10:00",
                "duration_hours": 2,
                "vehicle_plate": "ABC-123",
            },
        )
        for _ in range(80):
            await status_mod.append_reservation_status(
                thread_id="tg:42:reservation",
                status="approved",
            )

    start = time.perf_counter()
    asyncio.run(_run())
    elapsed = time.perf_counter() - start
    assert elapsed < 2.5
    latest = asyncio.run(status_mod.get_latest_reservation_status("tg:42:reservation"))
    assert latest == "approved"


@pytest.mark.load
@pytest.mark.system
def test_load_orchestration_step_chain_smoke() -> None:
    """Sequential orchestration helper operations remain stable under repetition."""
    start = time.perf_counter()
    runs = 200

    for _ in range(runs):
        # Simulate end-to-end helper-level orchestration chaining.
        output = _resume_reservation_thread_with_admin_decision(
            graph_app=_FakeResumeGraph(),
            thread_id="tg:7:reservation",
            conversation_id="tg:7",
            admin_decision="rejected",
        )
        assert output == "approved"

    elapsed = time.perf_counter() - start
    assert elapsed < 2.5
