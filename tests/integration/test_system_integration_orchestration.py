"""Integration-style orchestration tests for routing/execution/admin flow."""

from __future__ import annotations

from langchain_core.messages import AIMessage
import pytest

from parking_agent.main import (
    _invoke_execution_graph_for_text,
    _invoke_routing_graph_for_text,
    _resume_reservation_thread_with_admin_decision,
)


class _FakeRoutingGraph:
    def __init__(self, payload: dict):
        self._payload = payload

    def invoke(self, state_input, config=None):
        _ = state_input, config
        return self._payload


class _FakeExecutionGraph:
    def __init__(self, payload: dict):
        self._payload = payload

    def invoke(self, state_input, config=None):
        _ = state_input, config
        return self._payload


class _FakeResumeGraph:
    def invoke(self, command, config=None):
        _ = command, config
        return {"messages": [AIMessage(content="Reservation request approved by administrator.")]}


@pytest.mark.integration
@pytest.mark.system
def test_orchestration_reservation_path_with_admin_resume() -> None:
    """Routing -> reservation execution interrupt -> admin resume produces final text."""
    routing_graph = _FakeRoutingGraph(
        {"scope_decision": "in_scope", "intent": "reservation", "messages": []}
    )
    scope, intent, response = _invoke_routing_graph_for_text(
        routing_graph_app=routing_graph,
        shared_messages=[],
        shared_summary="",
        user_input="I want to book parking for tomorrow",
        thread_id="tg:1:info",
        conversation_id="tg:1",
    )
    assert scope == "in_scope"
    assert intent == "reservation"
    assert response == ""

    execution_graph = _FakeExecutionGraph(
        {
            "__interrupt__": [{"value": "Awaiting administrator decision"}],
            "reservation": {
                "customer_name": "John",
                "facility": "airport_parking",
                "date": "2026-03-20",
                "start_time": "10:00",
                "duration_hours": 2,
            },
            "messages": [],
            "awaiting_user_confirmation": False,
            "conversation_summary": "summary",
        }
    )
    text, interrupted, reservation, updated_summary, awaiting_confirmation = (
        _invoke_execution_graph_for_text(
            execution_graph_app=execution_graph,
            shared_messages=[],
            shared_summary="",
            user_input="John, airport parking, 2026-03-20, 10:00, 2 hours",
            thread_id="tg:1:reservation",
            conversation_id="tg:1",
            intent="reservation",
        )
    )
    assert interrupted is True
    assert text == ""
    assert reservation is not None
    assert reservation["facility"] == "airport_parking"
    assert awaiting_confirmation is False
    assert updated_summary == "summary"

    final_text = _resume_reservation_thread_with_admin_decision(
        graph_app=_FakeResumeGraph(),
        thread_id="tg:1:reservation",
        conversation_id="tg:1",
        admin_decision="approved",
    )
    assert "approved" in final_text.lower()


@pytest.mark.integration
@pytest.mark.system
def test_orchestration_out_of_scope_short_circuit() -> None:
    """Out-of-scope route returns direct refusal text from routing graph."""
    routing_graph = _FakeRoutingGraph(
        {
            "scope_decision": "out_of_scope",
            "intent": None,
            "messages": [AIMessage(content="This request is out of scope.")],
        }
    )
    scope, intent, response = _invoke_routing_graph_for_text(
        routing_graph_app=routing_graph,
        shared_messages=[],
        shared_summary="",
        user_input="Tell me a joke",
        thread_id="tg:1:info",
        conversation_id="tg:1",
    )
    assert scope == "out_of_scope"
    assert intent is None
    assert response == "This request is out of scope."
