"""Tests for schemas module: Pydantic models validation."""

from __future__ import annotations

from parking_agent.schemas import (
    ReservationExtraction,
    ScopeDecision,
)


def test_scope_decision_valid() -> None:
    """ScopeDecision parses valid in_scope and out_of_scope values."""
    in_scope = ScopeDecision(scope_decision="in_scope", reasoning="Parking question.")
    assert in_scope.scope_decision == "in_scope"
    assert in_scope.reasoning == "Parking question."

    out_scope = ScopeDecision(scope_decision="out_of_scope", reasoning="Off-topic.")
    assert out_scope.scope_decision == "out_of_scope"


def test_reservation_extraction_partial() -> None:
    """ReservationExtraction parses with optional fields as None."""
    extraction = ReservationExtraction(customer_name="John", facility=None)
    assert extraction.customer_name == "John"
    assert extraction.facility is None
    assert extraction.date is None
    assert extraction.start_time is None
    assert extraction.duration_hours is None


def test_reservation_extraction_full() -> None:
    """ReservationExtraction parses all fields."""
    extraction = ReservationExtraction(
        customer_name="Jane",
        facility="airport_parking",
        date="2025-03-15",
        start_time="14:30",
        duration_hours=24,
        vehicle_plate="ABC-123",
    )
    assert extraction.customer_name == "Jane"
    assert extraction.facility == "airport_parking"
    assert extraction.date == "2025-03-15"
    assert extraction.start_time == "14:30"
    assert extraction.duration_hours == 24
    assert extraction.vehicle_plate == "ABC-123"
