"""Tests for graph module: validation helpers and _merge_reservation."""

from __future__ import annotations

from datetime import date, timedelta

from parking_agent.schemas import ReservationExtraction

from parking_agent.graph import (
    _is_valid_date,
    _is_valid_time,
    _merge_reservation,
)


def test_is_valid_date() -> None:
    """_is_valid_date accepts YYYY-MM-DD today or future, rejects past and invalid."""
    future = (date.today() + timedelta(days=30)).strftime("%Y-%m-%d")
    assert _is_valid_date(future) is True

    past = "2020-01-01"
    assert _is_valid_date(past) is False

    assert _is_valid_date("invalid") is False
    assert _is_valid_date("2025-13-01") is False  # invalid month


def test_is_valid_time() -> None:
    """_is_valid_time accepts valid HH:MM, rejects invalid."""
    assert _is_valid_time("14:30") is True
    assert _is_valid_time("00:00") is True
    assert _is_valid_time("23:59") is True

    assert _is_valid_time("25:00") is False
    assert _is_valid_time("12:60") is False
    assert _is_valid_time("invalid") is False


def test_merge_reservation() -> None:
    """_merge_reservation merges extracted fields into existing, skipping None."""
    existing: dict = {"customer_name": "Old", "facility": "airport_parking"}
    extracted = ReservationExtraction(
        customer_name="New",
        facility=None,
        date="2025-03-15",
        start_time="14:30",
        duration_hours=24,
    )

    result = _merge_reservation(existing, extracted)

    assert result["customer_name"] == "New"
    assert result["facility"] == "airport_parking"  # unchanged, extracted was None
    assert result["date"] == "2025-03-15"
    assert result["start_time"] == "14:30"
    assert result["duration_hours"] == 24


def test_merge_reservation_empty_facility_does_not_overwrite() -> None:
    """Empty string for facility does not overwrite existing."""
    existing: dict = {"facility": "airport_parking"}
    extracted = ReservationExtraction(facility="", customer_name="John")

    result = _merge_reservation(existing, extracted)

    assert result["facility"] == "airport_parking"
    assert result["customer_name"] == "John"
