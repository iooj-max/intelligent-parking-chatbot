"""Tests for tools module: facility validation helpers and ask_clarifying_question."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from parking_agent.tools import (
    _derive_matched_from_results,
    _derive_unresolved_from_results,
    ask_clarifying_question,
    validate_facility_exists,
)


def test_derive_matched_from_results() -> None:
    """Extract matched parking_ids from validation results."""
    results = [
        {"original": "airport", "matched_parking_id": "airport_parking", "matched_name": "Airport"},
        {"original": "downtown", "matched_parking_id": "downtown_plaza", "matched_name": "Downtown"},
    ]
    assert _derive_matched_from_results(results) == ["airport_parking", "downtown_plaza"]


def test_derive_unresolved_from_results() -> None:
    """Extract unresolved originals from validation results."""
    results = [
        {"original": "airport", "matched_parking_id": "airport_parking"},
        {"original": "unknown xyz", "matched_parking_id": ""},
    ]
    assert _derive_unresolved_from_results(results) == ["unknown xyz"]


def test_validate_facility_exists_empty() -> None:
    """Empty facility list returns error."""
    is_valid, reason, matched = validate_facility_exists([])
    assert is_valid is False
    assert "required" in reason.lower()
    assert matched is None


def test_ask_clarifying_question() -> None:
    """ask_clarifying_question returns stripped question text."""
    result = ask_clarifying_question.invoke({"question": "  Which facility?  "})
    assert result == "Which facility?"
