"""
Tests for facility validation: input list of strings, output FacilityValidationResponse structure.

Uses mocked DB fetch for unit tests. Requires LLM (OpenAI) and .env with OPENAI_API_KEY.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from parking_agent.facility_validation import validate_facility


FACILITIES_TUPLES: list[tuple[str, str, str, str]] = [
    ("airport_parking", "Airport Long-Term Parking", "4500 Airport Boulevard", "Mobile"),
    ("downtown_plaza", "Downtown Plaza Parking", "123 Main Street", "Saint Julian's"),
]


@pytest.fixture(autouse=True)
def mock_fetch():
    """Mock DB fetch to avoid PostgreSQL dependency."""
    with patch(
        "parking_agent.facility_validation._fetch_parking_facilities",
        return_value=FACILITIES_TUPLES,
    ):
        yield


def _assert_structure(result: dict) -> None:
    """Assert result has expected FacilityValidationResponse structure."""
    assert "status" in result
    assert result["status"] in ("ok", "error")
    assert "results" in result
    assert isinstance(result["results"], list)
    assert "is_valid" in result
    assert isinstance(result["is_valid"], bool)
    assert "reason" in result
    assert isinstance(result["reason"], str)
    for item in result["results"]:
        assert isinstance(item, dict)
        assert "original" in item
        assert "matched_parking_id" in item
        assert "matched_name" in item
        assert "matched_address" in item
        assert "matched_city" in item


def _matched_ids(result: dict) -> list[str]:
    """Extract matched parking_ids from results."""
    return [
        str(r.get("matched_parking_id", "")).strip()
        for r in result.get("results") or []
        if isinstance(r, dict) and str(r.get("matched_parking_id", "")).strip()
    ]


def _unresolved(result: dict) -> list[str]:
    """Extract unresolved originals from results."""
    return [
        str(r.get("original", "")).strip()
        for r in result.get("results") or []
        if isinstance(r, dict) and not str(r.get("matched_parking_id", "")).strip()
    ]


@pytest.mark.parametrize(
    "input_list,expected_matched_contains,expected_unresolved_empty,expected_is_valid",
    [
        (["airport parking"], ["airport_parking"], True, True),
        (["парковка аэропорта"], ["airport_parking"], True, True),
        (["даунтаун парковка"], ["downtown_plaza"], True, True),
        (["даунтаун плаза"], ["downtown_plaza"], True, True),
        (["downtown plaza"], ["downtown_plaza"], True, True),
        (["parking in Mobile"], ["airport_parking"], True, True),
        (["parking near 4500 Airport Boulevard"], ["airport_parking"], True, True),
        (["парковка в Saint Julian's"], ["downtown_plaza"], True, True),
        (["airport parking", "downtown plaza"], ["airport_parking", "downtown_plaza"], True, True),
        (["airport parking", "xyz unknown"], ["airport_parking"], False, False),
        (["nonexistent xyz"], [], False, False),
    ],
)
def test_validate_facility_structure_and_content(
    input_list: list[str],
    expected_matched_contains: list[str],
    expected_unresolved_empty: bool,
    expected_is_valid: bool,
) -> None:
    """Assert validate_facility returns correct structure and expected content."""
    result = validate_facility(input_list)
    _assert_structure(result)

    matched = _matched_ids(result)
    unresolved = _unresolved(result)

    if expected_matched_contains:
        for pid in expected_matched_contains:
            assert pid in matched, f"Expected {pid} in matched, got {matched}"
    if expected_unresolved_empty:
        assert len(unresolved) == 0
    else:
        assert len(unresolved) > 0
    assert result["is_valid"] == expected_is_valid


def test_validate_facility_empty_list() -> None:
    """Empty input returns valid structure with empty results."""
    result = validate_facility([])
    _assert_structure(result)
    assert result["status"] == "ok"
    assert result["results"] == []
    assert result["is_valid"] is False


def test_validate_facility_structure_only() -> None:
    """Smoke test: any non-empty input returns valid structure."""
    result = validate_facility(["airport_parking"])
    _assert_structure(result)


def test_validate_facility_results_have_original_and_matched_fields() -> None:
    """Each result item has original, matched_parking_id, matched_name, matched_address, matched_city."""
    result = validate_facility(["airport_parking"])
    _assert_structure(result)
    assert len(result["results"]) == 1
    item = result["results"][0]
    assert item["original"] == "airport_parking"
    assert item["matched_parking_id"] == "airport_parking"
    assert item["matched_name"]  # non-empty when matched
    assert "matched_address" in item
    assert "matched_city" in item
