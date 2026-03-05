"""Deterministic facility validation: DB fetch + exact match + LLM fallback."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI
from sqlalchemy import text
from sqlalchemy.engine import create_engine

from parking_agent.clients import build_postgres_uri
from parking_agent.prompts import facility_validation_prompt
from parking_agent.schemas import FacilityValidationResponse
from src.config import settings


def _normalize(s: str) -> str:
    """Normalize for case-insensitive exact match."""
    return (s or "").strip().lower()


def _fetch_parking_facilities() -> list[tuple[str, str, str, str]]:
    """Fetch (parking_id, name, address, city) from DB. No user input in query."""
    engine = create_engine(build_postgres_uri())
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT parking_id, name, address, city FROM public.parking_facilities ORDER BY parking_id"
                )
            )
            rows = result.fetchall()
    finally:
        engine.dispose()

    out: list[tuple[str, str, str, str]] = []
    for row in rows:
        parking_id = str(row[0]).strip() if row[0] is not None else ""
        name = str(row[1]).strip() if row[1] is not None else ""
        address = str(row[2]).strip() if len(row) > 2 and row[2] is not None else ""
        city = str(row[3]).strip() if len(row) > 3 and row[3] is not None else ""
        out.append((parking_id, name, address, city))
    return out


def _exact_match(user_str: str, facilities: list[tuple[str, str, str, str]]) -> tuple[str, str, str, str] | None:
    """Return (parking_id, name, address, city) if user_str exactly matches any field, else None."""
    norm = _normalize(user_str)
    if not norm:
        return None
    for parking_id, name, address, city in facilities:
        if norm == _normalize(parking_id) or norm == _normalize(name) or norm == _normalize(address) or norm == _normalize(city):
            return (parking_id, name, address, city)
    return None


def get_facility_display_name(parking_id: str) -> str:
    """Return human-readable facility name for a parking_id, or parking_id if not found."""
    if not (parking_id or "").strip():
        return ""
    try:
        facilities = _fetch_parking_facilities()
        norm = _normalize(parking_id)
        for pid, name, _addr, _city in facilities:
            if _normalize(pid) == norm and name:
                return name
    except Exception:
        logging.getLogger(__name__).exception("Failed to fetch facility name for parking_id=%s", parking_id)
    return str(parking_id).strip()


def _facilities_to_text(facilities: list[tuple[str, str, str, str]]) -> str:
    """Convert facilities list to text for LLM prompt."""
    lines = ["parking_id | name | address | city", "---"]
    for parking_id, name, address, city in facilities:
        lines.append(f"{parking_id} | {name} | {address} | {city}")
    return "\n".join(lines)


def validate_facility(facility_value: list[str]) -> dict[str, Any]:
    """Validate facility strings against DB. Returns FacilityValidationResponse as dict."""
    filtered = [s for s in facility_value if s and str(s).strip()]

    try:
        facilities = _fetch_parking_facilities()
    except Exception:
        logging.getLogger(__name__).exception("Failed to fetch parking facilities")
        return {
            "status": "error",
            "results": [],
            "is_valid": False,
            "reason": "Facility validation failed.",
        }

    if not filtered:
        return {
            "status": "ok",
            "results": [],
            "is_valid": False,
            "reason": "No facility strings provided.",
        }

    # Exact match first: city, name, address, parking_id (case-insensitive)
    results: list[dict[str, Any]] = []
    all_matched = True
    for user_str in filtered:
        match = _exact_match(user_str, facilities)
        if match:
            parking_id, name, address, city = match
            results.append({
                "original": user_str,
                "matched_parking_id": parking_id,
                "matched_name": name,
                "matched_address": address,
                "matched_city": city,
            })
        else:
            all_matched = False
            results.append({
                "original": user_str,
                "matched_parking_id": "",
                "matched_name": "",
                "matched_address": "",
                "matched_city": "",
            })

    if all_matched:
        return {
            "status": "ok",
            "results": results,
            "is_valid": True,
            "reason": "",
        }

    # Fallback to LLM for fuzzy/synonym/translation matching
    facilities_text = _facilities_to_text(facilities)
    facility_strings = json.dumps(filtered, ensure_ascii=True)
    prompt = facility_validation_prompt()
    formatted = prompt.invoke({"facilities_text": facilities_text, "facility_strings": facility_strings})
    llm = ChatOpenAI(model=settings.parking_agent_model, max_retries=0)
    structured_llm = llm.with_structured_output(FacilityValidationResponse)
    result = structured_llm.invoke(formatted.to_messages())
    return result if isinstance(result, dict) else result.model_dump()
