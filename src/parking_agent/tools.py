"""Tooling for parking information retrieval, facility validation, and agent orchestration."""

from __future__ import annotations

import logging

from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore

from parking_agent.agent_runners import run_info_react_agent
from parking_agent.facility_validation import validate_facility
from parking_agent.clients import build_weaviate_client
from parking_agent.retrieval import (
    build_weaviate_retriever,
    deduplicate_documents,
    normalize_document,
)
from pydantic import SecretStr
from src.config import settings


def _safe_tool_error_message() -> str:
    return "The requested data is currently unavailable."


def _derive_matched_from_results(results: list[dict]) -> list[str]:
    """Extract matched parking_ids from FacilityValidationResponse.results."""
    matched = []
    for r in results or []:
        if isinstance(r, dict):
            pid = r.get("matched_parking_id") or ""
            if str(pid).strip():
                matched.append(str(pid).strip())
    return matched


def _derive_unresolved_from_results(results: list[dict]) -> list[str]:
    """Extract unresolved originals from FacilityValidationResponse.results."""
    return [
        str(r.get("original", "")).strip()
        for r in results or []
        if isinstance(r, dict) and not str(r.get("matched_parking_id", "")).strip()
    ]


def validate_facility_exists(facility: list[str]) -> tuple[bool, str, list[str] | None]:
    """Validate facility strings against dynamic PostgreSQL parking_facilities data."""
    if not facility:
        return (
            False,
            "Facility is required and cannot be empty.",
            None,
        )

    validation_result = validate_facility(facility)
    if validation_result.get("status") != "ok":
        return (
            False,
            "Facility could not be validated right now. Please provide the parking facility again.",
            None,
        )

    results = validation_result.get("results") or []
    if not isinstance(results, list):
        results = []
    matched = _derive_matched_from_results(results)
    unresolved = _derive_unresolved_from_results(results)
    is_valid = bool(validation_result.get("is_valid"))

    if len(unresolved) > 0:
        return (
            False,
            "Facility is invalid. It must match one of the available parking facilities."
            + (f" Could not recognize: {', '.join(unresolved)}" if unresolved else ""),
            None,
        )
    if len(matched) == 0 or not is_valid:
        return (
            False,
            "Facility is invalid. It must match one of the available parking facilities.",
            None,
        )
    return True, "", matched


@tool("resolve_facility")
def resolve_facility(facility_token: str) -> dict:
    """Resolve a user-provided facility reference to a canonical parking_id.

    Input contract (strict):
    - facility_token MUST contain facility identifying details and MUST be composed only of:
      - city
      - facility name
      - address
    
    Call ONLY when the user has provided at least one of:
      - a facility name
      - an address
      - a city 
    """
    normalized = (facility_token or "").strip()
    if not normalized:
        return {"status": "error", "parking_id": [], "reason": "Facility token is required and cannot be empty."}
    is_valid, reason, parking_ids = validate_facility_exists([normalized])
    if is_valid and parking_ids:
        return {"status": "ok", "parking_id": parking_ids, "reason": ""}
    return {"status": "error", "parking_id": [], "reason": reason or "Facility is invalid."}


@tool("ask_clarifying_question")
def ask_clarifying_question(question: str) -> str:
    """Ask the user exactly one targeted clarifying question and stop.
    Call this tool ONLY when:
    - User provides absolutely no details about which facility they mean (e.g. 'tell me about parking')
    - A specific parameter is missing to complete the answer
    Respond in the same language as the user's message."""
    return question.strip()


@tool("retrieve_static_parking_info")
def retrieve_static_parking_info(query: str) -> dict:
    """Read static parking content from Weaviate.

    Static data has the following content files:
    - booking_process.md (How to Book Parking)
    - faq.md (Frequently Asked Questions)
    - features.md (Parking Features & Amenities)
    - general_info.md (Overview, Contact Information, etc)
    - location.md (Location & Access)
    - policies.md (Parking Policies & Rules)

    Args:
        query: Natural-language retrieval query.
    """
    collection = settings.weaviate_collection
    top_k = settings.weaviate_top_k
    top_k = min(max(top_k, 1), 20)
    candidate_k = settings.weaviate_candidate_k
    if candidate_k <= 0:
        candidate_k = max(20, top_k * 4)
    candidate_k = min(max(candidate_k, top_k), 100)
    max_chunks_per_source = settings.weaviate_max_chunks_per_source
    alpha = settings.weaviate_query_alpha

    client = None
    try:
        client = build_weaviate_client()
        vector_store = WeaviateVectorStore(
            client=client,
            index_name=collection,
            text_key="content",
            embedding=OpenAIEmbeddings(api_key=SecretStr(settings.openai_api_key)),
            attributes=[
                "parking_id",
                "content_type",
                "source_file",
                "chunk_index",
                "metadata",
            ],
        )
        retriever = build_weaviate_retriever(
            vector_store,
            k=candidate_k,
            alpha=alpha,
            parking_ids=None,
        )
        candidate_documents = retriever.invoke(query)
        documents = deduplicate_documents(
            candidate_documents,
            top_k=top_k,
            max_chunks_per_source_file=max_chunks_per_source,
        )
        results = [normalize_document(doc) for doc in documents]
        return {
            "status": "ok",
            "count": len(results),
            "results": results,
        }
    except Exception:
        logging.getLogger(__name__).exception("retrieve_static_parking_info failed")
        return {
            "status": "error",
            "count": 0,
            "results": [],
            "safe_error_message": _safe_tool_error_message(),
        }
    finally:
        if client is not None:
            client.close()


# Re-export for backward compatibility (graph imports from tools)
__all__ = [
    "ask_clarifying_question",
    "resolve_facility",
    "retrieve_static_parking_info",
    "run_info_react_agent",
    "validate_facility_exists",
]
