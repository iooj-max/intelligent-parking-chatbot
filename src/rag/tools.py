"""
LangChain tools for parking chatbot agent.

This module defines the tool functions that the LLM agent can invoke to
retrieve parking information, check availability, calculate costs, get
operating hours, and initiate the reservation workflow.

Each tool returns a plain string suitable for LLM consumption and handles
errors gracefully (returning error descriptions rather than raising).
"""

import logging
import math
from typing import Optional

from langchain_core.tools import tool

from src.data.embeddings import EmbeddingGenerator
from src.rag.retriever import ParkingRetriever
from src.rag.sql_store import SQLStore
from src.rag.vector_store import WeaviateStore
from src.services.parking_service import ParkingFacilityService, get_parking_service

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton getters  (mirrors the pattern used in nodes.py)
# ---------------------------------------------------------------------------

_retriever: Optional[ParkingRetriever] = None
_sql_store: Optional[SQLStore] = None


def get_retriever() -> ParkingRetriever:
    """Get or create a ParkingRetriever singleton."""
    global _retriever
    if _retriever is None:
        vector_store = WeaviateStore()
        sql_store = SQLStore()
        embedding_generator = EmbeddingGenerator()
        _retriever = ParkingRetriever(vector_store, sql_store, embedding_generator)
    return _retriever


def get_sql_store() -> SQLStore:
    """Get or create a SQLStore singleton."""
    global _sql_store
    if _sql_store is None:
        _sql_store = SQLStore()
    return _sql_store


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


@tool
def search_parking_info(query: str, parking_id: Optional[str] = None) -> str:
    """Search for parking facility information.

    Use this tool for questions about features, location, policies, amenities,
    booking procedures, payment methods, accessibility, security, contact
    details, EV charging, and other general information about a parking
    facility.

    Args:
        query: The user's natural-language question about parking.
        parking_id: Optional parking facility identifier to narrow the search
            (e.g. "downtown_plaza" or "airport_parking"). When omitted the
            search covers all facilities.

    Returns:
        A formatted string with the most relevant parking information.
    """
    logger.info(
        "Tool search_parking_info called | query=%s | parking_id=%s",
        query,
        parking_id,
    )
    try:
        retriever = get_retriever()
        result = retriever.retrieve(
            query=query,
            parking_id=parking_id,
            return_format="string",
        )
        return result
    except Exception as exc:
        logger.error("search_parking_info failed: %s", exc, exc_info=True)
        return f"Error searching parking information: {exc}"


@tool
def check_availability(parking_id: str) -> str:
    """Get real-time parking space availability for a specific facility.

    Use this tool when the user asks how many spaces are free, whether
    a parking lot is full, or wants current occupancy data.

    Args:
        parking_id: The unique identifier of the parking facility
            (e.g. "downtown_plaza" or "airport_parking").

    Returns:
        A formatted string showing total spaces, available spaces, and
        the timestamp of the last update.
    """
    logger.info("Tool check_availability called | parking_id=%s", parking_id)
    try:
        sql_store = get_sql_store()
        availability = sql_store.get_availability(parking_id)

        if availability is None:
            return f"No availability data found for parking facility '{parking_id}'."

        updated_str = (
            availability.last_updated.strftime("%Y-%m-%d %H:%M:%S")
            if availability.last_updated
            else "unknown"
        )

        return (
            f"Current availability for '{parking_id}':\n"
            f"- Total spaces: {availability.total_spaces}\n"
            f"- Available spaces: {availability.available_spaces}\n"
            f"- Occupied spaces: {availability.occupied_spaces}\n"
            f"- Last updated: {updated_str}"
        )
    except Exception as exc:
        logger.error("check_availability failed: %s", exc, exc_info=True)
        return f"Error checking availability: {exc}"


@tool
def calculate_parking_cost(parking_id: str, duration_hours: int) -> str:
    """Calculate the estimated parking cost for a given duration.

    Use this tool when the user wants to know how much it will cost to
    park for a certain number of hours at a specific facility.

    Args:
        parking_id: The unique identifier of the parking facility
            (e.g. "downtown_plaza" or "airport_parking").
        duration_hours: The desired parking duration in whole hours.

    Returns:
        A formatted string showing the duration, applicable rate, and
        the calculated total cost.
    """
    logger.info(
        "Tool calculate_parking_cost called | parking_id=%s | duration_hours=%d",
        parking_id,
        duration_hours,
    )
    try:
        if duration_hours <= 0:
            return "Duration must be a positive number of hours."

        sql_store = get_sql_store()
        pricing_rules = sql_store.get_pricing_rules(parking_id, active_only=True)

        if not pricing_rules:
            return f"No active pricing rules found for parking facility '{parking_id}'."

        # Use the first (highest-priority) applicable rule
        rule = pricing_rules[0]

        # Convert the requested duration into the rule's time unit
        time_unit = rule.time_unit  # hour, day, week, month
        if time_unit == "hour":
            units = duration_hours
        elif time_unit == "day":
            units = math.ceil(duration_hours / 24)
        elif time_unit == "week":
            units = math.ceil(duration_hours / (24 * 7))
        elif time_unit == "month":
            units = math.ceil(duration_hours / (24 * 30))
        else:
            units = duration_hours  # fallback

        total_cost = float(rule.price_per_unit) * units

        return (
            f"Parking cost estimate for '{parking_id}':\n"
            f"- Duration: {duration_hours} hour(s)\n"
            f"- Rate: ${float(rule.price_per_unit):.2f}/{rule.time_unit} "
            f"({rule.rule_name})\n"
            f"- Units charged: {units} {rule.time_unit}(s)\n"
            f"- Estimated total: ${total_cost:.2f}"
        )
    except Exception as exc:
        logger.error("calculate_parking_cost failed: %s", exc, exc_info=True)
        return f"Error calculating parking cost: {exc}"


@tool
def get_facility_hours(parking_id: str) -> str:
    """Get the regular operating hours for a parking facility.

    Use this tool when the user asks what time a parking lot opens or
    closes, or wants to see the weekly schedule.

    Args:
        parking_id: The unique identifier of the parking facility
            (e.g. "downtown_plaza" or "airport_parking").

    Returns:
        A formatted weekly schedule showing opening and closing times
        for each day.
    """
    logger.info("Tool get_facility_hours called | parking_id=%s", parking_id)
    try:
        sql_store = get_sql_store()
        working_hours = sql_store.get_working_hours(parking_id)

        if not working_hours:
            return f"No operating hours found for parking facility '{parking_id}'."

        day_names = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]

        lines = [f"Operating hours for '{parking_id}':"]
        for entry in working_hours:
            day_name = day_names[entry.day_of_week]
            if entry.is_closed:
                lines.append(f"- {day_name}: Closed")
            else:
                open_str = entry.open_time.strftime("%H:%M")
                close_str = entry.close_time.strftime("%H:%M")
                lines.append(f"- {day_name}: {open_str} - {close_str}")

        return "\n".join(lines)
    except Exception as exc:
        logger.error("get_facility_hours failed: %s", exc, exc_info=True)
        return f"Error retrieving facility hours: {exc}"


@tool
def start_reservation_process(parking_id: str) -> str:
    """Initiate the parking reservation workflow for a specific facility.

    CRITICAL: Call this tool when the user explicitly wants to BOOK or
    RESERVE a parking spot. Do NOT call this tool for general inquiries;
    it triggers a mode switch into the reservation collection flow.

    Args:
        parking_id: The unique identifier of the parking facility where
            the user wants to reserve a spot.

    Returns:
        A special marker string that the system uses to switch into
        reservation mode.
    """
    logger.info(
        "Tool start_reservation_process called | parking_id=%s", parking_id
    )
    try:
        return f"SWITCH_TO_RESERVATION_MODE:{parking_id}"
    except Exception as exc:
        logger.error("start_reservation_process failed: %s", exc, exc_info=True)
        return f"Error starting reservation: {exc}"


# ---------------------------------------------------------------------------
# Exported tool list for binding to the LLM agent
# ---------------------------------------------------------------------------

PARKING_TOOLS = [
    search_parking_info,
    check_availability,
    calculate_parking_cost,
    get_facility_hours,
    start_reservation_process,
]
