"""
LangGraph workflow for parking chatbot.

Defines the StateGraph with nodes and conditional edges for:
- Information mode: answer questions using RAG
- Reservation mode: collect booking details step-by-step

The compiled graph is exported as 'graph' for langgraph.json compatibility.
"""

import logging

from langgraph.graph import END, StateGraph

from src.chatbot.nodes import (
    check_completion,
    collect_input,
    confirm_reservation,
    generate,
    retrieve,
    router,
    validate_input,
)
from src.chatbot.state import ChatbotState

# Initialize logger
logger = logging.getLogger(__name__)

# Build the workflow
workflow = StateGraph(ChatbotState)

# Add all nodes to the graph
workflow.add_node("router", router)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("collect_input", collect_input)
workflow.add_node("validate_input", validate_input)
workflow.add_node("check_completion", check_completion)
workflow.add_node("confirm_reservation", confirm_reservation)

# Set entry point
workflow.set_entry_point("router")


# Conditional routing functions
def route_from_router(state: ChatbotState) -> str:
    """
    Route from router to either retrieve (info mode) or collect_input (reservation mode).

    Args:
        state: Current chatbot state

    Returns:
        Next node name: "retrieve" or "collect_input"
    """
    mode = state.get("mode", "info")
    if mode == "reservation":
        logger.info("Routing to reservation mode (collect_input)")
        return "collect_input"
    else:
        logger.info("Routing to info mode (retrieve)")
        return "retrieve"


def should_continue_info(state: ChatbotState) -> str:
    """
    Check if conversation should continue after generate node.

    If user's last message contains booking keywords, loop back to router
    to switch to reservation mode. Otherwise, end conversation.

    Args:
        state: Current chatbot state

    Returns:
        Next node name: "router" or END
    """
    # Check if last AI message mentions booking (this would trigger reservation mode)
    # For now, we just end the turn and let the user's next message trigger router
    # In a more advanced version, we could detect booking intent in AI's response
    return END


def route_from_validation(state: ChatbotState) -> str:
    """
    Route from validate_input based on validation result.

    If validation errors exist, loop back to collect_input to retry.
    Otherwise, proceed to check_completion.

    Args:
        state: Current chatbot state

    Returns:
        Next node name: "collect_input" or "check_completion"
    """
    reservation = state.get("reservation", {})
    validation_errors = reservation.get("validation_errors", {})

    if validation_errors:
        logger.info("Validation failed, looping back to collect_input")
        return "collect_input"
    else:
        logger.info("Validation succeeded, proceeding to check_completion")
        return "check_completion"


def route_from_completion(state: ChatbotState) -> str:
    """
    Route from check_completion based on whether all fields are collected.

    If all required fields are completed, go to confirm_reservation.
    Otherwise, loop back to collect_input for next field.

    Args:
        state: Current chatbot state

    Returns:
        Next node name: "confirm_reservation" or "collect_input"
    """
    required_fields = ["name", "parking_id", "date", "start_time", "duration_hours"]
    reservation = state.get("reservation", {})
    completed_fields = reservation.get("completed_fields", [])

    all_complete = all(field in completed_fields for field in required_fields)

    if all_complete:
        logger.info("All fields collected, proceeding to confirmation")
        return "confirm_reservation"
    else:
        logger.info(
            f"Fields collected: {len(completed_fields)}/{len(required_fields)}, continuing collection"
        )
        return "collect_input"


# Add conditional edges
workflow.add_conditional_edges(
    "router",
    route_from_router,
    {
        "retrieve": "retrieve",
        "collect_input": "collect_input",
    },
)

# Info mode flow: retrieve → generate → END
workflow.add_edge("retrieve", "generate")
workflow.add_conditional_edges("generate", should_continue_info, {END: END})

# Reservation mode flow: collect_input → validate_input → ...
workflow.add_edge("collect_input", "validate_input")

workflow.add_conditional_edges(
    "validate_input",
    route_from_validation,
    {
        "collect_input": "collect_input",
        "check_completion": "check_completion",
    },
)

workflow.add_conditional_edges(
    "check_completion",
    route_from_completion,
    {
        "confirm_reservation": "confirm_reservation",
        "collect_input": "collect_input",
    },
)

# After confirmation, end the conversation
workflow.add_edge("confirm_reservation", END)

# Compile the graph
# CRITICAL: This must be exported as 'graph' for langgraph.json compatibility
graph = workflow.compile()

logger.info("LangGraph workflow compiled successfully")
