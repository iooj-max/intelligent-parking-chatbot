"""
LangGraph workflow for parking chatbot.

Defines the StateGraph with nodes and conditional edges for:
- Information mode: agentic tool-calling loop (assistant <-> tools)
- Reservation mode: collect booking details step-by-step

Architecture:
  Entry: llm_router
  Info flow:     llm_router -> assistant -> (tools -> assistant)* -> END
  Reservation:   llm_router -> collect_input -> validate -> check -> confirm -> END
  Transition:    assistant can switch to collect_input when start_reservation_process is called

The compiled graph is exported as 'graph' for langgraph.json compatibility.
"""

import logging

from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.chatbot.nodes import (
    assistant_node,
    llm_router,
    check_completion,
    collect_input,
    confirm_reservation,
    validate_input,
)
from src.chatbot.state import ChatbotState
from src.rag.tools import PARKING_TOOLS

# Initialize logger
logger = logging.getLogger(__name__)

# Create tool execution node
tool_node = ToolNode(PARKING_TOOLS)

# Build the workflow
workflow = StateGraph(ChatbotState)

# Add all nodes to the graph
workflow.add_node("router", llm_router)
workflow.add_node("assistant", assistant_node)
workflow.add_node("tools", tool_node)
workflow.add_node("collect_input", collect_input)
workflow.add_node("validate_input", validate_input)
workflow.add_node("check_completion", check_completion)
workflow.add_node("confirm_reservation", confirm_reservation)

# Set entry point
workflow.set_entry_point("router")


# Conditional routing functions
def route_from_router(state: ChatbotState) -> str:
    """
    Route from router based on mode.

    In info mode, always goes to assistant (which handles tool calling).
    In reservation mode, goes to collect_input.
    If the router already added an error message (guardrail rejection), go to END.

    Args:
        state: Current chatbot state

    Returns:
        Next node name: "assistant", "collect_input", or END
    """
    mode = state.get("mode", "info")

    # Check if router already added error message (guardrail rejection)
    messages = state.get("messages", [])
    if messages and len(messages) > 0:
        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage):
            # Router already handled it (guardrail rejection)
            logger.info("Router produced AI message (guardrail rejection), routing to END")
            return END

    if mode == "reservation":
        logger.info("Routing to reservation mode (collect_input)")
        return "collect_input"
    else:
        logger.info("Routing to info mode (assistant)")
        return "assistant"


def should_continue_assistant(state: ChatbotState) -> str:
    """
    Route based on whether LLM called tools or switched to reservation mode.

    Checks:
    1. If mode switched to reservation (start_reservation_process was called) -> collect_input
    2. If assistant made tool calls -> tools (execute them)
    3. Otherwise -> END (final answer ready)

    Args:
        state: Current chatbot state

    Returns:
        Next node name: "tools", "collect_input", or END
    """
    # Check if mode switched to reservation (start_reservation_process called)
    mode = state.get("mode", "info")
    if mode == "reservation":
        logger.info("Mode switched to reservation, routing to collect_input")
        return "collect_input"

    # Check if assistant made tool calls
    messages = state.get("messages", [])
    if messages and len(messages) > 0:
        last_message = messages[-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            logger.info("Assistant made tool calls, routing to tools")
            return "tools"

    # No tool calls, no mode switch -> final answer
    logger.info("No tool calls or mode switch, routing to END")
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

    # Check both field completion AND value presence
    all_complete = all(field in completed_fields for field in required_fields)
    all_values_present = all(reservation.get(field) is not None for field in required_fields)

    if all_complete and all_values_present:
        logger.info("All fields collected and validated, proceeding to confirmation")
        return "confirm_reservation"
    else:
        missing = [f for f in required_fields if f not in completed_fields or reservation.get(f) is None]
        logger.info(f"Fields incomplete: {missing}, continuing collection")
        return "collect_input"


# --- Conditional edges ---

# Router decides: assistant (info mode) | collect_input (reservation) | END (guardrail)
workflow.add_conditional_edges(
    "router",
    route_from_router,
    {
        "assistant": "assistant",
        "collect_input": "collect_input",
        END: END,
    },
)

# Agentic tool-calling loop with reservation mode detection
workflow.add_conditional_edges(
    "assistant",
    should_continue_assistant,
    {
        "tools": "tools",
        "collect_input": "collect_input",
        END: END,
    },
)

# After tool execution, loop back to assistant for next decision
workflow.add_edge("tools", "assistant")

# --- Reservation mode flow (unchanged) ---

# collect_input -> validate_input
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
