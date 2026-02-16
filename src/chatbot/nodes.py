"""
Node implementations for LangGraph chatbot workflow.

Nodes are pure functions that take ChatbotState and return state updates.
Each node is responsible for a specific part of the conversation flow:

Active nodes:
- assistant_node: Tool-calling agent for parking workflows
- llm_router: Security-focused router for incoming user messages
- collect_input: Determine next field to collect in reservation mode
- validate_input: Validate user input for reservation fields
- check_completion: Check if all reservation fields collected
- confirm_reservation: Show reservation summary
"""

import logging
from datetime import datetime
from typing import Any, Dict

from pydantic import BaseModel

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from src.chatbot.prompts import (
    CONFIRMATION_TEMPLATE,
    FIELD_PROMPTS,
    INFO_PROMPT_TEMPLATE,
    PARKING_ASSISTANT_CONSTITUTION,
    ROUTER_SYSTEM_PROMPT,
    STRICT_INFO_SYSTEM_PROMPT,
)
from src.chatbot.state import ChatbotState
from src.config import settings
from src.guardrails.input_filter import InputValidator
from src.guardrails.output_filter import OutputFilter

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize LLM dependency (singleton pattern)
_llm = None
_router_llm = None



def get_llm() -> ChatOpenAI:
    """Get or create ChatOpenAI singleton."""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=settings.openai_api_key,
            temperature=0.7,
        )
    return _llm


def get_router_llm():
    """LLM for routing decisions with structured output."""
    global _router_llm
    if _router_llm is None:
        _router_llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=settings.openai_api_key,
            temperature=0,
        )
    return _router_llm


class RouterDecision(BaseModel):
    parking_related: bool = True
    needs_data: bool = True
    intent: str = "info"


def assistant_node(state: ChatbotState) -> Dict[str, Any]:
    """
    Assistant node with tool calling + output validation.

    Replaces old retrieve + generate pattern.
    LLM decides which tools to call and when.

    Args:
        state: Current chatbot state

    Returns:
        State updates with AI message or mode switch
    """
    from langchain.agents import create_agent
    from langchain_core.messages import SystemMessage
    from src.rag.tools import PARKING_TOOLS

    try:
        # Get LLM
        llm = get_llm()

        # Get current messages
        messages = state.get("messages", [])

        # LAYER 2: System prompt with constitution
        system_message = SystemMessage(content=PARKING_ASSISTANT_CONSTITUTION)

        # Create ReAct agent graph
        agent = create_agent(
            llm,
            tools=PARKING_TOOLS,
            system_prompt=system_message
        )

        # Invoke agent with full message history
        result = agent.invoke({"messages": messages})

        # Get agent's response messages (last message in result)
        result_messages = result.get("messages", [])
        if not result_messages:
            return {"error": "No response from agent"}

        # Get the last AI message
        last_ai_message = None
        for msg in reversed(result_messages):
            if isinstance(msg, AIMessage):
                last_ai_message = msg
                break

        if not last_ai_message:
            return {"error": "No AI message in agent response"}

        output = last_ai_message.content

        # Check if agent called start_reservation_process
        if "SWITCH_TO_RESERVATION_MODE:" in output:
            parking_id = output.split("SWITCH_TO_RESERVATION_MODE:")[1].strip()
            logger.info(f"Switching to reservation mode for {parking_id}")
            return {
                "mode": "reservation",
                "reservation": {
                    "parking_id": parking_id,
                    "completed_fields": ["parking_id"],
                    "validation_errors": {}
                },
                "messages": [AIMessage(content=f"Great! I'll help you reserve parking at {parking_id}. What name should I use for the reservation?")]
            }

        # OUTPUT GUARDRAIL: PII filter only (security, not domain validation)
        output_filter = OutputFilter()
        filtered = output_filter.filter_response(output)

        if filtered["pii_found"]:
            logger.warning(f"PII masked in response: {filtered['pii_found']}")

        # Return only the new AI message (not all messages)
        return {"messages": [AIMessage(content=filtered["filtered_response"])], "mode": state.get("mode", "info")}

    except Exception as e:
        logger.error(f"Error in assistant_node: {e}", exc_info=True)
        return {
            "messages": [AIMessage(content="I'm sorry, I encountered an error. Please try again.")],
            "error": f"Assistant error: {str(e)}"
        }


def strict_answer_node(state: ChatbotState) -> Dict[str, Any]:
    """
    Produce a strict answer using ONLY tool outputs from the latest turn.
    If no information is present, return a fixed refusal.
    """
    try:
        llm = get_llm()
        messages = state.get("messages", [])

        # Find last user message index
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                last_user_idx = i
                break

        last_user_message = messages[last_user_idx].content if last_user_idx is not None else ""

        # Collect tool outputs after last user message
        tool_outputs = []
        if last_user_idx is not None:
            for msg in messages[last_user_idx + 1:]:
                if isinstance(msg, ToolMessage) and msg.content:
                    tool_outputs.append(msg.content)

        context = "\n\n".join(tool_outputs)
        strict_user = INFO_PROMPT_TEMPLATE.format(context=context, query=last_user_message)
        strict_messages = [
            SystemMessage(content=STRICT_INFO_SYSTEM_PROMPT),
            HumanMessage(content=strict_user),
        ]

        response = llm.invoke(strict_messages)
        output = response.content

        output_filter = OutputFilter()
        filtered = output_filter.filter_response(output)
        if filtered["pii_found"]:
            logger.warning(f"PII masked in response: {filtered['pii_found']}")

        return {"messages": [AIMessage(content=filtered["filtered_response"])], "mode": "info"}
    except Exception as e:
        logger.error(f"Error in strict_answer_node: {e}", exc_info=True)
        return {
            "messages": [AIMessage(content="I'm sorry, I encountered an error. Please try again.")],
            "error": f"Strict answer error: {str(e)}"
        }


def llm_router(state: ChatbotState) -> Dict[str, Any]:
    """
    Minimal router with security guardrails only.

    Domain validation happens in assistant node via constitution.
    LLM decides if query is parking-related.

    Args:
        state: Current chatbot state

    Returns:
        State updates with mode
    """
    try:
        iteration_count = state.get("iteration_count", 0) + 1

        # Get last user message
        last_user_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                last_user_message = msg.content
                break

        if not last_user_message:
            return {"mode": "info", "iteration_count": iteration_count}

        # SECURITY GUARDRAILS ONLY (not domain validation)
        validator = InputValidator()
        validation = validator.validate(last_user_message)

        if not validation["is_valid"]:
            logger.warning(f"Security violation: {validation['error_message']}")
            return {
                "mode": "info",
                "iteration_count": iteration_count,
                "messages": [AIMessage(content=validation["error_message"])]
            }

        # LLM-based routing for parking-related queries (structured output)
        router_llm = get_router_llm().with_structured_output(RouterDecision)
        router_messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=last_user_message),
        ]

        parking_related = True
        needs_data = True
        intent = "info"

        try:
            decision = router_llm.invoke(router_messages)
            parking_related = bool(decision.parking_related)
            needs_data = bool(decision.needs_data)
            intent = decision.intent or "info"
        except Exception as e:
            logger.error(f"Router LLM error: {e}", exc_info=True)
            # Default to in-scope + data lookup to avoid false refusals
            parking_related = True
            needs_data = True
            intent = "info"

        if not parking_related:
            logger.info("Input not parking-related by LLM router, routing to END with rejection")
            return {
                "mode": "info",
                "iteration_count": iteration_count,
                "messages": [AIMessage(content="I can only help with parking-related questions like availability, pricing, reservations, and operating hours. How can I help with your parking needs?")],
            }

        # Parking-related: force info flow for data requests (non-reservation)
        logger.info("Input passed security checks, routing to assistant")
        logger.info(
            "Router decision: parking_related=%s needs_data=%s intent=%s",
            parking_related, needs_data, intent
        )
        mode = "reservation" if intent == "reservation" else "info_strict"
        return {
            "mode": mode,
            "iteration_count": iteration_count,
        }

    except Exception as e:
        logger.error(f"Router error: {e}", exc_info=True)
        return {"mode": "info", "iteration_count": iteration_count}


def collect_input(state: ChatbotState) -> Dict[str, Any]:
    """
    Determine next field to collect and prompt user.

    Ordered field collection:
    1. name
    2. parking_id
    3. date
    4. start_time
    5. duration_hours

    Args:
        state: Current chatbot state

    Returns:
        State updates with AI message prompting for next field
    """
    try:
        # Ordered list of fields to collect
        required_fields = ["name", "parking_id", "date", "start_time", "duration_hours"]

        # Get completed fields
        reservation = state.get("reservation", {})
        completed = reservation.get("completed_fields", [])

        # Find next field to collect
        next_field = None
        for field in required_fields:
            if field not in completed:
                next_field = field
                break

        if not next_field:
            # All fields collected - shouldn't reach here
            logger.warning("collect_input called but all fields already collected")
            return {}

        # Get prompt for next field
        prompt = FIELD_PROMPTS.get(next_field, f"Please provide {next_field}")

        logger.info(f"Collecting field: {next_field}")

        # Return AI message with prompt
        return {"messages": [AIMessage(content=prompt)]}

    except Exception as e:
        logger.error(f"Error in collect_input node: {e}")
        return {
            "messages": [
                AIMessage(content="I'm having trouble with the reservation form. Let me try again.")
            ],
            "error": f"Collect input error: {str(e)}",
        }


def validate_input(state: ChatbotState) -> Dict[str, Any]:
    """
    Validate user input for current reservation field.

    Validation rules:
    - name: non-empty string
    - parking_id: match "downtown" or "airport" → normalize to ID
    - date: parse YYYY-MM-DD, must be today or future
    - start_time: parse HH:MM, valid time
    - duration_hours: integer 1-168

    Args:
        state: Current chatbot state

    Returns:
        State updates with validated field or error
    """
    try:
        # Get reservation data
        reservation = state.get("reservation", {})
        completed = reservation.get("completed_fields", [])
        validation_errors = {}

        # Determine next field to validate
        required_fields = ["name", "parking_id", "date", "start_time", "duration_hours"]
        next_field = None
        for field in required_fields:
            if field not in completed:
                next_field = field
                break

        if not next_field:
            # No field to validate - this shouldn't happen
            logger.error("validate_input called but no next_field found")
            return {
                "error": "Validation error: no field to validate",
                "messages": [AIMessage(content="Something went wrong with the reservation. Let's start over.")]
            }

        # Get last user message with null check
        last_user_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                last_user_message = msg.content.strip()
                break

        if not last_user_message:
            logger.warning("No user input found for validation")
            return {
                "error": "No user input to validate",
                "messages": [AIMessage(content="I didn't receive your input. Could you please try again?")]
            }

        # Validate based on field type
        validated_value = None

        if next_field == "name":
            if len(last_user_message) > 0:
                validated_value = last_user_message
            else:
                validation_errors["name"] = "Name cannot be empty"

        elif next_field == "parking_id":
            # Use semantic matching to find parking facility
            from src.services.parking_service import get_parking_service
            from src.services.parking_matcher import ParkingFacilityMatcher

            service = get_parking_service()
            matcher = ParkingFacilityMatcher(threshold=0.6)

            # Try to match user input to facilities
            facilities = service.get_all_facilities()
            matches = matcher.match_facility(last_user_message, facilities, limit=3)

            if len(matches) == 0:
                facility_names = [f.name for f in facilities]
                validation_errors["parking_id"] = f"Please specify one of: {', '.join(facility_names)}"
            elif len(matches) > 1 and matches[0]["score"] < 0.9:
                # Multiple matches without clear winner - ask for disambiguation
                facility_list = ", ".join([m['name'] for m in matches])
                validation_errors["parking_id"] = f"Multiple facilities match: {facility_list}. Please be more specific."
            else:
                # Clear match
                validated_value = matches[0]['parking_id']

        elif next_field == "date":
            # Parse date in YYYY-MM-DD format
            try:
                from datetime import timezone

                parsed_date = datetime.strptime(last_user_message, "%Y-%m-%d").date()
                # Use UTC for consistent comparison across timezones
                today_utc = datetime.now(timezone.utc).date()

                if parsed_date < today_utc:
                    validation_errors["date"] = "Date must be today or in the future"
                else:
                    validated_value = parsed_date
            except ValueError:
                validation_errors["date"] = "Invalid date format. Please use YYYY-MM-DD (e.g., 2024-03-15)"

        elif next_field == "start_time":
            # Parse time in HH:MM format
            try:
                parsed_time = datetime.strptime(last_user_message, "%H:%M").time()
                validated_value = parsed_time
            except ValueError:
                validation_errors["start_time"] = "Invalid time format. Please use HH:MM in 24-hour format (e.g., 14:30)"

        elif next_field == "duration_hours":
            # Parse integer duration
            try:
                duration = int(last_user_message)
                if duration < 1 or duration > 168:
                    validation_errors["duration_hours"] = "Duration must be between 1 and 168 hours (1 week max)"
                else:
                    validated_value = duration
            except ValueError:
                validation_errors["duration_hours"] = "Please enter a valid number of hours"

        # Update reservation state
        updated_reservation = dict(reservation)

        if validation_errors:
            # Validation failed
            updated_reservation["validation_errors"] = validation_errors
            logger.warning(f"Validation failed for {next_field}: {validation_errors}")

            # Return error message
            error_message = validation_errors.get(next_field, "Invalid input")
            return {
                "reservation": updated_reservation,
                "messages": [
                    AIMessage(content=f"{error_message}\n\n{FIELD_PROMPTS[next_field]}")
                ],
            }
        else:
            # Validation succeeded
            updated_reservation[next_field] = validated_value
            updated_reservation["completed_fields"] = completed + [next_field]
            updated_reservation["validation_errors"] = {}

            logger.info(f"Validated {next_field}: {validated_value}")

            return {"reservation": updated_reservation}

    except Exception as e:
        logger.error(f"Error in validate_input node: {e}")
        return {"error": f"Validation error: {str(e)}"}


def check_completion(state: ChatbotState) -> Dict[str, Any]:
    """
    Check if all reservation fields have been collected.

    This node doesn't return state updates, just used for routing.

    Args:
        state: Current chatbot state

    Returns:
        Empty dict (routing handled by conditional edges)
    """
    # No state updates needed - routing logic is in graph.py
    return {}


def confirm_reservation(state: ChatbotState) -> Dict[str, Any]:
    """
    Show reservation summary and ask for confirmation.

    Args:
        state: Current chatbot state with completed reservation

    Returns:
        State updates with confirmation message
    """
    try:
        reservation = state.get("reservation", {})

        # Get parking name dynamically from ParkingFacilityService
        from src.services.parking_service import get_parking_service

        parking_id = reservation.get("parking_id")
        service = get_parking_service()
        facility = service.get_facility_by_id(parking_id)
        parking_name = facility.name if facility else parking_id

        # Format confirmation message
        confirmation = CONFIRMATION_TEMPLATE.format(
            name=reservation.get("name", "Unknown"),
            parking_name=parking_name,
            date=reservation.get("date", "Unknown"),
            start_time=reservation.get("start_time", "Unknown"),
            duration_hours=reservation.get("duration_hours", "Unknown"),
        )

        logger.info("Showing reservation confirmation")

        return {"messages": [AIMessage(content=confirmation)]}

    except Exception as e:
        logger.error(f"Error in confirm_reservation node: {e}")
        return {
            "messages": [
                AIMessage(content="I'm having trouble showing the reservation summary. Please try again.")
            ],
            "error": f"Confirmation error: {str(e)}",
        }
