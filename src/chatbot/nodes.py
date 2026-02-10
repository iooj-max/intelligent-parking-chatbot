"""
Node implementations for LangGraph chatbot workflow.

Nodes are pure functions that take ChatbotState and return state updates.
Each node is responsible for a specific part of the conversation flow:
- router: Classify user intent (info vs reservation)
- retrieve: Get RAG context from ParkingRetriever
- generate: Generate LLM response using context
- collect_input: Determine next field to collect in reservation mode
- validate_input: Validate user input for reservation fields
- check_completion: Check if all reservation fields collected
- confirm_reservation: Show reservation summary
"""

import logging
from datetime import date, datetime, time
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from src.chatbot.prompts import (
    CANCELLATION_KEYWORDS,
    CONFIRMATION_TEMPLATE,
    FIELD_PROMPTS,
    INFO_PROMPT_TEMPLATE,
    INFO_SYSTEM_PROMPT,
    PARKING_NAMES,
    RESERVATION_SYSTEM_PROMPT,
)
from src.chatbot.state import ChatbotState
from src.config import settings
from src.guardrails.input_filter import InputValidator
from src.guardrails.output_filter import OutputFilter
from src.rag.retriever import ParkingRetriever
from src.rag.sql_store import SQLStore
from src.rag.vector_store import WeaviateStore
from src.data.embeddings import EmbeddingGenerator

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize RAG dependencies (singleton pattern)
_retriever = None
_llm = None


def get_parking_retriever() -> ParkingRetriever:
    """Get or create ParkingRetriever singleton."""
    global _retriever
    if _retriever is None:
        vector_store = WeaviateStore()
        sql_store = SQLStore()
        embedding_generator = EmbeddingGenerator()
        _retriever = ParkingRetriever(vector_store, sql_store, embedding_generator)
    return _retriever


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


# Booking keywords for intent detection
BOOKING_KEYWORDS = [
    "book",
    "reserve",
    "reservation",
    "make a booking",
    "schedule",
    "i want to book",
    "i'd like to book",
]


def router(state: ChatbotState) -> Dict[str, Any]:
    """
    Route user to info or reservation mode based on intent.

    Logic:
    - If last message contains booking keywords → "reservation"
    - If already in reservation mode and not cancelled → stay in "reservation"
    - Otherwise → "info"

    Args:
        state: Current chatbot state

    Returns:
        State updates with mode set
    """
    try:
        # Increment iteration count for loop prevention
        iteration_count = state.get("iteration_count", 0) + 1

        # Check for cancellation keywords
        if state["messages"]:
            last_user_message = None
            # Find last human message
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    last_user_message = msg.content.lower()
                    break

            if last_user_message:
                # Check for cancellation
                if any(keyword in last_user_message for keyword in CANCELLATION_KEYWORDS):
                    logger.info("User requested cancellation, switching to info mode")
                    return {
                        "mode": "info",
                        "iteration_count": iteration_count,
                        "reservation": {
                            "completed_fields": [],
                            "validation_errors": {},
                        },
                    }

                # Check for booking intent
                if any(keyword in last_user_message for keyword in BOOKING_KEYWORDS):
                    logger.info("Detected booking intent, switching to reservation mode")
                    return {"mode": "reservation", "iteration_count": iteration_count}

        # Stay in current mode if already in reservation mode
        if state.get("mode") == "reservation":
            return {"mode": "reservation", "iteration_count": iteration_count}

        # Default to info mode
        return {"mode": "info", "iteration_count": iteration_count}

    except Exception as e:
        logger.error(f"Error in router node: {e}")
        return {"error": f"Router error: {str(e)}", "iteration_count": iteration_count}


def retrieve(state: ChatbotState) -> Dict[str, Any]:
    """
    Retrieve context from ParkingRetriever for user query.

    Args:
        state: Current chatbot state

    Returns:
        State updates with context and intent from retriever
    """
    try:
        # Get last user message
        last_user_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                last_user_message = msg.content
                break

        if not last_user_message:
            logger.warning("No user message found in retrieve node")
            return {"error": "No user message to retrieve context for"}

        # INPUT GUARDRAIL: Validate user input
        validator = InputValidator()
        validation = validator.validate(last_user_message)

        if not validation["is_valid"]:
            logger.warning(f"Input rejected: {validation['error_message']}")
            return {
                "messages": [AIMessage(content=validation["error_message"])],
                "error": validation["error_message"],
            }

        # Call retriever
        retriever = get_parking_retriever()
        result = retriever.retrieve(
            query=last_user_message, return_format="structured"
        )

        logger.info(f"Retrieved context for query: {last_user_message[:50]}...")

        return {
            "context": result.context_string,
            "intent": result.intent.value if result.intent else None,
        }

    except Exception as e:
        logger.error(f"Error in retrieve node: {e}")
        # Graceful degradation - continue without context
        return {
            "context": "I'm having trouble accessing the parking database right now. Please try again later.",
            "error": f"Retrieval error: {str(e)}",
        }


def generate(state: ChatbotState) -> Dict[str, Any]:
    """
    Generate LLM response using retrieved context.

    Args:
        state: Current chatbot state with context

    Returns:
        State updates with AI message added
    """
    try:
        # Get last user message
        last_user_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                last_user_message = msg.content
                break

        if not last_user_message:
            return {"error": "No user message found"}

        # Get context from state
        context = state.get("context", "No context available")

        # Format prompt with context
        user_prompt = INFO_PROMPT_TEMPLATE.format(
            context=context, query=last_user_message
        )

        # Get LLM
        llm = get_llm()

        # Generate response
        messages = [
            {"role": "system", "content": INFO_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        response = llm.invoke(messages)

        # OUTPUT GUARDRAIL: Filter response for PII
        output_filter = OutputFilter()
        filtered = output_filter.filter_response(response.content)

        final_content = filtered["filtered_response"]

        logger.info("Generated LLM response")
        if filtered["pii_found"]:
            logger.warning(f"PII masked in response: {filtered['pii_found']}")

        # Return AI message to be added to state
        return {"messages": [AIMessage(content=final_content)]}

    except Exception as e:
        logger.error(f"Error in generate node: {e}")
        return {
            "messages": [
                AIMessage(
                    content="I'm sorry, I'm having trouble generating a response right now. Please try again."
                )
            ],
            "error": f"Generation error: {str(e)}",
        }


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
            # No field to validate
            return {}

        # Get last user message
        last_user_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                last_user_message = msg.content.strip()
                break

        if not last_user_message:
            return {"error": "No user input to validate"}

        # Validate based on field type
        validated_value = None

        if next_field == "name":
            if len(last_user_message) > 0:
                validated_value = last_user_message
            else:
                validation_errors["name"] = "Name cannot be empty"

        elif next_field == "parking_id":
            # Normalize parking facility name to ID
            user_input_lower = last_user_message.lower()
            if "downtown" in user_input_lower:
                validated_value = "downtown_plaza"
            elif "airport" in user_input_lower:
                validated_value = "airport_parking"
            else:
                validation_errors["parking_id"] = "Please specify either Downtown Plaza or Airport Parking"

        elif next_field == "date":
            # Parse date in YYYY-MM-DD format
            try:
                parsed_date = datetime.strptime(last_user_message, "%Y-%m-%d").date()
                # Check if date is today or future
                if parsed_date < date.today():
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

        # Get parking name
        parking_id = reservation.get("parking_id")
        parking_name = PARKING_NAMES.get(parking_id, parking_id)

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
