"""LangGraph workflow for scope validation and intent routing."""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Annotated, Any, Optional, TypedDict, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.types import interrupt

from parking_agent.message_reducer import RECENT_MESSAGES_TO_KEEP, trim_to_last_n_messages
from parking_agent.utils import message_content_to_text

from parking_agent.prompts import (
    conversation_summary_prompt,
    final_response_guardrail_prompt,
    intent_router_prompt,
    out_of_scope_response_prompt,
    reservation_admin_result_prompt,
    reservation_cancelled_response_prompt,
    reservation_confirmation_decision_prompt,
    reservation_confirmation_prompt,
    reservation_extraction_prompt,
    reservation_question_prompt,
    scope_guardrail_prompt,
)
from parking_agent.schemas import (
    FinalResponseGuardrailDecision,
    IntentDecision,
    ReservationConfirmationDecision,
    ReservationData,
    ReservationExtraction,
    RESERVATION_FIELD_CONSTRAINTS,
    RESERVATION_FIELD_DESCRIPTIONS,
    RESERVATION_FIELD_ORDER,
    ReservationField,
    ScopeDecision,
)
from parking_agent.tools import (
    run_info_react_agent,
    validate_facility_exists,
)
from src.config import settings


class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    handoff_to_reservation: Optional[bool]
    awaiting_user_confirmation: Optional[bool]
    should_create_pending_file: Optional[bool]
    admin_decision: Optional[str]
    scope_decision: Optional[str]
    scope_reasoning: Optional[str]
    intent: Optional[str]
    intent_reasoning: Optional[str]
    reservation: ReservationData
    reservation_validation: "ReservationValidationCache"
    missing_field: Optional[ReservationField]
    missing_field_reason: Optional[str]
    conversation_summary: Optional[str]
    guardrail_risk_level: Optional[str]
    guardrail_action: Optional[str]
    guardrail_reasoning: Optional[str]


class FieldValidationCache(TypedDict, total=False):
    value: str | list[str]  # Input that was validated (str for legacy, list for facility)
    is_valid: bool
    reason: str
    parking_id: str  # Resolved single parking_id when valid (for reservation)


class ReservationValidationCache(TypedDict, total=False):
    facility: FieldValidationCache


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=settings.parking_agent_model)


def _latest_user_input(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message_content_to_text(message.content)
    return message_content_to_text(messages[-1].content) if messages else ""


def _format_recent_messages(messages: list[BaseMessage], max_messages: int = 6) -> str:
    recent = messages[-max_messages:]
    lines = []
    for message in recent:
        role = getattr(message, "type", message.__class__.__name__.lower())
        text = message_content_to_text(getattr(message, "content", ""))
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


def _latest_ai_message(messages: list[BaseMessage]) -> Optional[AIMessage]:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


def _merge_reservation(
    existing: ReservationData, extracted: ReservationExtraction
) -> ReservationData:
    updated: ReservationData = cast(ReservationData, dict(existing))
    for field in RESERVATION_FIELD_ORDER:
        value = getattr(extracted, field)
        if value is None:
            continue
        if field == "facility" and isinstance(value, str) and not value.strip():
            continue  # Empty string = not provided, do not overwrite existing
        if field == "duration_hours":
            updated[field] = int(value)
        else:
            updated[field] = str(value)
    return updated


def _is_valid_date(value: str) -> bool:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return False
    return parsed >= date.today()


def _is_valid_time(value: str) -> bool:
    try:
        datetime.strptime(value, "%H:%M")
    except ValueError:
        return False
    return True


def _is_valid_duration(value: int) -> bool:
    return 1 <= value <= 168


def _is_valid_name(value: str) -> bool:
    return bool(value.strip())


def _is_valid_vehicle_plate(value: str) -> bool:
    return bool(value.strip())


def _is_valid_facility(value: list[str]) -> tuple[bool, str, list[str] | None]:
    return validate_facility_exists(value)


def _first_missing_or_invalid(
    reservation: ReservationData,
    validation_cache: Optional[ReservationValidationCache],
) -> tuple[
    Optional[ReservationField],
    Optional[str],
    ReservationValidationCache,
    ReservationData,
]:
    updated_cache: ReservationValidationCache = cast(
        ReservationValidationCache, dict(validation_cache or {})
    )
    normalized_reservation: ReservationData = cast(ReservationData, dict(reservation))

    for field in RESERVATION_FIELD_ORDER:
        if field not in reservation:
            return (
                field,
                "Value is missing and must be provided.",
                updated_cache,
                normalized_reservation,
            )
        value = normalized_reservation.get(field)
        if value is None:
            return (
                field,
                "Value is missing and must be provided.",
                updated_cache,
                normalized_reservation,
            )
        if field == "customer_name" and not _is_valid_name(str(value)):
            return (
                field,
                "Customer name is invalid. Expected any non-empty string.",
                updated_cache,
                normalized_reservation,
            )
        if field == "vehicle_plate" and not _is_valid_vehicle_plate(str(value)):
            return (
                field,
                "Vehicle plate is invalid. Expected any non-empty string.",
                updated_cache,
                normalized_reservation,
            )
        if field == "facility":
            facility_list = value if isinstance(value, list) else [str(value)] if value else []
            if not facility_list:
                return (
                    field,
                    "Value is missing and must be provided.",
                    updated_cache,
                    normalized_reservation,
                )
            if len(facility_list) > 1:
                return (
                    field,
                    "You can only book one parking at a time. Please specify one facility.",
                    updated_cache,
                    normalized_reservation,
                )
            cached_facility = updated_cache.get("facility")
            if cached_facility is not None:
                cached_value = cached_facility.get("value")
                cached_is_valid = cached_facility.get("is_valid")
                if cached_value == facility_list and isinstance(cached_is_valid, bool):
                    if not cached_is_valid:
                        cached_reason = str(cached_facility.get("reason", "")).strip()
                        return (
                            field,
                            cached_reason
                            or "Facility is invalid. It must match one of the available parking facilities.",
                            updated_cache,
                            normalized_reservation,
                        )
                    cached_parking_id = str(cached_facility.get("parking_id", "")).strip()
                    if cached_parking_id:
                        normalized_reservation["facility"] = cached_parking_id
                    continue

            is_valid_facility, facility_reason, matched_ids = _is_valid_facility(facility_list)
            updated_cache["facility"] = {
                "value": facility_list,
                "is_valid": is_valid_facility,
                "reason": facility_reason,
                "parking_id": "",
            }
            if not is_valid_facility:
                return field, facility_reason, updated_cache, normalized_reservation
            if matched_ids and len(matched_ids) == 1:
                normalized_reservation["facility"] = matched_ids[0]
                updated_cache["facility"]["parking_id"] = matched_ids[0]
            elif matched_ids and len(matched_ids) > 1:
                return (
                    field,
                    "Please specify exactly one parking facility.",
                    updated_cache,
                    normalized_reservation,
                )
        if field == "date" and not _is_valid_date(str(value)):
            return (
                field,
                "Date is invalid. Expected YYYY-MM-DD and it must be today or later.",
                updated_cache,
                normalized_reservation,
            )
        if field == "start_time" and not _is_valid_time(str(value)):
            return (
                field,
                "Start time is invalid. Expected a valid 24-hour HH:MM time.",
                updated_cache,
                normalized_reservation,
            )
        if field == "duration_hours":
            if not isinstance(value, (int, str)):
                return (
                    field,
                    "Duration is invalid. Expected an integer between 1 and 168.",
                    updated_cache,
                    normalized_reservation,
                )
            try:
                duration_value = int(value)
            except (TypeError, ValueError):
                return (
                    field,
                    "Duration is invalid. Expected an integer between 1 and 168.",
                    updated_cache,
                    normalized_reservation,
                )
            if not _is_valid_duration(duration_value):
                return (
                    field,
                    "Duration is invalid. Expected an integer between 1 and 168.",
                    updated_cache,
                    normalized_reservation,
                )
    return None, None, updated_cache, normalized_reservation


def _reservation_runtime_reset_payload() -> dict[str, Any]:
    return {
        "reservation": {},
        "reservation_validation": {},
        "missing_field": None,
        "missing_field_reason": None,
        "awaiting_user_confirmation": False,
        "should_create_pending_file": False,
        "admin_decision": None,
        "handoff_to_reservation": False,
    }


def build_graph(checkpointer=None):
    llm = _get_llm()
    GENERIC_MISSING_REASON = "Value is missing and must be provided."

    def update_conversation_summary(state: GraphState) -> dict:
        messages = state.get("messages", [])
        prompt = conversation_summary_prompt()
        existing_summary = state.get("conversation_summary") or ""
        recent_conversation = _format_recent_messages(messages, max_messages=10)
        response = llm.invoke(
            prompt.format_messages(
                existing_summary=existing_summary,
                recent_conversation=recent_conversation,
            )
        )
        trimmed = trim_to_last_n_messages(messages)
        return {
            "conversation_summary": response.content,
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *trimmed],
        }

    def scope_guardrail(state: GraphState) -> dict:
        structured_llm = llm.with_structured_output(ScopeDecision)
        prompt = scope_guardrail_prompt()
        user_input = _latest_user_input(state["messages"])
        conversation = _format_recent_messages(
            state["messages"], max_messages=RECENT_MESSAGES_TO_KEEP
        )
        summary = state.get("conversation_summary") or ""
        decision = cast(
            ScopeDecision,
            structured_llm.invoke(
                prompt.format_messages(
                    user_input=user_input,
                    summary=summary,
                    conversation=conversation,
                )
            ),
        )
        return {
            "scope_decision": decision.scope_decision,
            "scope_reasoning": decision.reasoning,
        }

    def route_scope(state: GraphState) -> str:
        if state.get("scope_decision") == "out_of_scope":
            return "out_of_scope"
        return "intent_router"

    def out_of_scope_response(state: GraphState) -> dict:
        prompt = out_of_scope_response_prompt()
        user_input = _latest_user_input(state["messages"])
        scope_reasoning = state.get("scope_reasoning") or ""
        response = llm.invoke(
            prompt.format_messages(
                user_input=user_input,
                scope_reasoning=scope_reasoning,
            )
        )
        return {"messages": [response]}

    def intent_router(state: GraphState) -> dict:
        structured_llm = llm.with_structured_output(IntentDecision)
        prompt = intent_router_prompt()
        user_input = _latest_user_input(state["messages"])
        conversation = _format_recent_messages(
            state["messages"], max_messages=RECENT_MESSAGES_TO_KEEP
        )
        summary = state.get("conversation_summary") or ""
        decision = cast(
            IntentDecision,
            structured_llm.invoke(
                prompt.format_messages(
                    user_input=user_input,
                    summary=summary,
                    conversation=conversation,
                )
            ),
        )
        return {
            "intent": decision.intent,
            "intent_reasoning": decision.reasoning,
            "handoff_to_reservation": decision.intent == "reservation",
        }

    def route_intent(state: GraphState) -> str:
        return (
            "reservation"
            if state.get("intent") == "reservation"
            else "info_retrieval"
        )

    def reservation_extract(state: GraphState) -> dict:
        structured_llm = llm.with_structured_output(ReservationExtraction)
        prompt = reservation_extraction_prompt()
        messages = state.get("messages", [])
        reservation_state = json.dumps(state.get("reservation", {}), ensure_ascii=False)
        user_input = _latest_user_input(messages)
        conversation = _format_recent_messages(
            messages, max_messages=RECENT_MESSAGES_TO_KEEP
        )
        extracted = cast(
            ReservationExtraction,
            structured_llm.invoke(
                prompt.format_messages(
                    reservation_state=reservation_state,
                    user_input=user_input or "",
                    conversation=conversation,
                )
            ),
        )
        updated = _merge_reservation(state.get("reservation", {}), extracted)
        return {"reservation": updated}

    def reservation_check(state: GraphState) -> dict:
        missing, reason, updated_validation, normalized_reservation = _first_missing_or_invalid(
            state.get("reservation", {}),
            state.get("reservation_validation"),
        )
        return {
            "missing_field": missing,
            "missing_field_reason": reason,
            "reservation_validation": updated_validation,
            "reservation": normalized_reservation,
        }

    def route_reservation(state: GraphState) -> str:
        if state.get("awaiting_user_confirmation"):
            return "decision"
        return "ask_missing" if state.get("missing_field") else "confirm"

    def reservation_ask(state: GraphState) -> dict:
        missing_field = state.get("missing_field")
        if missing_field is None:
            return {"messages": []}
        missing_field_reason = state.get("missing_field_reason") or ""
        # When the field was never asked before (generic "missing"), pass a neutral hint
        # so the LLM asks naturally instead of framing it as an error.
        if missing_field_reason == GENERIC_MISSING_REASON:
            validation_issue = "Field not provided yet (no validation error)."
        else:
            validation_issue = missing_field_reason or "No validation issue provided."
        messages = state.get("messages", [])
        user_input = _latest_user_input(messages)
        conversation = _format_recent_messages(
            messages, max_messages=RECENT_MESSAGES_TO_KEEP
        )
        prompt = reservation_question_prompt()
        response = llm.invoke(
            prompt.format_messages(
                user_input=user_input or "",
                conversation=conversation,
                field_name=missing_field,
                field_description=RESERVATION_FIELD_DESCRIPTIONS[missing_field],
                field_constraints=RESERVATION_FIELD_CONSTRAINTS[missing_field],
                validation_issue=validation_issue,
            )
        )
        return {"messages": [response]}

    def reservation_confirm(state: GraphState) -> dict:
        prompt = reservation_confirmation_prompt()
        messages = state.get("messages", [])
        conversation = _format_recent_messages(
            messages, max_messages=RECENT_MESSAGES_TO_KEEP
        )
        reservation_state = json.dumps(state.get("reservation", {}), ensure_ascii=False)
        response = llm.invoke(
            prompt.format_messages(
                conversation=conversation,
                reservation_state=reservation_state,
            )
        )
        return {
            "messages": [response],
            "awaiting_user_confirmation": True,
            "should_create_pending_file": False,
        }

    def reservation_confirmation_decision(state: GraphState) -> dict:
        structured_llm = llm.with_structured_output(ReservationConfirmationDecision)
        prompt = reservation_confirmation_decision_prompt()
        messages = state.get("messages", [])
        conversation = _format_recent_messages(
            messages, max_messages=RECENT_MESSAGES_TO_KEEP
        )
        user_input = _latest_user_input(messages)
        decision = cast(
            ReservationConfirmationDecision,
            structured_llm.invoke(
                prompt.format_messages(
                    conversation=conversation,
                    user_input=user_input,
                )
            ),
        )
        return {"admin_decision": "confirmed" if decision.confirm else "cancelled"}

    def route_confirmation_decision(state: GraphState) -> str:
        return (
            "confirm_true"
            if state.get("admin_decision") == "confirmed"
            else "confirm_false"
        )

    def reservation_cancelled_response(state: GraphState) -> dict:
        prompt = reservation_cancelled_response_prompt()
        messages = state.get("messages", [])
        conversation = _format_recent_messages(
            messages, max_messages=RECENT_MESSAGES_TO_KEEP
        )
        user_input = _latest_user_input(messages)
        response = llm.invoke(
            prompt.format_messages(
                conversation=conversation,
                user_input=user_input,
            )
        )
        reset_payload = _reservation_runtime_reset_payload()
        reset_payload["admin_decision"] = "cancelled"
        reset_payload["messages"] = [response]
        return reset_payload

    def reservation_prepare_pending(state: GraphState) -> dict:
        _ = state
        return {
            "awaiting_user_confirmation": False,
            "should_create_pending_file": True,
            "admin_decision": None,
            "handoff_to_reservation": False,
        }

    def reservation_wait_admin_decision(state: GraphState) -> dict:
        decision_raw = interrupt(
            "Awaiting administrator decision: approved or rejected."
        )
        decision = str(decision_raw or "").strip().lower()
        normalized = "approved" if decision == "approved" else "rejected"
        return {
            "admin_decision": normalized,
            "should_create_pending_file": False,
        }

    def reservation_admin_result_response(state: GraphState) -> dict:
        prompt = reservation_admin_result_prompt()
        messages = state.get("messages", [])
        conversation = _format_recent_messages(
            messages, max_messages=RECENT_MESSAGES_TO_KEEP
        )
        decision = str(state.get("admin_decision") or "rejected")
        response = llm.invoke(
            prompt.format_messages(
                decision=decision,
                conversation=conversation,
            )
        )
        return {"messages": [response]}

    def reset_reservation_runtime_state(state: GraphState) -> dict:
        _ = state
        return _reservation_runtime_reset_payload()

    def info_agent_llm(state: GraphState, config: RunnableConfig) -> dict:
        message_history = state.get("messages", [])
        user_input = _latest_user_input(message_history)
        summary = state.get("conversation_summary") or ""
        response_text = run_info_react_agent(
            user_input=user_input,
            conversation_summary=summary,
            config=config,
        )
        return {"messages": [AIMessage(content=response_text)]}

    def final_response_guardrail(state: GraphState) -> dict:
        messages = state.get("messages", [])
        latest_ai = _latest_ai_message(messages)
        if latest_ai is None:
            return {
                "guardrail_risk_level": "low",
                "guardrail_action": "allow",
                "guardrail_reasoning": "No assistant response found to guard.",
            }

        structured_llm = llm.with_structured_output(FinalResponseGuardrailDecision)
        prompt = final_response_guardrail_prompt()
        decision = cast(
            FinalResponseGuardrailDecision,
            structured_llm.invoke(
                prompt.format_messages(assistant_response=latest_ai.content)
            ),
        )

        if decision.action == "allow":
            return {
                "guardrail_risk_level": decision.risk_level,
                "guardrail_action": decision.action,
                "guardrail_reasoning": decision.reasoning,
            }

        safe_message = AIMessage(
            content=decision.safe_response_text,
            id=getattr(latest_ai, "id", None),
        )
        return {
            "messages": [safe_message],
            "guardrail_risk_level": decision.risk_level,
            "guardrail_action": decision.action,
            "guardrail_reasoning": decision.reasoning,
        }

    graph = StateGraph(GraphState)
    graph.add_node("scope_guardrail", scope_guardrail)
    graph.add_node("out_of_scope_response", out_of_scope_response)
    graph.add_node("intent_router", intent_router)
    graph.add_node("reservation_extract", reservation_extract)
    graph.add_node("reservation_check", reservation_check)
    graph.add_node("reservation_ask", reservation_ask)
    graph.add_node("reservation_confirm", reservation_confirm)
    graph.add_node("reservation_confirmation_decision", reservation_confirmation_decision)
    graph.add_node("reservation_cancelled_response", reservation_cancelled_response)
    graph.add_node("reservation_prepare_pending", reservation_prepare_pending)
    graph.add_node("reservation_wait_admin_decision", reservation_wait_admin_decision)
    graph.add_node("reservation_admin_result_response", reservation_admin_result_response)
    graph.add_node("reset_reservation_runtime_state", reset_reservation_runtime_state)
    graph.add_node("info_agent_llm", info_agent_llm)
    graph.add_node("update_conversation_summary", update_conversation_summary)
    graph.add_node("final_response_guardrail", final_response_guardrail)

    graph.add_edge(START, "scope_guardrail")
    graph.add_conditional_edges(
        "scope_guardrail",
        route_scope,
        {
            "out_of_scope": "out_of_scope_response",
            "intent_router": "intent_router"
        },
    )
    graph.add_edge("out_of_scope_response", "final_response_guardrail")

    graph.add_conditional_edges(
        "intent_router",
        route_intent,
        {
            "info_retrieval": "info_agent_llm",
            "reservation": "reservation_extract",
        },
    )
    graph.add_edge("info_agent_llm", "final_response_guardrail")

    graph.add_edge("reservation_extract", "reservation_check")
    graph.add_conditional_edges(
        "reservation_check",
        route_reservation,
        {
            "ask_missing": "reservation_ask",
            "confirm": "reservation_confirm",
            "decision": "reservation_confirmation_decision",
        },
    )
    graph.add_conditional_edges(
        "reservation_confirmation_decision",
        route_confirmation_decision,
        {
            "confirm_true": "reservation_prepare_pending",
            "confirm_false": "reservation_cancelled_response",
        },
    )
    graph.add_edge("reservation_ask", "final_response_guardrail")
    graph.add_edge("reservation_confirm", "final_response_guardrail")
    graph.add_edge("reservation_cancelled_response", "reset_reservation_runtime_state")
    graph.add_edge("reservation_prepare_pending", "reservation_wait_admin_decision")
    graph.add_edge("reservation_wait_admin_decision", "reservation_admin_result_response")
    graph.add_edge("reservation_admin_result_response", "reset_reservation_runtime_state")
    graph.add_edge("reset_reservation_runtime_state", "final_response_guardrail")
    graph.add_edge("final_response_guardrail", "update_conversation_summary")
    graph.add_edge("update_conversation_summary", END)

    compiled = graph.compile(checkpointer=checkpointer)
    return compiled.with_config(recursion_limit=10)


def make_graph(config: RunnableConfig) -> object:
    """LangGraph CLI entrypoint for LangSmith Studio.

    The LangGraph CLI can call a graph factory with a `RunnableConfig`.
    A checkpointer is required for message history in Studio; thread_id is passed
    by the client and used to persist state across invocations.
    """
    _ = config
    from langgraph.checkpoint.memory import InMemorySaver

    return build_graph(checkpointer=InMemorySaver())
