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
    RESERVATION_FIELD_DISPLAY,
    RESERVATION_DISPLAY_ORDER,
    RESERVATION_FIELD_ORDER,
    ReservationField,
    ScopeDecision,
)
from parking_agent.facility_validation import get_facility_display_name
from parking_agent.tools import (
    run_info_react_agent,
    validate_facility_exists,
)
from src.config import settings

MISSING_FIELD_REASON = "Value is missing and must be provided."


class RoutingState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    scope_decision: Optional[str]
    scope_reasoning: Optional[str]
    intent: Optional[str]
    intent_reasoning: Optional[str]
    conversation_summary: Optional[str]
    guardrail_risk_level: Optional[str]
    guardrail_action: Optional[str]
    guardrail_reasoning: Optional[str]


class ExecutionState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    awaiting_user_confirmation: Optional[bool]
    user_confirmation_decision: Optional[str]  # "confirmed" | "cancelled" | "modified" from user
    intent: Optional[str]
    reservation: ReservationData
    reservation_validation: "ReservationValidationCache"
    missing_fields: Optional[list[tuple[ReservationField, str]]]
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


def _reservation_field_names_spec(reservation: ReservationData) -> str:
    field_names = [field for field in RESERVATION_FIELD_ORDER if field in reservation]
    readable_names = [
        RESERVATION_FIELD_DISPLAY[cast(ReservationField, field)].split(" ", 1)[1]
        for field in field_names
    ]
    return ", ".join(readable_names)


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


def _get_date_validation_reason(value: str) -> str | None:
    """Return reason if date is invalid, None if valid."""
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return "Date could not be understood. Please provide the date again."
    if parsed < date.today():
        return "Date is in the past and must be today or later."
    return None


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


def _is_missing_field_reason(reason: str) -> bool:
    return reason.strip() == MISSING_FIELD_REASON


def _all_missing_or_invalid(
    reservation: ReservationData,
    validation_cache: Optional[ReservationValidationCache],
) -> tuple[
    list[tuple[ReservationField, str]],
    ReservationValidationCache,
    ReservationData,
]:
    """Return all missing or invalid fields with reasons."""
    updated_cache: ReservationValidationCache = cast(
        ReservationValidationCache, dict(validation_cache or {})
    )
    normalized_reservation: ReservationData = cast(ReservationData, dict(reservation))
    issues: list[tuple[ReservationField, str]] = []

    for field in RESERVATION_FIELD_ORDER:
        if field not in reservation:
            issues.append((field, MISSING_FIELD_REASON))
            continue
        value = normalized_reservation.get(field)
        if value is None:
            issues.append((field, MISSING_FIELD_REASON))
            continue
        if field == "customer_name" and not _is_valid_name(str(value)):
            issues.append((
                field,
                "Customer name is invalid. Expected any non-empty string.",
            ))
            continue
        if field == "vehicle_plate" and not _is_valid_vehicle_plate(str(value)):
            issues.append((
                field,
                "Vehicle plate is invalid. Expected any non-empty string.",
            ))
            continue
        if field == "facility":
            facility_list = value if isinstance(value, list) else [str(value)] if value else []
            if not facility_list:
                issues.append((field, MISSING_FIELD_REASON))
                continue
            if len(facility_list) > 1:
                issues.append((
                    field,
                    "You can only book one parking at a time. Please specify one facility.",
                ))
                continue
            cached_facility = updated_cache.get("facility")
            if cached_facility is not None:
                cached_value = cached_facility.get("value")
                cached_is_valid = cached_facility.get("is_valid")
                if cached_value == facility_list and isinstance(cached_is_valid, bool):
                    if not cached_is_valid:
                        cached_reason = str(cached_facility.get("reason", "")).strip()
                        issues.append((
                            field,
                            cached_reason
                            or "Facility is invalid. It must match one of the available parking facilities.",
                        ))
                        continue
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
                issues.append((field, facility_reason))
                continue
            if matched_ids and len(matched_ids) == 1:
                normalized_reservation["facility"] = matched_ids[0]
                updated_cache["facility"]["parking_id"] = matched_ids[0]
            elif matched_ids and len(matched_ids) > 1:
                issues.append((field, "Please specify exactly one parking facility."))
                continue
        if field == "date":
            date_reason = _get_date_validation_reason(str(value))
            if date_reason is not None:
                issues.append((field, date_reason))
                continue
        if field == "start_time" and not _is_valid_time(str(value)):
            issues.append((
                field,
                "Start time could not be understood. Please provide the time again.",
            ))
            continue
        if field == "duration_hours":
            if not isinstance(value, (int, str)):
                issues.append((
                    field,
                    "Duration could not be understood or is out of range. Please provide a number of hours between 1 and 168.",
                ))
                continue
            try:
                duration_value = int(value)
            except (TypeError, ValueError):
                issues.append((
                    field,
                    "Duration could not be understood or is out of range. Please provide a number of hours between 1 and 168.",
                ))
                continue
            if not _is_valid_duration(duration_value):
                issues.append((
                    field,
                    "Duration must be between 1 and 168 hours.",
                ))
                continue
    return issues, updated_cache, normalized_reservation


def build_routing_graph(checkpointer=None):
    llm = _get_llm()

    def scope_guardrail(state: RoutingState) -> dict:
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

    def route_scope(state: RoutingState) -> str:
        if state.get("scope_decision") == "out_of_scope":
            return "out_of_scope"
        return "intent_router"

    def out_of_scope_response(state: RoutingState) -> dict:
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

    def intent_router(state: RoutingState) -> dict:
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
        }

    def final_response_guardrail(state: RoutingState) -> dict:
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

    graph = StateGraph(RoutingState)
    graph.add_node("scope_guardrail", scope_guardrail)
    graph.add_node("out_of_scope_response", out_of_scope_response)
    graph.add_node("intent_router", intent_router)
    graph.add_node("final_response_guardrail", final_response_guardrail)

    graph.add_edge(START, "scope_guardrail")
    graph.add_conditional_edges(
        "scope_guardrail",
        route_scope,
        {
            "out_of_scope": "out_of_scope_response",
            "intent_router": "intent_router",
        },
    )
    graph.add_edge("out_of_scope_response", "final_response_guardrail")
    graph.add_edge("final_response_guardrail", END)
    graph.add_edge("intent_router", END)

    compiled = graph.compile(checkpointer=checkpointer)
    return compiled.with_config(recursion_limit=6)


def build_execution_graph(checkpointer=None):
    llm = _get_llm()

    def update_conversation_summary(state: ExecutionState) -> dict:
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

    def route_execution_entry(state: ExecutionState) -> str:
        intent = state.get("intent")
        if intent == "reservation":
            return "reservation"
        return "info_retrieval"

    def reservation_extract(state: ExecutionState) -> dict:
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
        return {
            "reservation": updated,
            "user_confirmation_decision": None,
        }

    def reservation_check(state: ExecutionState) -> dict:
        issues, updated_validation, normalized_reservation = _all_missing_or_invalid(
            state.get("reservation", {}),
            state.get("reservation_validation"),
        )
        return {
            "missing_fields": issues if issues else None,
            "reservation_validation": updated_validation,
            "reservation": normalized_reservation,
        }

    def route_reservation(state: ExecutionState) -> str:
        if state.get("awaiting_user_confirmation"):
            return "decision"
        return "ask_missing" if state.get("missing_fields") else "confirm"

    def reservation_ask(state: ExecutionState) -> dict:
        missing_fields = state.get("missing_fields")
        if not missing_fields:
            return {"messages": []}
        messages = state.get("messages", [])
        user_input = _latest_user_input(messages)
        conversation = _format_recent_messages(
            messages, max_messages=RECENT_MESSAGES_TO_KEEP
        )
        missing_field_names = {f for f, _ in missing_fields}
        display_lines = []
        for field in RESERVATION_DISPLAY_ORDER:
            if field in missing_field_names:
                display_lines.append(RESERVATION_FIELD_DISPLAY[field])
        if "duration_hours" in missing_field_names:
            display_lines.append(RESERVATION_FIELD_DISPLAY["duration_hours"])
        fields_spec = "\n".join(display_lines)
        validation_error_lines: list[str] = []
        for field, reason in missing_fields:
            if _is_missing_field_reason(reason):
                continue
            field_display = RESERVATION_FIELD_DISPLAY.get(field, field)
            validation_error_lines.append(f"- {field_display}: {reason}")
        validation_errors_spec = "\n".join(validation_error_lines)
        prompt = reservation_question_prompt()
        response = llm.invoke(
            prompt.format_messages(
                user_input=user_input or "",
                conversation=conversation,
                missing_fields_spec=fields_spec,
                validation_errors_spec=validation_errors_spec,
            )
        )
        return {"messages": [response]}

    def reservation_confirm(state: ExecutionState) -> dict:
        prompt = reservation_confirmation_prompt()
        messages = state.get("messages", [])
        conversation = _format_recent_messages(
            messages, max_messages=RECENT_MESSAGES_TO_KEEP
        )
        reservation = cast(ReservationData, state.get("reservation", {}))
        reservation_for_display = cast(ReservationData, dict(reservation))
        facility_id = reservation.get("facility")
        if facility_id:
            display_name = get_facility_display_name(str(facility_id))
            if display_name:
                reservation_for_display["facility"] = display_name
        reservation_state = json.dumps(reservation_for_display, ensure_ascii=False)
        reservation_field_names_spec = _reservation_field_names_spec(reservation)
        response = llm.invoke(
            prompt.format_messages(
                conversation=conversation,
                reservation_state=reservation_state,
                reservation_field_names_spec=reservation_field_names_spec,
            )
        )
        return {
            "messages": [response],
            "awaiting_user_confirmation": True,
            "user_confirmation_decision": None,
        }

    def reservation_confirmation_decision(state: ExecutionState) -> dict:
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
        decision_state = "modified"
        if decision.confirm is True:
            decision_state = "confirmed"
        elif decision.confirm is False:
            decision_state = "cancelled"
        return {
            "user_confirmation_decision": decision_state,
            "awaiting_user_confirmation": None,
        }

    def route_confirmation_decision(state: ExecutionState) -> str:
        user_decision = state.get("user_confirmation_decision")
        if user_decision == "confirmed":
            return "confirm_true"
        if user_decision == "cancelled":
            return "confirm_false"
        return "confirm_modify"

    def reservation_cancelled_response(state: ExecutionState) -> dict:
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
        return {
            "messages": [response],
            "user_confirmation_decision": None,
        }

    def reservation_wait_admin_decision(state: ExecutionState) -> dict:
        decision_raw = interrupt(
            "Awaiting administrator decision: approved or rejected."
        )
        decision = str(decision_raw or "").strip().lower()
        normalized = "approved" if decision == "approved" else "rejected"
        prompt = reservation_admin_result_prompt()
        messages = state.get("messages", [])
        conversation = _format_recent_messages(
            messages, max_messages=RECENT_MESSAGES_TO_KEEP
        )
        response = llm.invoke(
            prompt.format_messages(
                decision=normalized,
                conversation=conversation,
            )
        )
        return {
            "messages": [response],
            "user_confirmation_decision": None,
            "reservation": {},
            "reservation_validation": {},
            "missing_fields": None,
        }

    def info_agent_llm(state: ExecutionState, config: RunnableConfig) -> dict:
        message_history = state.get("messages", [])
        user_input = _latest_user_input(message_history)
        summary = state.get("conversation_summary") or ""
        response_text = run_info_react_agent(
            user_input=user_input,
            conversation_summary=summary,
            config=config,
        )
        return {
            "messages": [AIMessage(content=response_text)],
            "awaiting_user_confirmation": None,
        }

    def final_response_guardrail(state: ExecutionState) -> dict:
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

    graph = StateGraph(ExecutionState)
    graph.add_node("reservation_extract", reservation_extract)
    graph.add_node("reservation_check", reservation_check)
    graph.add_node("reservation_ask", reservation_ask)
    graph.add_node("reservation_confirm", reservation_confirm)
    graph.add_node("reservation_confirmation_decision", reservation_confirmation_decision)
    graph.add_node("reservation_cancelled_response", reservation_cancelled_response)
    graph.add_node("reservation_wait_admin_decision", reservation_wait_admin_decision)
    graph.add_node("info_agent_llm", info_agent_llm)
    graph.add_node("update_conversation_summary", update_conversation_summary)
    graph.add_node("final_response_guardrail", final_response_guardrail)

    graph.add_conditional_edges(
        START,
        route_execution_entry,
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
            "confirm_true": "reservation_wait_admin_decision",
            "confirm_false": "reservation_cancelled_response",
            "confirm_modify": "reservation_extract",
        },
    )
    graph.add_edge("reservation_ask", "final_response_guardrail")
    graph.add_edge("reservation_confirm", "final_response_guardrail")
    graph.add_edge("reservation_cancelled_response", "final_response_guardrail")
    graph.add_edge("reservation_wait_admin_decision", "final_response_guardrail")
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

    return build_execution_graph(checkpointer=InMemorySaver())


def make_routing_graph(config: RunnableConfig) -> object:
    """LangGraph CLI entrypoint for the routing graph."""
    _ = config
    from langgraph.checkpoint.memory import InMemorySaver

    return build_routing_graph(checkpointer=InMemorySaver())
