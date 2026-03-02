"""Schemas and typed state for the parking agent."""

from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field


class ScopeDecision(BaseModel):
    """Scope classification output."""

    scope_decision: Literal["in_scope", "out_of_scope"] = Field(
        description="Whether the request is within the parking assistant scope."
    )
    reasoning: str = Field(
        description="Short reasoning for the scope decision, without suggestions."
    )


class IntentDecision(BaseModel):
    """Intent routing output for in-scope requests."""

    intent: Literal["info_retrieval", "reservation"] = Field(
        description="Whether the request is for information or reservation workflow."
    )
    reasoning: str = Field(description="Short reasoning for the intent decision.")


class ReservationConfirmationDecision(BaseModel):
    """Decision on whether the user confirms reservation submission."""

    confirm: bool = Field(
        description="True if the user confirms reservation submission, otherwise False."
    )
    reasoning: str = Field(description="Short reasoning for the confirmation decision.")


class FinalResponseGuardrailDecision(BaseModel):
    """Decision for final response safety filtering."""

    risk_level: Literal["low", "medium", "high"] = Field(
        description="Estimated sensitivity risk in the final assistant response."
    )
    action: Literal["allow", "redact", "block"] = Field(
        description="Guardrail action based on risk level."
    )
    reasoning: str = Field(
        description="Short reasoning for the risk/action decision."
    )
    safe_response_text: str = Field(
        description=(
            "Safe response text to send to user. For allow it can mirror the "
            "original response. For redact/block it must be sanitized."
        )
    )


class FacilityMatchItem(BaseModel):
    """Single facility match result: original user string and matched DB fields."""

    original: str = Field(description="User-provided facility string.")
    matched_parking_id: str = Field(
        default="",
        description="Matched parking_id from DB. Empty if not matched.",
    )
    matched_name: str = Field(
        default="",
        description="Matched facility name from DB. Empty if not matched.",
    )
    matched_address: str = Field(
        default="",
        description="Matched address from DB. Empty if not matched.",
    )
    matched_city: str = Field(
        default="",
        description="Matched city from DB. Empty if not matched.",
    )


class FacilityValidationResponse(BaseModel):
    """Structured response from facility validation (deterministic DB + LLM)."""

    status: Literal["ok", "error"]
    results: list[FacilityMatchItem] = Field(
        default_factory=list,
        description="One entry per user input string: original, matched_parking_id, matched_name.",
    )
    is_valid: bool = Field(
        default=False,
        description="True if all items have non-empty matched_parking_id.",
    )
    reason: str = Field(default="Explain why the facility validation failed.", description="Optional explanation.")


class ReservationExtraction(BaseModel):
    """Extracted reservation information from user input."""

    customer_name: Optional[str] = Field(
        default=None, description="Customer name for the reservation."
    )
    facility: Optional[str] = Field(
        default=None,
        description="Single parking facility string, or null if not mentioned.",
    )
    date: Optional[str] = Field(
        default=None, description="Reservation date in YYYY-MM-DD."
    )
    start_time: Optional[str] = Field(
        default=None, description="Start time in 24h HH:MM."
    )
    duration_hours: Optional[int] = Field(
        default=None, description="Reservation duration in hours (1-168)."
    )
    vehicle_plate: Optional[str] = Field(
        default=None, description="Vehicle license plate as provided by user."
    )


class ReservationData(TypedDict, total=False):
    customer_name: str
    facility: str  # parking_id after validation, or facility string from extraction
    date: str
    start_time: str
    duration_hours: int
    vehicle_plate: str


ReservationField = Literal[
    "customer_name",
    "facility",
    "date",
    "start_time",
    "duration_hours",
    "vehicle_plate",
]

RESERVATION_FIELD_ORDER: list[ReservationField] = [ 
    "facility",
    "customer_name",
    "vehicle_plate",
    "date",
    "start_time",
    "duration_hours",
]

RESERVATION_FIELD_DESCRIPTIONS: dict[ReservationField, str] = {
    "customer_name": "customer name",
    "facility": "parking facility",
    "vehicle_plate": "vehicle plate number",
    "date": "reservation date",
    "start_time": "reservation start time",
    "duration_hours": "reservation duration in hours",
}

RESERVATION_FIELD_CONSTRAINTS: dict[ReservationField, str] = {
    "customer_name": "any non-empty string",
    "facility": "must match an available parking facility",
    "vehicle_plate": "any non-empty string",
    "date": "YYYY-MM-DD, today or a future date",
    "start_time": "24-hour HH:MM time",
    "duration_hours": "integer between 1 and 168",
}
