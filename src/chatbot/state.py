"""
Chatbot state definition for LangGraph workflow.

Defines the state structure for the parking chatbot including:
- Message history management
- Conversation mode tracking (info vs reservation)
- Reservation data collection
- Error handling
"""

from datetime import date, time
from typing import Annotated, Optional, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict

# Custom annotation for messages - uses LangGraph's add_messages operator
# This automatically appends new messages to the history instead of replacing
MessagesAnnotation = Annotated[Sequence[BaseMessage], add_messages]


class ReservationData(TypedDict, total=False):
    """
    Nested state for reservation data collection.

    Fields are collected step-by-step during reservation mode.
    All fields are optional (total=False) since they're populated incrementally.
    """

    # Customer information
    name: Optional[str]

    # Parking facility identifier (downtown_plaza or airport_parking)
    parking_id: Optional[str]

    # Reservation date (YYYY-MM-DD)
    date: Optional[date]

    # Arrival time (HH:MM)
    start_time: Optional[time]

    # Duration in hours (1-168)
    duration_hours: Optional[int]

    # Track which fields have been successfully collected
    completed_fields: list[str]

    # Track validation errors for retry logic
    validation_errors: dict[str, str]


class ChatbotState(TypedDict):
    """
    Main state for parking chatbot conversation.

    LangGraph will automatically manage state updates across nodes.
    Each node receives the current state and returns updates to merge.
    """

    # Conversation history
    # Uses add_messages operator to automatically append new messages
    messages: MessagesAnnotation

    # Current conversation mode: "info" or "reservation"
    mode: str

    # Query intent from retriever (STATIC, DYNAMIC, HYBRID, RESERVATION)
    intent: Optional[str]

    # Retrieved context from RAG (passed from retrieve → generate nodes)
    context: Optional[str]

    # Reservation data (only used in reservation mode)
    reservation: ReservationData

    # Error tracking for graceful degradation
    error: Optional[str]

    # Iteration count for loop prevention (safety mechanism)
    iteration_count: int

    # Intent classification hint from router (e.g., "parking_availability", "reservation")
    parking_intent: Optional[str]
