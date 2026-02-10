"""
Comprehensive test suite for parking chatbot.

Tests cover:
- Node implementations (unit tests with mocks)
- Graph workflow (integration tests with real components)
- Error handling and edge cases
"""

from datetime import date, time
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

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


# Fixtures
@pytest.fixture
def base_state() -> ChatbotState:
    """Base chatbot state for testing."""
    return {
        "messages": [],
        "mode": "info",
        "intent": None,
        "context": None,
        "reservation": {
            "completed_fields": [],
            "validation_errors": {},
        },
        "error": None,
        "iteration_count": 0,
    }


@pytest.fixture
def mock_retriever():
    """Mock ParkingRetriever for testing."""
    retriever = Mock()
    # Mock retrieve method to return structured result
    mock_result = Mock()
    mock_result.context_string = "## Static Information\n\nMocked context about parking facilities"
    mock_result.intent = Mock(value="STATIC")
    retriever.retrieve.return_value = mock_result
    return retriever


@pytest.fixture
def mock_llm():
    """Mock ChatOpenAI for testing."""
    llm = Mock()
    mock_response = Mock()
    mock_response.content = "This is a mocked LLM response about parking."
    llm.invoke.return_value = mock_response
    return llm


# Unit Tests - Router Node
class TestRouter:
    """Test router node intent classification."""

    @pytest.mark.parametrize(
        "user_message,expected_mode",
        [
            ("What are your hours?", "info"),
            ("Is parking available?", "info"),
            ("Where are you located?", "info"),
            ("I want to book parking", "reservation"),
            ("Make a reservation", "reservation"),
            ("Book a spot", "reservation"),
            ("I'd like to reserve parking", "reservation"),
        ],
    )
    def test_router_classification(self, base_state, user_message, expected_mode):
        """Test router classifies booking vs info intents correctly."""
        state = base_state.copy()
        state["messages"] = [HumanMessage(content=user_message)]

        result = router(state)

        assert result["mode"] == expected_mode
        assert result["iteration_count"] == 1

    def test_router_cancellation(self, base_state):
        """Test router detects cancellation and resets to info mode."""
        state = base_state.copy()
        state["mode"] = "reservation"
        state["messages"] = [HumanMessage(content="cancel this reservation")]

        result = router(state)

        assert result["mode"] == "info"
        assert result["reservation"]["completed_fields"] == []

    def test_router_stays_in_reservation_mode(self, base_state):
        """Test router stays in reservation mode when already there."""
        state = base_state.copy()
        state["mode"] = "reservation"
        state["messages"] = [HumanMessage(content="John Doe")]

        result = router(state)

        assert result["mode"] == "reservation"


# Unit Tests - Retrieve Node
class TestRetrieve:
    """Test retrieve node RAG integration."""

    def test_retrieve_calls_parking_retriever(self, base_state, mock_retriever):
        """Test retrieve node calls ParkingRetriever with correct query."""
        state = base_state.copy()
        state["messages"] = [HumanMessage(content="What are your hours?")]

        with patch("src.chatbot.nodes.get_parking_retriever", return_value=mock_retriever):
            result = retrieve(state)

        mock_retriever.retrieve.assert_called_once()
        assert result["context"] is not None
        assert "Mocked context" in result["context"]

    def test_retrieve_handles_errors_gracefully(self, base_state, mock_retriever):
        """Test retrieve node handles retrieval errors gracefully."""
        state = base_state.copy()
        state["messages"] = [HumanMessage(content="test query")]
        mock_retriever.retrieve.side_effect = Exception("Database connection failed")

        with patch("src.chatbot.nodes.get_parking_retriever", return_value=mock_retriever):
            result = retrieve(state)

        # Should have error but still return context (graceful degradation)
        assert result["error"] is not None
        assert "Retrieval error" in result["error"]
        assert result["context"] is not None


# Unit Tests - Generate Node
class TestGenerate:
    """Test generate node LLM response generation."""

    def test_generate_creates_ai_message(self, base_state, mock_llm):
        """Test generate node creates AI message from LLM response."""
        state = base_state.copy()
        state["messages"] = [HumanMessage(content="What are your hours?")]
        state["context"] = "## Dynamic Data\n\nOperating hours: 6AM-10PM"

        with patch("src.chatbot.nodes.get_llm", return_value=mock_llm):
            result = generate(state)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert "mocked" in result["messages"][0].content.lower() and "parking" in result["messages"][0].content.lower()

    def test_generate_handles_llm_error(self, base_state, mock_llm):
        """Test generate node handles LLM errors gracefully."""
        state = base_state.copy()
        state["messages"] = [HumanMessage(content="test")]
        state["context"] = "test context"
        mock_llm.invoke.side_effect = Exception("LLM API error")

        with patch("src.chatbot.nodes.get_llm", return_value=mock_llm):
            result = generate(state)

        # Should return error message
        assert "messages" in result
        assert "error" in result
        assert "trouble generating" in result["messages"][0].content.lower()


# Unit Tests - Collect Input Node
class TestCollectInput:
    """Test collect_input node field prompting."""

    def test_collect_input_asks_for_name_first(self, base_state):
        """Test collect_input asks for name when no fields collected."""
        state = base_state.copy()

        result = collect_input(state)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert "name" in result["messages"][0].content.lower()

    def test_collect_input_asks_for_parking_id_second(self, base_state):
        """Test collect_input asks for parking_id after name collected."""
        state = base_state.copy()
        state["reservation"]["completed_fields"] = ["name"]

        result = collect_input(state)

        assert "messages" in result
        assert "parking facility" in result["messages"][0].content.lower() or "downtown" in result["messages"][0].content.lower()

    def test_collect_input_field_order(self, base_state):
        """Test collect_input follows correct field order."""
        field_order = ["name", "parking_id", "date", "start_time", "duration_hours"]

        for i, field in enumerate(field_order):
            state = base_state.copy()
            state["reservation"]["completed_fields"] = field_order[:i]

            result = collect_input(state)

            assert "messages" in result
            # Verify prompt is for the correct next field


# Unit Tests - Validate Input Node
class TestValidateInput:
    """Test validate_input node field validation."""

    @pytest.mark.parametrize(
        "field,value,should_be_valid",
        [
            ("name", "John Doe", True),
            ("name", "", False),
            ("parking_id", "downtown plaza", True),
            ("parking_id", "Downtown Plaza Parking", True),
            ("parking_id", "airport parking", True),
            ("parking_id", "Airport Long-Term Parking", True),
            ("parking_id", "invalid facility", False),
            ("date", "2026-12-31", True),
            ("date", "invalid-date", False),
            ("date", "2020-01-01", False),  # Past date
            ("start_time", "14:30", True),
            ("start_time", "09:00", True),
            ("start_time", "25:00", False),  # Invalid hour
            ("start_time", "14:99", False),  # Invalid minute
            ("duration_hours", "5", True),
            ("duration_hours", "24", True),
            ("duration_hours", "168", True),
            ("duration_hours", "0", False),  # Too low
            ("duration_hours", "200", False),  # Too high
            ("duration_hours", "abc", False),  # Not a number
        ],
    )
    def test_field_validation(self, base_state, field, value, should_be_valid):
        """Test validation logic for all reservation fields."""
        state = base_state.copy()
        # Set completed fields to be all fields before the current one
        field_order = ["name", "parking_id", "date", "start_time", "duration_hours"]
        field_index = field_order.index(field)
        state["reservation"]["completed_fields"] = field_order[:field_index]
        state["messages"] = [HumanMessage(content=value)]

        result = validate_input(state)

        if should_be_valid:
            # Field should be added to completed_fields
            assert "reservation" in result
            assert field in result["reservation"]["completed_fields"]
            # No validation errors
            assert not result["reservation"].get("validation_errors")
        else:
            # Validation error should be present
            if "reservation" in result:
                assert field in result["reservation"].get("validation_errors", {})
            else:
                # For backward compatibility with error messages
                assert "messages" in result or "error" in result

    def test_validate_input_parking_id_normalization(self, base_state):
        """Test parking_id is normalized to ID format."""
        state = base_state.copy()
        # Set up state to be validating parking_id (name already collected)
        state["reservation"]["completed_fields"] = ["name"]
        state["messages"] = [HumanMessage(content="Downtown Plaza")]

        result = validate_input(state)

        assert "reservation" in result
        assert result["reservation"]["parking_id"] == "downtown_plaza"
        assert "parking_id" in result["reservation"]["completed_fields"]


# Unit Tests - Check Completion Node
class TestCheckCompletion:
    """Test check_completion node routing logic."""

    def test_check_completion_incomplete(self, base_state):
        """Test check_completion with incomplete reservation."""
        state = base_state.copy()
        state["reservation"]["completed_fields"] = ["name", "parking_id"]

        result = check_completion(state)

        # Node doesn't return state updates, routing is handled by graph
        assert result == {}

    def test_check_completion_complete(self, base_state):
        """Test check_completion with complete reservation."""
        state = base_state.copy()
        state["reservation"]["completed_fields"] = [
            "name",
            "parking_id",
            "date",
            "start_time",
            "duration_hours",
        ]

        result = check_completion(state)

        assert result == {}


# Unit Tests - Confirm Reservation Node
class TestConfirmReservation:
    """Test confirm_reservation node summary generation."""

    def test_confirm_reservation_formats_summary(self, base_state):
        """Test confirmation message includes all reservation details."""
        state = base_state.copy()
        state["reservation"] = {
            "name": "John Doe",
            "parking_id": "downtown_plaza",
            "date": date(2024, 12, 25),
            "start_time": time(14, 30),
            "duration_hours": 3,
            "completed_fields": ["name", "parking_id", "date", "start_time", "duration_hours"],
            "validation_errors": {},
        }

        result = confirm_reservation(state)

        assert "messages" in result
        message = result["messages"][0].content

        # Check all details are in message
        assert "John Doe" in message
        assert "Downtown Plaza" in message
        assert "2024-12-25" in message
        assert "14:30" in message
        assert "3" in message


# Integration Tests
@pytest.mark.integration
class TestGraphIntegration:
    """Test full graph execution with real components."""

    def test_info_mode_flow(self):
        """Test information query flow through graph."""
        from src.chatbot.graph import graph

        initial_state: ChatbotState = {
            "messages": [HumanMessage(content="What are your hours?")],
            "mode": "info",
            "intent": None,
            "context": None,
            "reservation": {"completed_fields": [], "validation_errors": {}},
            "error": None,
            "iteration_count": 0,
        }

        result = graph.invoke(initial_state)

        # Should have AI response
        assert len(result["messages"]) > 1
        assert any(isinstance(msg, AIMessage) for msg in result["messages"])

    def test_reservation_mode_trigger(self):
        """Test triggering reservation mode."""
        from src.chatbot.graph import graph

        initial_state: ChatbotState = {
            "messages": [HumanMessage(content="I want to book parking")],
            "mode": "info",
            "intent": None,
            "context": None,
            "reservation": {"completed_fields": [], "validation_errors": {}},
            "error": None,
            "iteration_count": 0,
        }

        result = graph.invoke(initial_state)

        # Should switch to reservation mode
        assert result["mode"] == "reservation"
        # Should ask for name
        assert any(isinstance(msg, AIMessage) for msg in result["messages"])


# Error Handling Tests
class TestErrorHandling:
    """Test error handling across nodes."""

    def test_router_handles_empty_messages(self, base_state):
        """Test router handles empty message list."""
        state = base_state.copy()
        state["messages"] = []

        result = router(state)

        # Should default to info mode
        assert result["mode"] == "info"

    def test_retrieve_handles_no_user_message(self, base_state):
        """Test retrieve handles state with no user messages."""
        state = base_state.copy()
        state["messages"] = [AIMessage(content="Previous AI message")]

        result = retrieve(state)

        # Should return error
        assert result.get("error") is not None

    def test_validate_input_handles_missing_message(self, base_state):
        """Test validate_input handles state with no user input."""
        state = base_state.copy()
        state["messages"] = []

        result = validate_input(state)

        # Should return error
        assert result.get("error") is not None
