"""
Tests for intent classification and output validation.

Tests strict domain constraints:
- Intent classifier correctly identifies 6 categories
- OUT_OF_SCOPE detection for off-topic queries
- Output validator rejects forbidden content
- Output validator accepts parking responses
"""

import pytest
from src.chatbot.intent_classifier import IntentClassifier, ParkingIntent
from src.chatbot.output_validator import OutputValidator


class TestIntentClassifier:
    """Test intent classification layer"""

    @pytest.fixture
    def classifier(self):
        """Fixture for IntentClassifier instance"""
        return IntentClassifier()

    def test_out_of_scope_detection(self, classifier):
        """Test various off-topic queries are classified as OUT_OF_SCOPE"""
        # Should all be OUT_OF_SCOPE
        out_of_scope_queries = [
            "Write me a poem about cars",
            "Tell me a joke",
            "What's 2 + 2?",
            "Who is the president?",
            "Write a story about parking",
            "Calculate fibonacci sequence",
            "How to cook pasta?",
            "Tell me about cryptocurrency",
            "What's the weather like?",
        ]

        for query in out_of_scope_queries:
            intent = classifier.classify(query)
            assert intent == ParkingIntent.OUT_OF_SCOPE, f"Failed to detect out_of_scope for: {query}"

    def test_parking_availability_intent(self, classifier):
        """Test parking availability queries"""
        availability_queries = [
            "Is there parking available?",
            "How many spaces are left?",
            "Is the lot full?",
            "Are there any spaces downtown?",
        ]

        for query in availability_queries:
            intent = classifier.classify(query)
            assert intent == ParkingIntent.PARKING_AVAILABILITY, f"Failed for: {query}"

    def test_pricing_info_intent(self, classifier):
        """Test pricing information queries"""
        pricing_queries = [
            "How much does it cost?",
            "What's the daily rate?",
            "Pricing for 3 hours?",
            "Do you have special rates?",
        ]

        for query in pricing_queries:
            intent = classifier.classify(query)
            assert intent == ParkingIntent.PRICING_INFO, f"Failed for: {query}"

    def test_reservation_intent(self, classifier):
        """Test reservation requests"""
        reservation_queries = [
            "I want to book parking",
            "Reserve a spot",
            "Make a reservation",
            "Book for tomorrow",
        ]

        for query in reservation_queries:
            intent = classifier.classify(query)
            assert intent == ParkingIntent.RESERVATION, f"Failed for: {query}"

    def test_working_hours_intent(self, classifier):
        """Test operating hours queries"""
        hours_queries = [
            "What are your hours?",
            "When do you open?",
            "Are you open on weekends?",
            "Holiday hours?",
        ]

        for query in hours_queries:
            intent = classifier.classify(query)
            assert intent == ParkingIntent.WORKING_HOURS, f"Failed for: {query}"

    def test_facility_info_intent(self, classifier):
        """Test facility information queries"""
        facility_queries = [
            "Where are you located?",
            "Do you have EV charging?",
            "What's the cancellation policy?",
            "How do I contact you?",
        ]

        for query in facility_queries:
            intent = classifier.classify(query)
            assert intent == ParkingIntent.FACILITY_INFO, f"Failed for: {query}"


class TestOutputValidator:
    """Test output validation layer"""

    @pytest.fixture
    def validator(self):
        """Fixture for OutputValidator instance"""
        return OutputValidator()

    def test_rejects_off_topic_responses(self, validator):
        """Test validation rejects off-topic outputs"""
        # Should be rejected - contain forbidden patterns
        off_topic_responses = [
            "Here's a poem about cars: Roses are red...",
            "Let me tell you a joke about parking...",
            "The recipe for success is simple...",
            "Let me calculate that for you: 2+2=4",
            "Here's a story about the parking lot...",
        ]

        for response in off_topic_responses:
            result = validator.validate(response)
            assert not result["is_valid"], f"Should reject: {response}"
            assert result["response"] == validator.OFF_TOPIC_MESSAGE

    def test_rejects_responses_without_parking_keywords(self, validator):
        """Test validation rejects responses missing parking keywords"""
        # Should be rejected - no parking keywords
        non_parking_responses = [
            "Hello, how can I help you today?",
            "I'm sorry, I don't understand.",
            "That's an interesting question.",
        ]

        for response in non_parking_responses:
            result = validator.validate(response)
            assert not result["is_valid"], f"Should reject: {response}"
            assert result["response"] == validator.OFF_TOPIC_MESSAGE

    def test_accepts_parking_responses(self, validator):
        """Test validation accepts parking outputs"""
        # Should be accepted - contain parking keywords
        parking_responses = [
            "Downtown Plaza has 45 available parking spaces.",
            "The hourly rate is $5 per hour.",
            "We are open 24/7 for your convenience.",
            "The facility is located at 123 Main Street.",
            "Yes, we have EV charging stations available.",
            "You can reserve a spot through our booking system.",
        ]

        for response in parking_responses:
            result = validator.validate(response)
            assert result["is_valid"], f"Should accept: {response}"
            assert result["response"] == response  # Original response returned

    def test_forbidden_patterns_take_precedence(self, validator):
        """Test forbidden patterns are checked before keyword check"""
        # Contains parking keywords BUT also forbidden pattern
        response = "Here's a joke about the parking garage..."

        result = validator.validate(response)
        assert not result["is_valid"]
        assert "joke" in result["reason"].lower()

    def test_empty_response_rejected(self, validator):
        """Test empty responses are rejected"""
        result = validator.validate("")
        assert not result["is_valid"]

    def test_multiple_parking_keywords_accepted(self, validator):
        """Test responses with multiple parking keywords are accepted"""
        response = "The parking facility has 250 available spaces. You can reserve a spot online. The hourly rate is $5."

        result = validator.validate(response)
        assert result["is_valid"]
        assert result["response"] == response
