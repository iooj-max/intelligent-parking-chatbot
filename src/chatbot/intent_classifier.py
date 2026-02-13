"""
Intent classification for parking chatbot queries.

Uses an LLM to classify user queries into one of six parking-related
intent categories. This classifier enables the chatbot to route queries
to the appropriate handler (e.g., availability lookup, pricing calculation,
reservation flow) or reject out-of-scope requests.

The classifier is designed to be fail-safe: any classification error
defaults to OUT_OF_SCOPE to prevent incorrect responses.
"""

import logging
from enum import Enum

from langchain_core.messages import HumanMessage

from src.chatbot.nodes import get_llm

logger = logging.getLogger(__name__)


class ParkingIntent(Enum):
    """Enumeration of recognized parking chatbot intents.

    Each value corresponds to a distinct category of user query
    that the chatbot can handle, plus an out-of-scope fallback.

    Attributes:
        PARKING_AVAILABILITY: Questions about available spaces, occupancy, or capacity.
        PRICING_INFO: Questions about rates, fees, costs, or billing.
        RESERVATION: Requests to book or reserve a parking spot.
        WORKING_HOURS: Questions about operating hours, schedules, or holidays.
        FACILITY_INFO: Questions about location, features, amenities, policies, or contact.
        OUT_OF_SCOPE: Anything not related to parking facilities.
    """

    PARKING_AVAILABILITY = "parking_availability"
    PRICING_INFO = "pricing_info"
    RESERVATION = "reservation"
    WORKING_HOURS = "working_hours"
    FACILITY_INFO = "facility_info"
    OUT_OF_SCOPE = "out_of_scope"


class IntentClassifier:
    """Classifies user queries into parking-related intent categories using an LLM.

    This classifier sends the user query to the LLM with a structured prompt
    that constrains the output to one of six predefined categories. It uses
    the shared LLM singleton from the chatbot nodes module.

    The classifier is fail-safe: if the LLM call fails or returns an
    unrecognized category, it defaults to OUT_OF_SCOPE.

    Usage:
        classifier = IntentClassifier()
        intent = classifier.classify("How much does parking cost?")
        # intent == ParkingIntent.PRICING_INFO
    """

    CLASSIFICATION_PROMPT = """You are a parking facility intent classifier.

Classify the user query into EXACTLY ONE category:

PARKING FACILITY DATA (what we can answer):
- Static: facility info, location, contact, features (EV charging, security, shuttle), policies, FAQ, booking process
- Dynamic: real-time availability (spaces), pricing rules (rates), operating hours, special hours/holidays

CATEGORIES:

1. parking_availability - Questions about available spaces, occupancy, capacity
   Examples: "Are there spaces?", "How many left?", "Is it full?"

2. pricing_info - Questions about rates, fees, costs, billing
   Examples: "How much?", "Daily rate?", "Cost for 3 hours?", "Special rates?"

3. reservation - Requests to book/reserve parking
   Examples: "Book parking", "Reserve a spot", "Make reservation"

4. working_hours - Questions about operating hours, schedule, holidays
   Examples: "What hours?", "When open?", "Open weekends?", "Holiday hours?"

5. facility_info - Questions about location, features, amenities, policies, contact
   Examples: "Where located?", "EV charging?", "Cancellation policy?", "Contact info?"

6. out_of_scope - ANYTHING not related to parking facilities
   If query asks to: write creative content, perform calculations unrelated to parking, provide general knowledge, tell jokes, etc. → out_of_scope

User query: "{query}"

Reply with ONLY the category name (e.g., "parking_availability"), nothing else."""

    def __init__(self):
        """Initialize the classifier with the shared LLM instance."""
        self.llm = get_llm()

    def classify(self, user_query: str) -> ParkingIntent:
        """Classify a user query into a parking intent category.

        Sends the query to the LLM with a structured classification prompt
        and maps the response to a ParkingIntent enum value.

        Args:
            user_query: The raw text of the user's message.

        Returns:
            ParkingIntent: The classified intent. Returns OUT_OF_SCOPE
            if the LLM call fails or returns an unrecognized category.
        """
        try:
            prompt = self.CLASSIFICATION_PROMPT.format(query=user_query)
            response = self.llm.invoke([HumanMessage(content=prompt)])
            raw_classification = response.content.strip().lower()

            # Attempt to match the LLM response to a known intent
            for intent in ParkingIntent:
                if intent.value == raw_classification:
                    logger.info(
                        "Classified query as %s: '%s'",
                        intent.name,
                        user_query[:80],
                    )
                    return intent

            # LLM returned something that does not match any known intent
            logger.warning(
                "Unrecognized classification '%s' for query: '%s'. "
                "Defaulting to OUT_OF_SCOPE.",
                raw_classification,
                user_query[:80],
            )
            return ParkingIntent.OUT_OF_SCOPE

        except Exception:
            logger.error(
                "Intent classification failed for query: '%s'. "
                "Defaulting to OUT_OF_SCOPE.",
                user_query[:80],
                exc_info=True,
            )
            return ParkingIntent.OUT_OF_SCOPE
