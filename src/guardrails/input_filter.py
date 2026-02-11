"""
Input validation and filtering for chatbot guardrails.

Protects against:
- Prompt injection attempts
- Off-topic queries
- PII in user input (logging concern)
- Malformed input
"""

import logging
from typing import Any, Dict, List

from src.guardrails.patterns import (
    CARD_REGEX,
    EMAIL_REGEX,
    INJECTION_REGEX,
    OFF_TOPIC_KEYWORDS,
    PARKING_KEYWORDS,
    PHONE_REGEX,
    SSN_REGEX,
)

logger = logging.getLogger(__name__)


class PromptInjectionDetector:
    """Detect prompt injection attempts using pattern matching."""

    def detect(self, text: str) -> Dict[str, Any]:
        """
        Check for prompt injection patterns.

        Args:
            text: User input to check

        Returns:
            {
                'detected': bool,
                'patterns_found': List[str],
                'score': float (0-1)
            }
        """
        patterns_found = []

        for i, pattern in enumerate(INJECTION_REGEX):
            if pattern.search(text):
                patterns_found.append(f"injection_pattern_{i}")

        # Normalize score to 0-1 range (threshold: 3 patterns = max score)
        score = min(len(patterns_found) / 3.0, 1.0)

        return {
            "detected": len(patterns_found) > 0,
            "patterns_found": patterns_found,
            "score": score,
        }


class TopicClassifier:
    """Classify query relevance to parking domain."""

    def is_parking_related(self, text: str) -> Dict[str, Any]:
        """
        Check if query is parking-related.

        Args:
            text: User input to classify

        Returns:
            {
                'is_relevant': bool,
                'parking_score': float,
                'off_topic_score': float,
                'reason': str
            }
        """
        text_lower = text.lower()

        # Count parking keywords
        parking_count = sum(1 for kw in PARKING_KEYWORDS if kw in text_lower)

        # Count off-topic keywords
        off_topic_count = sum(1 for kw in OFF_TOPIC_KEYWORDS if kw in text_lower)

        # Normalize scores
        parking_score = min(parking_count / 3.0, 1.0)
        off_topic_score = min(off_topic_count / 2.0, 1.0)

        # Decision logic
        if off_topic_count > 0 and parking_count == 0:
            return {
                "is_relevant": False,
                "parking_score": parking_score,
                "off_topic_score": off_topic_score,
                "reason": "Query contains off-topic keywords with no parking context",
            }

        # Be more lenient - only reject long queries with no parking keywords
        if parking_count == 0 and len(text.split()) > 10:
            return {
                "is_relevant": False,
                "parking_score": parking_score,
                "off_topic_score": off_topic_score,
                "reason": "Query lacks parking keywords and appears off-topic",
            }

        # For shorter queries or those with parking keywords, be permissive
        return {
            "is_relevant": True,
            "parking_score": parking_score,
            "off_topic_score": off_topic_score,
            "reason": "Query may be parking-related or requires context",
        }


class PIIDetector:
    """Detect PII in user input."""

    def detect_pii(self, text: str) -> Dict[str, Any]:
        """
        Detect PII patterns in text.

        Args:
            text: User input to scan

        Returns:
            {
                'found': bool,
                'types': List[str],
                'sanitized_text': str
            }
        """
        pii_types = []
        sanitized = text

        # Email
        if EMAIL_REGEX.search(text):
            pii_types.append("email")
            sanitized = EMAIL_REGEX.sub("[EMAIL_REDACTED]", sanitized)

        # Phone
        if PHONE_REGEX.search(text):
            pii_types.append("phone")
            sanitized = PHONE_REGEX.sub("[PHONE_REDACTED]", sanitized)

        # SSN
        if SSN_REGEX.search(text):
            pii_types.append("ssn")
            sanitized = SSN_REGEX.sub("[SSN_REDACTED]", sanitized)

        # Credit Card
        if CARD_REGEX.search(text):
            pii_types.append("credit_card")
            sanitized = CARD_REGEX.sub("[CARD_REDACTED]", sanitized)

        return {
            "found": len(pii_types) > 0,
            "types": pii_types,
            "sanitized_text": sanitized,
        }


class InputValidator:
    """
    Orchestrator for all input validation.

    Combines injection detection, topic filtering, and PII detection.
    """

    def __init__(self):
        self.injection_detector = PromptInjectionDetector()
        self.topic_classifier = TopicClassifier()
        self.pii_detector = PIIDetector()

    def validate(self, user_input: str) -> Dict[str, Any]:
        """
        Validate user input against all guardrails.

        Args:
            user_input: Raw user message

        Returns:
            {
                'is_valid': bool,
                'error_message': str | None,
                'warnings': List[str],
                'sanitized_input': str,
                'metadata': dict
            }
        """
        warnings = []

        # Length validation
        if len(user_input) < 1:
            return {
                "is_valid": False,
                "error_message": "Input cannot be empty.",
                "warnings": [],
                "sanitized_input": "",
                "metadata": {},
            }

        if len(user_input) > 1000:
            return {
                "is_valid": False,
                "error_message": "Input too long. Please keep messages under 1000 characters.",
                "warnings": [],
                "sanitized_input": user_input[:1000],
                "metadata": {},
            }

        # Injection detection
        injection_result = self.injection_detector.detect(user_input)
        if injection_result["detected"]:
            logger.warning(
                f"Injection attempt detected: {injection_result['patterns_found']}"
            )
            return {
                "is_valid": False,
                "error_message": "Your message contains patterns that look like prompt injection. Please rephrase your question about parking.",
                "warnings": [
                    f"Injection patterns: {injection_result['patterns_found']}"
                ],
                "sanitized_input": user_input,
                "metadata": injection_result,
            }

        # Topic relevance
        topic_result = self.topic_classifier.is_parking_related(user_input)
        if not topic_result["is_relevant"]:
            logger.info(f"Off-topic query rejected: {topic_result['reason']}")
            return {
                "is_valid": False,
                "error_message": "I can only help with parking-related questions. Please ask about parking availability, pricing, hours, or reservations.",
                "warnings": [topic_result["reason"]],
                "sanitized_input": user_input,
                "metadata": topic_result,
            }

        # PII detection (warning only, not blocking)
        pii_result = self.pii_detector.detect_pii(user_input)
        if pii_result["found"]:
            logger.warning(f"PII detected in user input: {pii_result['types']}")
            warnings.append(f"PII detected: {', '.join(pii_result['types'])}")

        return {
            "is_valid": True,
            "error_message": None,
            "warnings": warnings,
            "sanitized_input": pii_result["sanitized_text"],
            "metadata": {
                "injection": injection_result,
                "topic": topic_result,
                "pii": pii_result,
            },
        }
