"""
Input validation and filtering for security guardrails.

Protects against:
- Prompt injection attempts
- PII in user input (logging concern)
- Malformed input

NOTE: Topic validation (parking vs off-topic) is handled by LLM constitution,
not keyword matching.
"""

import logging
from typing import Any, Dict, List

from src.guardrails.patterns import (
    CARD_REGEX,
    EMAIL_REGEX,
    INJECTION_REGEX,
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
    Orchestrator for security input validation.

    Combines injection detection and PII detection.

    NOTE: Topic filtering (parking vs off-topic) is handled by LLM constitution,
    not keyword matching. This validator only checks SECURITY concerns.
    """

    def __init__(self):
        self.injection_detector = PromptInjectionDetector()
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

        # Injection detection (SECURITY)
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

        # PII detection (SECURITY - warning only, not blocking)
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
                "pii": pii_result,
            },
        }
