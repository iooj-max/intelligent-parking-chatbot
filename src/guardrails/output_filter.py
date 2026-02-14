"""
Output filtering for chatbot responses.

Masks or blocks sensitive data (PII) in LLM-generated responses.
"""

import logging
from typing import Any, Dict

from src.guardrails.patterns import (
    CARD_REGEX,
    EMAIL_REGEX,
    PHONE_REGEX,
    SSN_REGEX,
)

logger = logging.getLogger(__name__)


class PIIMasker:
    """Mask PII patterns in text."""

    def mask(self, text: str) -> Dict[str, Any]:
        """
        Mask PII in response text.

        Args:
            text: Response text to mask

        Returns:
            {
                'masked_text': str,
                'pii_found': List[Dict],
                'count': int
            }
        """
        pii_found = []
        masked_text = text

        # Email masking
        emails = EMAIL_REGEX.findall(text)
        if emails:
            for email in emails:
                pii_found.append({"type": "email", "value": email})
            masked_text = EMAIL_REGEX.sub("[EMAIL]", masked_text)

        # Phone masking
        phones = PHONE_REGEX.findall(text)
        if phones:
            for phone in phones:
                pii_found.append({"type": "phone", "value": phone})
            masked_text = PHONE_REGEX.sub("[PHONE]", masked_text)

        # SSN masking
        ssns = SSN_REGEX.findall(text)
        if ssns:
            for ssn in ssns:
                pii_found.append({"type": "ssn", "value": ssn})
            masked_text = SSN_REGEX.sub("[SSN]", masked_text)

        # Credit card masking
        cards = CARD_REGEX.findall(text)
        if cards:
            for card in cards:
                pii_found.append({"type": "credit_card", "value": card})
            masked_text = CARD_REGEX.sub("[CARD]", masked_text)

        return {
            "masked_text": masked_text,
            "pii_found": pii_found,
            "count": len(pii_found),
        }


class OutputFilter:
    """
    Orchestrator for output filtering.
    """

    def __init__(self):
        self.pii_masker = PIIMasker()

    def filter_response(self, response: str) -> Dict[str, Any]:
        """
        Filter LLM response for sensitive data.

        Args:
            response: Raw LLM output

        Returns:
            {
                'is_safe': bool,
                'filtered_response': str,
                'pii_found': List[Dict],
                'severity': str,  # 'safe', 'low', 'medium', 'high'
            }
        """
        # Mask PII
        mask_result = self.pii_masker.mask(response)

        # Determine severity
        severity = "safe"
        if mask_result["count"] > 0:
            # Check for high-severity patterns (full SSN, full credit card)
            has_ssn = any(
                pii["type"] == "ssn" for pii in mask_result["pii_found"]
            )
            has_card = any(
                pii["type"] == "credit_card" for pii in mask_result["pii_found"]
            )

            if has_ssn or has_card:
                severity = "high"
                logger.error(
                    f"HIGH severity PII in response: {mask_result['pii_found']}"
                )
            elif mask_result["count"] > 2:
                severity = "medium"
                logger.warning(
                    f"MEDIUM severity PII in response: {mask_result['pii_found']}"
                )
            else:
                severity = "low"
                logger.info(
                    f"LOW severity PII masked: {mask_result['pii_found']}"
                )

        # For high severity, block entire response
        if severity == "high":
            return {
                "is_safe": False,
                "filtered_response": "I apologize, but I cannot provide that information due to security policies.",
                "pii_found": mask_result["pii_found"],
                "severity": severity,
            }

        return {
            "is_safe": True,
            "filtered_response": mask_result["masked_text"],
            "pii_found": mask_result["pii_found"],
            "severity": severity,
        }
