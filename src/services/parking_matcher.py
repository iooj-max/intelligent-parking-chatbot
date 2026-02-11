"""Semantic matching for parking facility names."""

import logging
from typing import List, Dict, Any
from rapidfuzz import fuzz, process

from src.rag.sql_store import ParkingFacility

logger = logging.getLogger(__name__)


class ParkingFacilityMatcher:
    """
    Semantic matching for parking facility names.
    Uses fuzzy string matching for typo tolerance.
    """

    def __init__(self, threshold: float = 0.6):
        """
        Initialize matcher with similarity threshold.

        Args:
            threshold: Minimum similarity score (0-1) to consider a match
        """
        self.threshold = threshold * 100  # rapidfuzz uses 0-100 scale

    def match_facility(
        self,
        user_query: str,
        facilities: List[ParkingFacility],
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Match user query to parking facilities.

        Uses token set ratio for word order independence and typo tolerance.

        Args:
            user_query: User's search query
            facilities: List of parking facilities to search
            limit: Maximum number of results to return

        Returns:
            List of matches sorted by confidence score:
            [
                {"parking_id": "...", "name": "...", "address": "...", "score": 0.95},
                ...
            ]
        """
        if not user_query or not facilities:
            return []

        # Create searchable strings (name + address for better matching)
        choices = {
            f"{f.name} {f.address}": f
            for f in facilities
        }

        # Use token_set_ratio for word order independence
        matches = process.extract(
            user_query,
            choices.keys(),
            scorer=fuzz.token_set_ratio,
            limit=limit
        )

        results = []
        for match_text, score, _ in matches:
            if score >= self.threshold:
                facility = choices[match_text]
                results.append({
                    "parking_id": facility.parking_id,
                    "name": facility.name,
                    "address": facility.address,
                    "score": score / 100.0  # Normalize to 0-1
                })
                logger.debug(f"Matched '{user_query}' to '{facility.name}' (score: {score})")

        return results
