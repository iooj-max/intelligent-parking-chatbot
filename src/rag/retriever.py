"""
Unified RAG retriever for parking chatbot.

This module orchestrates retrieval from Weaviate (static content) and
PostgreSQL (dynamic data) to provide comprehensive context for LLM responses.

FOR TESTING PURPOSES ONLY - Not production-ready implementation.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from .sql_store import SQLStore
from .vector_store import WeaviateStore
from ..data.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Query intent classification."""

    STATIC = "static"  # General info, features, location, policies
    DYNAMIC = "dynamic"  # Availability, pricing, hours
    HYBRID = "hybrid"  # Needs both static and dynamic
    RESERVATION = "reservation"  # Booking intent


@dataclass
class RetrievalResult:
    """Structured retrieval result."""

    query: str
    intent: QueryIntent
    parking_id: Optional[str]
    static_chunks: List[Dict[str, Any]]
    dynamic_data: Dict[str, Any]
    context_string: str
    metadata: Dict[str, Any]


class RetrievalError(Exception):
    """Raised when retrieval fails critically."""

    pass


# Intent classification keywords
DYNAMIC_KEYWORDS = [
    "available",
    "availability",
    "spaces",
    "price",
    "pricing",
    "cost",
    "rate",
    "hours",
    "open",
    "close",
    "closed",
    "special hours",
    "holiday",
    "now",
    "today",
    "currently",
    "real-time",
    "occupied",
    "full",
]

STATIC_KEYWORDS = [
    "location",
    "address",
    "where",
    "features",
    "amenities",
    "electric",
    "charging",
    "security",
    "policy",
    "policies",
    "cancel",
    "refund",
    "booking process",
    "how to book",
    "payment",
    "accept",
    "handicap",
    "accessible",
    "height",
    "restriction",
    "valet",
    "monthly pass",
    "contact",
    "phone",
    "email",
    "faq",
    "question",
]

RESERVATION_KEYWORDS = [
    "book a spot",
    "book parking",
    "make reservation",
    "reserve parking",
    "i want to park",
    "i want to book",
]

# Parking ID inference patterns
PARKING_ID_PATTERNS = {
    "downtown_plaza": [
        "downtown plaza",
        "downtown parking",
        "main street",
        "123 main",
        "plaza parking",
    ],
    "airport_parking": [
        "airport parking",
        "airport lot",
        "long-term parking",
        "long term parking",
        "4500 airport",
        "airport boulevard",
    ],
}


class ParkingRetriever:
    """
    Unified retriever for parking chatbot RAG system.

    Orchestrates retrieval from Weaviate (static content) and PostgreSQL
    (dynamic data) based on query intent classification.
    """

    def __init__(
        self,
        vector_store: WeaviateStore,
        sql_store: SQLStore,
        embedding_generator: EmbeddingGenerator,
        max_static_chunks: int = 5,
        max_context_tokens: int = 2000,
    ):
        """
        Initialize retriever with data stores.

        Args:
            vector_store: Weaviate vector store instance
            sql_store: PostgreSQL store instance
            embedding_generator: Embedding generator for query vectors
            max_static_chunks: Maximum number of Weaviate chunks to retrieve
            max_context_tokens: Maximum total context length (approximate)
        """
        self.vector_store = vector_store
        self.sql_store = sql_store
        self.embedding_generator = embedding_generator
        self.max_static_chunks = max_static_chunks
        self.max_context_tokens = max_context_tokens

    def retrieve(
        self,
        query: str,
        parking_id: Optional[str] = None,
        intent: Optional[QueryIntent] = None,
        return_format: str = "string",
    ) -> Union[str, RetrievalResult]:
        """
        Main retrieval method.

        Args:
            query: User query
            parking_id: Specific parking facility ID (optional)
            intent: Pre-classified intent (optional, will auto-classify if None)
            return_format: "string" for LLM context, "structured" for RetrievalResult

        Returns:
            Context string or RetrievalResult object

        Raises:
            RetrievalError: If critical retrieval failure occurs
        """
        try:
            # Classify intent if not provided
            if intent is None:
                intent = self._classify_query_intent(query)

            logger.info(f"Query intent classified as: {intent.value}")

            # Infer parking_id if not provided
            if not parking_id:
                parking_id = self._infer_parking_id(query)
                if parking_id:
                    logger.info(f"Inferred parking_id: {parking_id}")

            static_chunks = []
            dynamic_data = {}

            # Retrieve static content
            if intent in (QueryIntent.STATIC, QueryIntent.HYBRID):
                try:
                    static_chunks = self._retrieve_static_content(query, parking_id)
                    logger.info(f"Retrieved {len(static_chunks)} static chunks")
                except Exception as e:
                    logger.error(f"Failed to retrieve static content: {e}")
                    # Continue without static content

            # Retrieve dynamic data
            if intent in (QueryIntent.DYNAMIC, QueryIntent.HYBRID) and parking_id:
                try:
                    dynamic_data = self._retrieve_dynamic_data(parking_id)
                    logger.info(f"Retrieved dynamic data for {parking_id}")
                except Exception as e:
                    logger.error(f"Failed to retrieve dynamic data: {e}")
                    # Continue without dynamic data

            # Handle empty results
            if not static_chunks and not dynamic_data:
                logger.warning(f"No results found for query: {query}")
                return self._format_empty_result(query, parking_id)

            # Format context
            context_string = self._format_context_string(static_chunks, dynamic_data, parking_id)

            # Return in requested format
            if return_format == "string":
                return context_string
            else:
                return RetrievalResult(
                    query=query,
                    intent=intent,
                    parking_id=parking_id,
                    static_chunks=static_chunks,
                    dynamic_data=dynamic_data,
                    context_string=context_string,
                    metadata=self._extract_metadata(static_chunks, dynamic_data),
                )

        except Exception as e:
            logger.exception(f"Critical error in retrieval: {e}")
            raise RetrievalError(f"Failed to retrieve context for query: {query}") from e

    def _classify_query_intent(self, query: str) -> QueryIntent:
        """
        Classify query intent using keyword matching.

        Args:
            query: User query

        Returns:
            QueryIntent enum value
        """
        query_lower = query.lower()

        # Count keyword matches
        dynamic_count = sum(kw in query_lower for kw in DYNAMIC_KEYWORDS)
        static_count = sum(kw in query_lower for kw in STATIC_KEYWORDS)

        # Reservation intent takes precedence
        if any(kw in query_lower for kw in RESERVATION_KEYWORDS):
            return QueryIntent.RESERVATION

        # If both types present, return HYBRID
        if dynamic_count > 0 and static_count > 0:
            return QueryIntent.HYBRID

        # Prefer dynamic if any dynamic keywords present
        if dynamic_count > 0:
            return QueryIntent.DYNAMIC

        if static_count > 0:
            return QueryIntent.STATIC

        # Default to HYBRID (safer to retrieve too much)
        return QueryIntent.HYBRID

    def _infer_parking_id(self, query: str) -> Optional[str]:
        """
        Attempt to infer parking_id from query text.

        Args:
            query: User query

        Returns:
            Parking ID if confidently inferred, None otherwise
        """
        query_lower = query.lower()

        for parking_id, patterns in PARKING_ID_PATTERNS.items():
            if any(pattern in query_lower for pattern in patterns):
                return parking_id

        return None

    def _retrieve_static_content(
        self,
        query: str,
        parking_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant static content from Weaviate.

        Args:
            query: User query
            parking_id: Filter by parking facility (optional)

        Returns:
            List of relevant content chunks with metadata
        """
        # Generate query embedding
        query_vector = self.embedding_generator.generate(query)

        # Search Weaviate
        results = self.vector_store.search_similar(
            query_vector=query_vector,
            parking_id=parking_id,
            limit=self.max_static_chunks,
            return_metadata=True,
        )

        return results

    def _retrieve_dynamic_data(self, parking_id: str) -> Dict[str, Any]:
        """
        Retrieve all dynamic data for a parking facility from PostgreSQL.

        Args:
            parking_id: Facility identifier

        Returns:
            Dictionary containing availability, hours, pricing, etc.
        """
        data = {}

        # Get availability
        availability = self.sql_store.get_availability(parking_id)
        if availability:
            data["availability"] = availability

        # Get working hours
        working_hours = self.sql_store.get_working_hours(parking_id)
        if working_hours:
            data["working_hours"] = working_hours

        # Get special hours
        special_hours = self.sql_store.get_special_hours(parking_id)
        if special_hours:
            data["special_hours"] = special_hours

        # Get pricing rules
        pricing_rules = self.sql_store.get_pricing_rules(parking_id, active_only=True)
        if pricing_rules:
            data["pricing_rules"] = pricing_rules

        return data

    def _format_context_string(
        self,
        static_chunks: List[Dict[str, Any]],
        dynamic_data: Dict[str, Any],
        parking_id: Optional[str],
    ) -> str:
        """
        Format retrieved context as structured markdown.

        Args:
            static_chunks: List of static content chunks from Weaviate
            dynamic_data: Dynamic data dictionary from PostgreSQL
            parking_id: Parking facility ID

        Returns:
            Formatted context string ready for LLM consumption
        """
        sections = []

        # 1. Static content section
        if static_chunks:
            static_section = "## Static Information\n\n"
            for i, chunk in enumerate(static_chunks, 1):
                metadata = chunk.get("metadata", {})
                parking_name = metadata.get("parking_name", chunk.get("parking_id", "Unknown"))

                static_section += f"### [{parking_name}] {chunk['content_type'].replace('_', ' ').title()}\n"
                static_section += f"{chunk['content']}\n\n"
                static_section += f"*Source: {chunk['source_file']}*\n\n"
                static_section += "---\n\n"

            sections.append(static_section)

        # 2. Dynamic data section
        if dynamic_data:
            dynamic_section = "## Dynamic Data (Real-time)\n\n"

            # Availability
            if "availability" in dynamic_data:
                avail = dynamic_data["availability"]
                if avail is not None:
                    dynamic_section += "### Current Availability\n"
                    dynamic_section += f"- Total Spaces: {avail.total_spaces}\n"
                    dynamic_section += f"- Occupied: {avail.occupied_spaces}\n"
                    dynamic_section += f"- **Available: {avail.available_spaces}**\n"
                    dynamic_section += f"- Last Updated: {avail.last_updated.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                else:
                    logger.warning("Availability data is None")

            # Working hours
            if "working_hours" in dynamic_data:
                dynamic_section += "### Operating Hours\n"
                day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                for hours in dynamic_data["working_hours"]:
                    day_name = day_names[hours.day_of_week]
                    if hours.is_closed:
                        dynamic_section += f"- {day_name}: Closed\n"
                    else:
                        dynamic_section += (
                            f"- {day_name}: {hours.open_time.strftime('%H:%M')} - {hours.close_time.strftime('%H:%M')}\n"
                        )
                dynamic_section += "\n"

            # Special hours
            if "special_hours" in dynamic_data and dynamic_data["special_hours"]:
                dynamic_section += "### Special Hours & Closures\n"
                for special in dynamic_data["special_hours"]:
                    if special.is_closed:
                        dynamic_section += f"- {special.date}: Closed"
                    else:
                        dynamic_section += f"- {special.date}: {special.open_time.strftime('%H:%M')} - {special.close_time.strftime('%H:%M')}"

                    if special.reason:
                        dynamic_section += f" ({special.reason})"
                    dynamic_section += "\n"
                dynamic_section += "\n"

            # Pricing rules
            if "pricing_rules" in dynamic_data:
                dynamic_section += "### Current Pricing\n"
                for rule in dynamic_data["pricing_rules"]:
                    dynamic_section += f"- **{rule.rule_name}**: ${rule.price_per_unit}/{rule.time_unit}\n"
                    if rule.time_start and rule.time_end:
                        dynamic_section += f"  - Valid: {rule.time_start.strftime('%H:%M')} - {rule.time_end.strftime('%H:%M')}\n"
                    if rule.day_of_week_start is not None and rule.day_of_week_end is not None:
                        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                        day_range = f"{days[rule.day_of_week_start]}-{days[rule.day_of_week_end]}"
                        dynamic_section += f"  - Days: {day_range}\n"
                dynamic_section += "\n"

            sections.append(dynamic_section)

        # Combine sections
        context = "\n".join(sections)

        # Add instructions footer
        context += "\n---\n\n"
        context += "**Instructions**: Use the above context to answer the user's query accurately. "
        context += "Prioritize dynamic data for real-time information. "
        context += "If specific information is not available, inform the user politely.\n"

        return context

    def _format_empty_result(self, query: str, parking_id: Optional[str]) -> str:
        """
        Format response when no results are found.

        Args:
            query: Original user query
            parking_id: Parking ID if known

        Returns:
            Helpful message string
        """
        message = "## No Information Found\n\n"
        message += f"I couldn't find specific information to answer your query: \"{query}\"\n\n"

        if not parking_id:
            message += "Could you please specify which parking facility you're asking about? "
            message += "We have Downtown Plaza Parking and Airport Long-Term Parking.\n"
        else:
            message += f"I don't have information about this for {parking_id}. "
            message += "Could you try rephrasing your question?\n"

        return message

    def _extract_metadata(
        self,
        static_chunks: List[Dict[str, Any]],
        dynamic_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extract metadata from retrieval results.

        Args:
            static_chunks: Static content chunks
            dynamic_data: Dynamic data dictionary

        Returns:
            Metadata dictionary
        """
        metadata = {}

        # Extract parking names and addresses from static chunks
        parking_facilities = {}
        for chunk in static_chunks:
            chunk_metadata = chunk.get("metadata", {})
            parking_id = chunk.get("parking_id")
            if parking_id and chunk_metadata:
                parking_facilities[parking_id] = {
                    "name": chunk_metadata.get("parking_name"),
                    "address": chunk_metadata.get("address"),
                    "city": chunk_metadata.get("city"),
                }

        metadata["parking_facilities"] = parking_facilities

        return metadata

    def _estimate_tokens(self, text: str) -> int:
        """
        Rough estimation of token count.

        Args:
            text: Text content

        Returns:
            Estimated token count
        """
        return len(text) // 4
