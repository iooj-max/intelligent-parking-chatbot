"""Retriever for static parking knowledge.

The assistant decides which tool to call. This retriever is used by the
`search_parking_info` tool for vector-based lookup of static content.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from .sql_store import SQLStore
from .vector_store import WeaviateStore
from ..data.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Structured retrieval result."""

    query: str
    parking_id: Optional[str]
    static_chunks: List[Dict[str, Any]]
    context_string: str
    metadata: Dict[str, Any]


class RetrievalError(Exception):
    """Raised when retrieval fails critically."""


class ParkingRetriever:
    """Vector retriever for static parking content."""

    def __init__(
        self,
        vector_store: WeaviateStore,
        sql_store: SQLStore,
        embedding_generator: EmbeddingGenerator,
        max_static_chunks: int = 5,
        max_context_tokens: int = 2000,
    ):
        self.vector_store = vector_store
        # kept for backward-compatible constructor signature
        self.sql_store = sql_store
        self.embedding_generator = embedding_generator
        self.max_static_chunks = max_static_chunks
        self.max_context_tokens = max_context_tokens

    def retrieve(
        self,
        query: str,
        parking_id: Optional[str] = None,
        return_format: str = "string",
    ) -> Union[str, RetrievalResult]:
        """Retrieve static context for a query."""
        try:
            if not parking_id:
                parking_id = self._infer_parking_id(query)
                if parking_id:
                    logger.info("Inferred parking_id: %s", parking_id)

            static_chunks = self._retrieve_static_content(query, parking_id)
            logger.info("Retrieved %d static chunks", len(static_chunks))

            if not static_chunks:
                logger.warning("No results found for query: %s", query)
                return self._format_empty_result(query, parking_id, return_format)

            context_string = self._format_context_string(static_chunks, parking_id)

            if return_format == "string":
                return context_string

            return RetrievalResult(
                query=query,
                parking_id=parking_id,
                static_chunks=static_chunks,
                context_string=context_string,
                metadata=self._extract_metadata(static_chunks),
            )

        except Exception as e:
            logger.exception("Critical error in retrieval: %s", e)
            raise RetrievalError(f"Failed to retrieve context for query: {query}") from e

    def _infer_parking_id(self, query: str) -> Optional[str]:
        """Attempt to infer parking_id from query text."""
        from src.services.parking_matcher import ParkingFacilityMatcher
        from src.services.parking_service import get_parking_service

        service = get_parking_service()
        matcher = ParkingFacilityMatcher(threshold=0.7)

        facilities = service.get_all_facilities()
        matches = matcher.match_facility(query, facilities, limit=1)

        if matches and matches[0]["score"] >= 0.7:
            parking_id = matches[0]["parking_id"]
            logger.info("Inferred parking_id: %s (score: %.2f)", parking_id, matches[0]["score"])
            return parking_id

        return None

    def _retrieve_static_content(
        self,
        query: str,
        parking_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant static content from Weaviate."""
        query_vector = self.embedding_generator.generate(query)
        return self.vector_store.search_similar(
            query_vector=query_vector,
            parking_id=parking_id,
            limit=self.max_static_chunks,
            return_metadata=True,
        )

    def _format_context_string(
        self,
        static_chunks: List[Dict[str, Any]],
        parking_id: Optional[str],
    ) -> str:
        """Format retrieved static context as markdown."""
        sections = ["# Parking Information Context\n"]

        if parking_id:
            sections.append(f"## Parking Facility: {parking_id}\n")

        if static_chunks:
            sections.append("## Relevant Information\n")
            current_tokens = 0

            for i, chunk in enumerate(static_chunks, start=1):
                content = chunk.get("text", "")
                metadata = chunk.get("metadata", {})

                chunk_tokens = self._estimate_tokens(content)
                if current_tokens + chunk_tokens > self.max_context_tokens:
                    logger.warning(
                        "Context token limit reached (%d), truncating",
                        self.max_context_tokens,
                    )
                    break

                source_file = chunk.get("source_file", "unknown")
                content_type = metadata.get("content_type", "general")

                sections.append(f"### Chunk {i} ({content_type})")
                sections.append(f"Source: `{source_file}`")
                sections.append(content)
                sections.append("")

                current_tokens += chunk_tokens

        context = "\n".join(sections)
        context += "\n---\n\n"
        context += "**Instructions**: Use the above context to answer the user's query accurately. "
        context += "If specific information is not available, inform the user politely.\n"

        return context

    def _format_empty_result(
        self,
        query: str,
        parking_id: Optional[str],
        return_format: str,
    ) -> Union[str, RetrievalResult]:
        if return_format != "string":
            return RetrievalResult(
                query=query,
                parking_id=parking_id,
                static_chunks=[],
                context_string="",
                metadata=self._extract_metadata([]),
            )

        if parking_id:
            return (
                f"I couldn't find relevant information for '{query}' at {parking_id}. "
                "Please try rephrasing your question or ask about a different topic."
            )

        return (
            f"I couldn't find relevant parking information for '{query}'. "
            "Please try rephrasing your question."
        )

    def _extract_metadata(
        self,
        static_chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Extract metadata summary from retrieved chunks."""
        metadata = {
            "num_static_chunks": len(static_chunks),
            "source_files": list(
                {
                    chunk.get("source_file")
                    for chunk in static_chunks
                    if chunk.get("source_file")
                }
            ),
            "content_types": list(
                {
                    chunk.get("metadata", {}).get("content_type")
                    for chunk in static_chunks
                    if chunk.get("metadata", {}).get("content_type")
                }
            ),
        }

        parking_facilities = {}
        for chunk in static_chunks:
            chunk_metadata = chunk.get("metadata", {})
            parking_id = chunk_metadata.get("parking_id")
            if parking_id and parking_id not in parking_facilities:
                parking_facilities[parking_id] = {
                    "name": chunk_metadata.get("parking_name"),
                    "address": chunk_metadata.get("address"),
                    "city": chunk_metadata.get("city"),
                }

        metadata["parking_facilities"] = parking_facilities
        return metadata

    def _estimate_tokens(self, text: str) -> int:
        """Rough estimation of token count."""
        return len(text) // 4
