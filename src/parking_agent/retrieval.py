"""Document retrieval helpers: retrieval, deduplication, normalization."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_weaviate.vectorstores import WeaviateVectorStore
from weaviate.collections.classes.filters import Filter, FilterReturn


def _document_identity(document: Document) -> tuple[str, str, str]:
    metadata = document.metadata or {}
    parking_id = str(metadata.get("parking_id", "unknown_parking")).strip()
    source_file = str(metadata.get("source_file", "unknown_source")).strip()
    chunk_index = str(metadata.get("chunk_index", "unknown_chunk")).strip()
    return parking_id, source_file, chunk_index


def _normalize_parking_ids(parking_ids: list[str] | None) -> list[str]:
    if not parking_ids:
        return []
    seen: set[str] = set()
    normalized: list[str] = []
    for raw_id in parking_ids:
        value = str(raw_id).strip().lower()
        if value and value not in seen:
            seen.add(value)
            normalized.append(value)
    return normalized


def _build_parking_id_filter(parking_ids: list[str] | None) -> FilterReturn | None:
    normalized_ids = _normalize_parking_ids(parking_ids)
    if not normalized_ids:
        return None
    if len(normalized_ids) == 1:
        return Filter.by_property("parking_id").equal(normalized_ids[0])
    return Filter.any_of(
        [Filter.by_property("parking_id").equal(pid) for pid in normalized_ids]
    )


def build_weaviate_retriever(
    vector_store: WeaviateVectorStore,
    *,
    k: int,
    alpha: float,
    parking_ids: list[str] | None = None,
) -> BaseRetriever:
    search_kwargs: dict[str, Any] = {"k": k, "alpha": alpha}
    parking_filter = _build_parking_id_filter(parking_ids)
    if parking_filter is not None:
        search_kwargs["filters"] = parking_filter
    return vector_store.as_retriever(search_kwargs=search_kwargs)


def deduplicate_documents(
    documents: list[Document],
    *,
    top_k: int,
    max_chunks_per_source_file: int,
) -> list[Document]:
    """Deduplicate and limit documents by (parking_id, source_file) pair."""
    unique_documents: list[Document] = []
    overflow_documents: list[Document] = []
    seen: set[tuple[str, str, str]] = set()
    per_source_key: dict[tuple[str, str], int] = {}

    max_per_source = max(1, max_chunks_per_source_file)
    for document in documents:
        identity = _document_identity(document)
        if identity in seen:
            continue
        seen.add(identity)
        parking_id, source_file = identity[0], identity[1]
        source_key = (parking_id, source_file)
        current_count = per_source_key.get(source_key, 0)
        if current_count < max_per_source:
            unique_documents.append(document)
            per_source_key[source_key] = current_count + 1
        else:
            overflow_documents.append(document)

    if len(unique_documents) < top_k:
        for document in overflow_documents:
            unique_documents.append(document)
            if len(unique_documents) >= top_k:
                break
    return unique_documents[:top_k]


def normalize_document(document: Document) -> dict[str, Any]:
    """Normalize a document to a dict for tool output."""
    metadata = document.metadata or {}
    return {
        "content": document.page_content,
        "content_type": metadata.get("content_type"),
        "source_file": metadata.get("source_file"),
        "chunk_index": metadata.get("chunk_index"),
        "parking_id": metadata.get("parking_id"),
    }
