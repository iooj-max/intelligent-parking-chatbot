"""Tests for retrieval module: deduplicate_documents and normalize_document."""

from __future__ import annotations

import pytest

from langchain_core.documents import Document

from parking_agent.retrieval import deduplicate_documents, normalize_document


def test_deduplicate_documents() -> None:
    """Deduplicate by identity and respect top_k and max_chunks_per_source_file."""
    docs = [
        Document(
            page_content="chunk 1",
            metadata={"parking_id": "p1", "source_file": "f1.md", "chunk_index": 0},
        ),
        Document(
            page_content="chunk 2",
            metadata={"parking_id": "p1", "source_file": "f1.md", "chunk_index": 1},
        ),
        Document(
            page_content="chunk 3",
            metadata={"parking_id": "p1", "source_file": "f2.md", "chunk_index": 0},
        ),
        Document(
            page_content="duplicate",
            metadata={"parking_id": "p1", "source_file": "f1.md", "chunk_index": 0},
        ),
    ]

    result = deduplicate_documents(
        docs,
        top_k=3,
        max_chunks_per_source_file=2,
    )

    assert len(result) == 3
    contents = [d.page_content for d in result]
    assert "chunk 1" in contents
    assert "chunk 2" in contents
    assert "chunk 3" in contents
    assert "duplicate" not in contents


def test_normalize_document() -> None:
    """normalize_document returns dict with content and metadata fields."""
    doc = Document(
        page_content="Parking is available 24/7.",
        metadata={
            "content_type": "general_info",
            "source_file": "general_info.md",
            "chunk_index": 0,
            "parking_id": "airport_parking",
        },
    )

    result = normalize_document(doc)

    assert result["content"] == "Parking is available 24/7."
    assert result["content_type"] == "general_info"
    assert result["source_file"] == "general_info.md"
    assert result["chunk_index"] == 0
    assert result["parking_id"] == "airport_parking"


def test_normalize_document_empty_metadata() -> None:
    """normalize_document handles document with no metadata."""
    doc = Document(page_content="Some text", metadata={})

    result = normalize_document(doc)

    assert result["content"] == "Some text"
    assert result["content_type"] is None
    assert result["source_file"] is None
    assert result["chunk_index"] is None
    assert result["parking_id"] is None
