"""
Markdown text chunking utilities for RAG retrieval.

This module provides functions to chunk markdown content into semantically
meaningful pieces for optimal vector embedding and retrieval.

⚠️ MVP only!
"""

import re
from pathlib import Path
from typing import List, Tuple


def extract_content_type_from_filename(filename: str) -> str:
    """
    Extract content type from markdown filename.

    Args:
        filename: File name (e.g., "general_info.md", "features.md")

    Returns:
        Content type string (e.g., "general_info", "features")

    Examples:
        >>> extract_content_type_from_filename("general_info.md")
        'general_info'
        >>> extract_content_type_from_filename("booking_process.md")
        'booking_process'
    """
    return Path(filename).stem


def chunk_by_heading(text: str, source_file: str) -> List[Tuple[str, int]]:
    """
    Split markdown by H2 headings (##) to preserve semantic boundaries.

    Each chunk includes the heading and content until the next heading.
    If no H2 headings are found, returns the entire text as one chunk.

    Args:
        text: Markdown text content
        source_file: Source file name for reference

    Returns:
        List of tuples (chunk_text, chunk_index)

    Examples:
        >>> text = "# Title\\n\\n## Section 1\\nContent 1\\n\\n## Section 2\\nContent 2"
        >>> chunks = chunk_by_heading(text, "test.md")
        >>> len(chunks)
        3
    """
    # Find all H2 headings with their positions
    h2_pattern = r"^## .+$"
    matches = list(re.finditer(h2_pattern, text, re.MULTILINE))

    if not matches:
        # No H2 headings found - return entire text as one chunk
        return [(text.strip(), 0)]

    chunks = []
    chunk_index = 0

    # Extract content before first H2 heading
    first_heading_pos = matches[0].start()
    if first_heading_pos > 0:
        intro_text = text[:first_heading_pos].strip()
        if intro_text:
            chunks.append((intro_text, chunk_index))
            chunk_index += 1

    # Extract content for each H2 section
    for i, match in enumerate(matches):
        start = match.start()

        # Find end position (start of next H2 or end of text)
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append((chunk_text, chunk_index))
            chunk_index += 1

    return chunks


def chunk_by_paragraphs(text: str, max_paragraphs: int = 3) -> List[Tuple[str, int]]:
    """
    Split text into chunks of N paragraphs.

    Useful as fallback when heading-based chunking produces very large chunks.

    Args:
        text: Text content
        max_paragraphs: Maximum paragraphs per chunk

    Returns:
        List of tuples (chunk_text, chunk_index)
    """
    # Split by double newlines (paragraph boundaries)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    for i in range(0, len(paragraphs), max_paragraphs):
        chunk_paragraphs = paragraphs[i : i + max_paragraphs]
        chunk_text = "\n\n".join(chunk_paragraphs)
        chunks.append((chunk_text, i // max_paragraphs))

    return chunks


def estimate_tokens(text: str) -> int:
    """
    Rough estimation of token count (approx 4 chars per token for English).

    Args:
        text: Text content

    Returns:
        Estimated token count

    Note:
        This is a very rough estimate. For production, use tiktoken library.
    """
    return len(text) // 4


def chunk_text_smart(
    text: str,
    source_file: str,
    max_tokens: int = 500,
    prefer_headings: bool = True,
) -> List[Tuple[str, int]]:
    """
    Smart chunking strategy that prefers headings but falls back to paragraphs.

    Args:
        text: Markdown text content
        source_file: Source file name
        max_tokens: Maximum tokens per chunk (approximate)
        prefer_headings: Try heading-based chunking first

    Returns:
        List of tuples (chunk_text, chunk_index)

    Strategy:
        1. Try heading-based chunking
        2. If any chunk exceeds max_tokens, split it by paragraphs
        3. Ensure no chunk is too large
    """
    if prefer_headings:
        chunks = chunk_by_heading(text, source_file)
    else:
        chunks = [(text, 0)]

    # Check if any chunk is too large
    final_chunks = []
    chunk_idx = 0

    for chunk_text, _ in chunks:
        tokens = estimate_tokens(chunk_text)

        if tokens <= max_tokens:
            # Chunk is good size
            final_chunks.append((chunk_text, chunk_idx))
            chunk_idx += 1
        else:
            # Chunk too large - split by paragraphs
            sub_chunks = chunk_by_paragraphs(chunk_text, max_paragraphs=2)
            for sub_text, _ in sub_chunks:
                final_chunks.append((sub_text, chunk_idx))
                chunk_idx += 1

    return final_chunks


def prepare_chunk_for_insertion(
    parking_id: str,
    source_file: str,
    chunk_text: str,
    chunk_index: int
) -> dict:
    """
    Prepare chunk dictionary for Weaviate insertion.

    Args:
        parking_id: Parking facility identifier
        source_file: Source markdown file name
        chunk_text: Text content of chunk
        chunk_index: Index of chunk in source document
        metadata: Additional metadata (parking_name, address, city, coordinates)

    Returns:
        Dictionary ready for Weaviate insertion (without vector)

    Note:
        Caller must add 'vector' field with embedding before insertion.
    """
    content_type = extract_content_type_from_filename(source_file)

    return {
        "parking_id": parking_id,
        "content": chunk_text,
        "content_type": content_type,
        "source_file": source_file,
        "chunk_index": chunk_index,
        "metadata": {},
    }
