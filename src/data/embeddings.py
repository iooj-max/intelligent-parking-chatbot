"""
OpenAI embeddings generation utilities.

This module provides functions to generate embeddings for text chunks
using OpenAI's embedding models.

MVP only - Not production-ready implementation.
"""

import time
from typing import List

from langchain_openai import OpenAIEmbeddings

from ..config import settings


class EmbeddingGenerator:
    """
    Wrapper for OpenAI embeddings generation with batching and error handling.

    MVP only - Not production-ready implementation.
    """

    def __init__(self, model: str = "text-embedding-3-small", batch_size: int = 100):
        """
        Initialize embedding generator.

        Args:
            model: OpenAI embedding model name
            batch_size: Number of texts to embed in one batch
        """
        self.model = model
        self.batch_size = batch_size
        self.embeddings = OpenAIEmbeddings(
            model=model,
            openai_api_key=settings.openai_api_key,
        )

    def generate(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats

        Raises:
            Exception: If API call fails after retries
        """
        return self._generate_with_retry([text])[0]

    def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            Exception: If API call fails after retries
        """
        if not texts:
            return []

        all_embeddings = []

        # Process in batches to avoid rate limits
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = self._generate_with_retry(batch)
            all_embeddings.extend(batch_embeddings)

            # Small delay between batches to avoid rate limiting
            if i + self.batch_size < len(texts):
                time.sleep(0.1)

        return all_embeddings

    def _generate_with_retry(self, texts: List[str], max_retries: int = 3) -> List[List[float]]:
        """
        Generate embeddings with retry logic.

        Args:
            texts: List of texts to embed
            max_retries: Maximum number of retry attempts

        Returns:
            List of embedding vectors

        Raises:
            Exception: If all retries fail
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                # Use LangChain's embed_documents which handles batching
                embeddings = self.embeddings.embed_documents(texts)
                return embeddings
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Exponential backoff: wait 1s, 2s, 4s
                    wait_time = 2**attempt
                    print(f"Embedding API error (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)

        # All retries failed
        raise Exception(f"Failed to generate embeddings after {max_retries} attempts: {last_error}")


def generate_embeddings(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """
    Convenience function to generate embeddings for a list of texts.

    Args:
        texts: List of texts to embed
        model: OpenAI embedding model name

    Returns:
        List of embedding vectors

    Examples:
        >>> texts = ["Hello world", "Goodbye world"]
        >>> embeddings = generate_embeddings(texts)
        >>> len(embeddings)
        2
        >>> len(embeddings[0]) # text-embedding-3-small has 1536 dimensions
        1536
    """
    generator = EmbeddingGenerator(model=model)
    return generator.generate_batch(texts)
