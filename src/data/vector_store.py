"""
Weaviate vector store for static parking content.

This module provides a Weaviate client wrapper for managing the ParkingContent
collection, including schema creation, batch insertion with embeddings, and
vector similarity search with metadata filtering.

⚠️ MVP only!
"""

from typing import Any, Dict, List, Optional

import weaviate
from weaviate.classes.config import Configure, DataType, Property, VectorDistances
from weaviate.classes.query import Filter

from parking_agent.clients import build_weaviate_client
from src.config import settings

class WeaviateStore:
    """
    Weaviate vector store for static parking content.

    Manages the ParkingContent collection with chunked markdown content,
    embeddings, and metadata for RAG retrieval.

    MVP only - Not production-ready implementation.
    """

    COLLECTION_NAME = "ParkingContent"

    def __init__(self, url: Optional[str] = None):
        """
        Initialize WeaviateStore with connection.

        Args:
            url: Deprecated. Connection details are read from settings.
        """
        _ = url
        self.collection_name = settings.weaviate_collection or self.COLLECTION_NAME
        self.client = build_weaviate_client()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close connection."""
        self.close()

    def close(self):
        """Close Weaviate connection."""
        if self.client:
            self.client.close()

    def __del__(self):
        """Destructor to ensure connection is closed."""
        try:
            if self.client and hasattr(self.client, 'close'):
                self.client.close()
        except Exception:
            pass  # Suppress errors during cleanup

    def collection_exists(self) -> bool:
        """
        Check if ParkingContent collection exists.

        Returns:
            True if collection exists, False otherwise
        """
        return self.client.collections.exists(self.collection_name)

    def create_collection(self) -> None:
        """
        Create ParkingContent collection with schema.

        Idempotent - safe to call if collection already exists.
        """
        if self.collection_exists():
            return

        self.client.collections.create(
            name=self.collection_name,
            description="Chunked parking information for RAG retrieval",
            vectorizer_config=Configure.Vectorizer.none(),  # Using external embeddings
            properties=[
                Property(
                    name="parking_id",
                    data_type=DataType.TEXT,
                    description="Unique identifier for the parking facility",
                    index_filterable=True,
                    index_searchable=False,
                ),
                Property(
                    name="content",
                    data_type=DataType.TEXT,
                    description="The actual text content chunk",
                    index_filterable=False,
                    index_searchable=True,
                ),
                Property(
                    name="content_type",
                    data_type=DataType.TEXT,
                    description="Type: general_info, features, location, booking_process, policies, faq",
                    index_filterable=True,
                    index_searchable=False,
                ),
                Property(
                    name="source_file",
                    data_type=DataType.TEXT,
                    description="Source markdown file name",
                    index_filterable=True,
                    index_searchable=False,
                ),
                Property(
                    name="chunk_index",
                    data_type=DataType.INT,
                    description="Order of this chunk within the source document",
                    index_filterable=True,
                    index_searchable=False,
                ),
                Property(
                    name="metadata",
                    data_type=DataType.OBJECT,
                    description="Additional structured metadata",
                    nested_properties=[
                        Property(name="parking_name", data_type=DataType.TEXT),
                        Property(name="address", data_type=DataType.TEXT),
                        Property(name="city", data_type=DataType.TEXT),
                        Property(
                            name="coordinates",
                            data_type=DataType.OBJECT,
                            nested_properties=[
                                Property(name="latitude", data_type=DataType.NUMBER),
                                Property(name="longitude", data_type=DataType.NUMBER),
                            ],
                        ),
                    ],
                ),
            ],
            vector_index_config=Configure.VectorIndex.hnsw(distance_metric=VectorDistances.COSINE),
        )

    def delete_collection(self) -> bool:
        """
        Delete ParkingContent collection.

        FOR TESTING ONLY - Destroys all data.

        Returns:
            True if collection was deleted, False if it didn't exist
        """
        if not self.collection_exists():
            return False

        self.client.collections.delete(self.collection_name)
        return True

    def delete_by_parking_id(self, parking_id: str) -> int:
        """
        Delete all objects for a specific parking facility.

        Args:
            parking_id: Facility identifier

        Returns:
            Number of objects deleted
        """
        if not self.collection_exists():
            return 0

        collection = self.client.collections.get(self.collection_name)

        # Delete all objects matching parking_id
        result = collection.data.delete_many(where=Filter.by_property("parking_id").equal(parking_id))

        return result.successful if result.successful else 0

    def count_objects(self, parking_id: Optional[str] = None) -> int:
        """
        Count objects in collection.

        Args:
            parking_id: Filter by parking facility (optional)

        Returns:
            Number of objects
        """
        if not self.collection_exists():
            return 0

        collection = self.client.collections.get(self.collection_name)

        if parking_id:
            response = collection.aggregate.over_all(
                where=Filter.by_property("parking_id").equal(parking_id),
            )
        else:
            response = collection.aggregate.over_all()

        return response.total_count
