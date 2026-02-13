"""
Weaviate vector store for static parking content.

This module provides a Weaviate client wrapper for managing the ParkingContent
collection, including schema creation, batch insertion with embeddings, and
vector similarity search with metadata filtering.

MVP only - Not production-ready implementation.
"""

from typing import Any, Dict, List, Optional

import weaviate
from weaviate.classes.config import Configure, DataType, Property, VectorDistances
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.util import generate_uuid5

from ..config import settings


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
            url: Weaviate URL. If None, uses settings.weaviate_url
        """
        self.url = url or settings.weaviate_url
        self.client = weaviate.connect_to_local(host=self.url.replace("http://", "").replace(":8080", ""))

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
        return self.client.collections.exists(self.COLLECTION_NAME)

    def create_collection(self) -> None:
        """
        Create ParkingContent collection with schema.

        Idempotent - safe to call if collection already exists.
        """
        if self.collection_exists():
            return

        self.client.collections.create(
            name=self.COLLECTION_NAME,
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

        self.client.collections.delete(self.COLLECTION_NAME)
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

        collection = self.client.collections.get(self.COLLECTION_NAME)

        # Delete all objects matching parking_id
        result = collection.data.delete_many(where=Filter.by_property("parking_id").equal(parking_id))

        return result.successful if result.successful else 0

    def batch_insert_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Batch insert chunks with embeddings.

        Each chunk should have:
        - parking_id: str
        - content: str
        - content_type: str
        - source_file: str
        - chunk_index: int
        - metadata: dict with parking_name, address, city, coordinates
        - vector: list of floats (embedding)

        Args:
            chunks: List of chunk dictionaries with embeddings

        Returns:
            Number of objects inserted

        Raises:
            ValueError: If collection doesn't exist or chunks are invalid
        """
        if not self.collection_exists():
            raise ValueError(f"Collection {self.COLLECTION_NAME} does not exist. Call create_collection() first.")

        if not chunks:
            return 0

        collection = self.client.collections.get(self.COLLECTION_NAME)

        # Insert objects in batch
        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                # Generate deterministic UUID based on parking_id, source_file, and chunk_index
                # This ensures idempotency - same content gets same UUID
                uuid_key = f"{chunk['parking_id']}_{chunk['source_file']}_{chunk['chunk_index']}"
                uuid = generate_uuid5(uuid_key)

                batch.add_object(
                    properties={
                        "parking_id": chunk["parking_id"],
                        "content": chunk["content"],
                        "content_type": chunk["content_type"],
                        "source_file": chunk["source_file"],
                        "chunk_index": chunk["chunk_index"],
                        "metadata": chunk.get("metadata", {}),
                    },
                    vector=chunk["vector"],
                    uuid=uuid,
                )

        # Check for failed objects
        if collection.batch.failed_objects:
            failed_count = len(collection.batch.failed_objects)
            raise ValueError(f"Failed to insert {failed_count} objects: {collection.batch.failed_objects[:5]}")

        return len(chunks)

    def search_similar(
        self,
        query_vector: List[float],
        parking_id: Optional[str] = None,
        content_type: Optional[str] = None,
        limit: int = 5,
        return_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Vector similarity search with optional filters.

        Args:
            query_vector: Query embedding vector
            parking_id: Filter by parking facility (optional)
            content_type: Filter by content type (optional)
            limit: Maximum number of results
            return_metadata: Include metadata in results

        Returns:
            List of search results with content, metadata, and distance scores
        """
        if not self.collection_exists():
            return []

        collection = self.client.collections.get(self.COLLECTION_NAME)

        # Build filter
        filters = []
        if parking_id:
            filters.append(Filter.by_property("parking_id").equal(parking_id))
        if content_type:
            filters.append(Filter.by_property("content_type").equal(content_type))

        # Combine filters with AND
        where_filter = None
        if filters:
            where_filter = filters[0]
            for f in filters[1:]:
                where_filter = where_filter & f

        # Execute search
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=limit,
            where=where_filter,
            return_metadata=MetadataQuery(distance=True) if return_metadata else None,
        )

        # Format results
        results = []
        for obj in response.objects:
            result = {
                "parking_id": obj.properties.get("parking_id"),
                "content": obj.properties.get("content"),
                "content_type": obj.properties.get("content_type"),
                "source_file": obj.properties.get("source_file"),
                "chunk_index": obj.properties.get("chunk_index"),
            }

            if return_metadata:
                result["metadata"] = obj.properties.get("metadata", {})
                result["distance"] = obj.metadata.distance if obj.metadata else None

            results.append(result)

        return results

    def get_by_parking_id(self, parking_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific parking facility.

        Args:
            parking_id: Facility identifier
            limit: Maximum number of results

        Returns:
            List of chunks ordered by source_file and chunk_index
        """
        if not self.collection_exists():
            return []

        collection = self.client.collections.get(self.COLLECTION_NAME)

        response = collection.query.fetch_objects(
            where=Filter.by_property("parking_id").equal(parking_id),
            limit=limit,
        )

        results = []
        for obj in response.objects:
            results.append(
                {
                    "parking_id": obj.properties.get("parking_id"),
                    "content": obj.properties.get("content"),
                    "content_type": obj.properties.get("content_type"),
                    "source_file": obj.properties.get("source_file"),
                    "chunk_index": obj.properties.get("chunk_index"),
                    "metadata": obj.properties.get("metadata", {}),
                }
            )

        # Sort by source_file and chunk_index for consistent ordering
        results.sort(key=lambda x: (x["source_file"], x["chunk_index"]))

        return results

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

        collection = self.client.collections.get(self.COLLECTION_NAME)

        if parking_id:
            response = collection.aggregate.over_all(
                where=Filter.by_property("parking_id").equal(parking_id),
            )
        else:
            response = collection.aggregate.over_all()

        return response.total_count
