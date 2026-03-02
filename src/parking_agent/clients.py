"""Client builders for Weaviate and PostgreSQL."""

from __future__ import annotations

import weaviate

from src.config import settings


def build_weaviate_client() -> weaviate.WeaviateClient:
    """Build and return a Weaviate client."""
    return weaviate.connect_to_custom(
        http_host=settings.weaviate_http_host,
        http_port=settings.weaviate_http_port,
        http_secure=settings.weaviate_http_secure,
        grpc_host=settings.weaviate_grpc_host or settings.weaviate_http_host,
        grpc_port=settings.weaviate_grpc_port,
        grpc_secure=settings.weaviate_grpc_secure,
    )


def build_postgres_uri() -> str:
    """Return the PostgreSQL connection URI."""
    return settings.postgres_dsn
