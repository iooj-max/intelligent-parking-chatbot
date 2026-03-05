"""Client builders for Weaviate and PostgreSQL."""

from __future__ import annotations

from urllib.parse import urlparse

import weaviate

from src.config import settings


def _is_weaviate_cloud_url(url: str) -> bool:
    """Check if URL points to Weaviate Cloud."""
    return "weaviate.cloud" in url or "weaviate.network" in url


def _extract_weaviate_cloud_host(url: str) -> str:
    """Extract hostname from Weaviate URL (with or without scheme)."""
    s = url.strip()
    if not s.startswith(("http://", "https://")):
        s = "https://" + s
    return urlparse(s).netloc or urlparse(s).path or s


def build_weaviate_client() -> weaviate.WeaviateClient:
    """Build and return a Weaviate client."""
    if _is_weaviate_cloud_url(settings.weaviate_url) and settings.weaviate_api_key:
        cluster_host = _extract_weaviate_cloud_host(settings.weaviate_url)
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=cluster_host,
            auth_credentials=weaviate.classes.init.Auth.api_key(settings.weaviate_api_key),
        )
    auth = (
        weaviate.classes.init.Auth.api_key(settings.weaviate_api_key)
        if settings.weaviate_api_key
        else None
    )
    return weaviate.connect_to_custom(
        http_host=settings.weaviate_http_host,
        http_port=settings.weaviate_http_port,
        http_secure=settings.weaviate_http_secure,
        grpc_host=settings.weaviate_grpc_host or settings.weaviate_http_host,
        grpc_port=settings.weaviate_grpc_port,
        grpc_secure=settings.weaviate_grpc_secure,
        auth_credentials=auth,
    )


def build_postgres_uri() -> str:
    """Return the PostgreSQL connection URI."""
    return settings.postgres_dsn
