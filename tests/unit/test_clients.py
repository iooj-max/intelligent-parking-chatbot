"""Tests for clients module: build_postgres_uri and build_weaviate_client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from parking_agent.clients import build_postgres_uri, build_weaviate_client


def test_build_postgres_uri() -> None:
    """build_postgres_uri returns settings.postgres_dsn."""
    expected = "postgresql+psycopg://user:pass@host:5432/db"
    with patch("parking_agent.clients.settings") as mock_settings:
        mock_settings.postgres_dsn = expected
        assert build_postgres_uri() == expected


def test_build_weaviate_client() -> None:
    """build_weaviate_client calls weaviate.connect_to_custom with settings."""
    mock_client = MagicMock()
    with patch("parking_agent.clients.weaviate.connect_to_custom", return_value=mock_client) as mock_connect:
        with patch("parking_agent.clients.settings") as mock_settings:
            mock_settings.weaviate_http_host = "weaviate.example.com"
            mock_settings.weaviate_http_port = 8080
            mock_settings.weaviate_http_secure = False
            mock_settings.weaviate_grpc_host = "weaviate.example.com"
            mock_settings.weaviate_grpc_port = 50051
            mock_settings.weaviate_grpc_secure = False

            result = build_weaviate_client()

            mock_connect.assert_called_once()
            call_kwargs = mock_connect.call_args[1]
            assert call_kwargs["http_host"] == "weaviate.example.com"
            assert call_kwargs["http_port"] == 8080
            assert call_kwargs["grpc_port"] == 50051
            assert result is mock_client
