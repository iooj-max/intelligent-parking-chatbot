"""Tests for config module: Settings and postgres_dsn."""

from __future__ import annotations

import pytest

from src.config import Settings


def test_postgres_dsn_format(monkeypatch: pytest.MonkeyPatch) -> None:
    """postgres_dsn contains expected URI components."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("POSTGRES_USER", "testuser")
    monkeypatch.setenv("POSTGRES_PASSWORD", "testpass")
    monkeypatch.setenv("POSTGRES_HOST", "db.example.com")
    monkeypatch.setenv("POSTGRES_PORT", "5433")
    monkeypatch.setenv("POSTGRES_DB", "testdb")

    settings = Settings()
    dsn = settings.postgres_dsn

    assert "postgresql+psycopg://" in dsn
    assert "testuser" in dsn
    assert "testpass" in dsn
    assert "db.example.com" in dsn
    assert "5433" in dsn
    assert "testdb" in dsn


def test_settings_has_postgres_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings has expected postgres attributes and postgres_dsn is well-formed."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_DB", "parking")
    monkeypatch.setenv("POSTGRES_USER", "u")
    monkeypatch.setenv("POSTGRES_PASSWORD", "p")

    settings = Settings()

    assert hasattr(settings, "postgres_host")
    assert hasattr(settings, "postgres_port")
    assert hasattr(settings, "postgres_dsn")
    assert settings.postgres_dsn.startswith("postgresql+psycopg://")
    assert "/parking" in settings.postgres_dsn
