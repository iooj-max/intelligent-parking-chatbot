"""Tests for package-level exports."""

from __future__ import annotations

import parking_agent
import parking_agent.eval as parking_eval
import parking_agent.utils as parking_utils


def test_parking_agent_exports_graph_symbol() -> None:
    """Top-level package exports graph in __all__."""
    assert "graph" in parking_agent.__all__
    assert hasattr(parking_agent, "graph")


def test_parking_agent_module_docstring_is_present() -> None:
    """Top-level package docstring exists."""
    assert isinstance(parking_agent.__doc__, str)
    assert "Parking assistant" in parking_agent.__doc__


def test_utils_package_exports_message_content_to_text() -> None:
    """Utilities package re-exports message helper."""
    assert "message_content_to_text" in parking_utils.__all__
    assert callable(parking_utils.message_content_to_text)


def test_utils_package_docstring_is_present() -> None:
    """Utilities package docstring exists."""
    assert isinstance(parking_utils.__doc__, str)
    assert "utilities" in parking_utils.__doc__.lower()


def test_eval_package_docstring_mentions_evaluation() -> None:
    """Eval package docstring describes evaluation utilities."""
    assert isinstance(parking_eval.__doc__, str)
    assert "Evaluation" in parking_eval.__doc__


def test_eval_package_has_no_explicit_exports() -> None:
    """Eval package should not force __all__ exports."""
    assert not hasattr(parking_eval, "__all__")
