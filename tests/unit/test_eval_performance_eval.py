"""Tests for performance evaluation helpers."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from parking_agent.eval import performance_eval as performance_eval_mod


def test_percentile_interpolates_between_values() -> None:
    """Percentile helper computes expected interpolated values."""
    values = [10.0, 20.0, 30.0, 40.0]
    assert performance_eval_mod._percentile(values, 0.50) == 25.0
    assert performance_eval_mod._percentile(values, 0.95) == 38.5


def test_extract_latest_ai_text_supports_dict_and_message() -> None:
    """Latest assistant text is extracted from mixed message formats."""
    messages = [
        {"role": "assistant", "content": "first"},
        AIMessage(content="second"),
    ]
    assert performance_eval_mod._extract_latest_ai_text(messages) == "second"


def test_summarize_samples_reports_latency_and_error_rate() -> None:
    """Summary contains basic latency stats and error ratio."""
    samples = [
        {"latency_ms": 100.0, "status": "ok"},
        {"latency_ms": 300.0, "status": "error"},
    ]
    result = performance_eval_mod._summarize_samples(samples)
    assert result["sample_count"] == 2
    assert result["error_count"] == 1
    assert result["error_rate"] == 0.5
    assert result["latency_ms"]["p50"] == 200.0
