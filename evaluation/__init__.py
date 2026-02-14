"""Evaluation infrastructure for parking chatbot."""

from importlib import import_module
from typing import Any

__all__ = ["RAGEvaluator", "AnswerEvaluator", "PerformanceTester"]


def __getattr__(name: str) -> Any:
    if name == "RAGEvaluator":
        return import_module(".evaluate_rag", __name__).RAGEvaluator
    if name == "AnswerEvaluator":
        return import_module(".evaluate_answers", __name__).AnswerEvaluator
    if name == "PerformanceTester":
        return import_module(".performance", __name__).PerformanceTester
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
