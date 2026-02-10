"""
Evaluation infrastructure for parking chatbot.

Provides:
- RAG retrieval quality metrics (recall, precision, MRR)
- Answer quality metrics (faithfulness, relevance via RAGAS)
- Performance benchmarks (latency, throughput)
"""

from .evaluate_rag import RAGEvaluator
from .evaluate_answers import AnswerEvaluator
from .performance import PerformanceTester

__all__ = [
    "RAGEvaluator",
    "AnswerEvaluator",
    "PerformanceTester",
]
