"""
Unit tests for evaluation infrastructure.
"""

import pytest
from evaluation.evaluate_rag import RAGEvaluator
from evaluation.evaluate_answers import AnswerEvaluator
from evaluation.performance import PerformanceTester


class TestRAGEvaluator:
    """Test RAG metrics calculation."""

    def test_recall_at_k_perfect(self):
        evaluator = RAGEvaluator(retriever=None)  # Don't need retriever for metric tests

        retrieved = ['doc1', 'doc2', 'doc3']
        relevant = ['doc1', 'doc2']

        recall = evaluator.calculate_recall_at_k(retrieved, relevant, k=3)
        assert recall == 1.0  # Both relevant docs retrieved

    def test_recall_at_k_partial(self):
        evaluator = RAGEvaluator(retriever=None)

        retrieved = ['doc1', 'doc3', 'doc4']
        relevant = ['doc1', 'doc2']

        recall = evaluator.calculate_recall_at_k(retrieved, relevant, k=3)
        assert recall == 0.5  # Only 1 of 2 relevant docs retrieved

    def test_precision_at_k(self):
        evaluator = RAGEvaluator(retriever=None)

        retrieved = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
        relevant = ['doc1', 'doc3']

        precision = evaluator.calculate_precision_at_k(retrieved, relevant, k=5)
        assert precision == 2 / 5  # 2 relevant out of 5 retrieved

    def test_mrr_first_position(self):
        evaluator = RAGEvaluator(retriever=None)

        retrieved = ['doc1', 'doc2', 'doc3']
        relevant = ['doc1']

        mrr = evaluator.calculate_mrr(retrieved, relevant)
        assert mrr == 1.0  # First position

    def test_mrr_second_position(self):
        evaluator = RAGEvaluator(retriever=None)

        retrieved = ['doc1', 'doc2', 'doc3']
        relevant = ['doc2']

        mrr = evaluator.calculate_mrr(retrieved, relevant)
        assert mrr == 0.5  # Second position (1/2)

    def test_mrr_no_relevant(self):
        evaluator = RAGEvaluator(retriever=None)

        retrieved = ['doc1', 'doc2', 'doc3']
        relevant = ['doc5']

        mrr = evaluator.calculate_mrr(retrieved, relevant)
        assert mrr == 0.0  # No relevant doc found


class TestAnswerEvaluator:
    """Test answer quality evaluation."""

    def test_evaluator_initialization(self):
        evaluator = AnswerEvaluator()
        assert evaluator.metrics is not None
        assert len(evaluator.metrics) == 2  # faithfulness and answer_relevancy


class TestPerformanceTester:
    """Test performance benchmarking."""

    def test_latency_measurement(self):
        tester = PerformanceTester()
        result = tester.measure_latency("What are your hours?")

        assert 'total_time' in result
        assert result['total_time'] > 0

    def test_latency_benchmark(self):
        tester = PerformanceTester()
        queries = ["What are your hours?", "How much does it cost?"]

        results = tester.run_latency_benchmark(queries, num_iterations=2)

        assert results['num_queries'] == 2
        assert results['num_iterations'] == 2
        assert results['total_requests'] == 4
        assert 'p50' in results
        assert 'p95' in results
        assert 'p99' in results
        assert 'avg' in results
