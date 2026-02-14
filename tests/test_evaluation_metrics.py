from types import SimpleNamespace

from evaluation.evaluate_rag import RAGEvaluator
from evaluation.performance import PerformanceTester
from src.rag.retriever import ParkingRetriever, RetrievalResult


class DummyRetriever:
    def __init__(self, result):
        self._result = result

    def retrieve(self, query: str, return_format: str = "structured"):
        return self._result


def test_rag_evaluator_extracts_static_and_dynamic_ids():
    result = SimpleNamespace(
        static_chunks=[
            {"source_file": "data/static/downtown_plaza/location.md"},
            {"source_file": "data/static/downtown_plaza/location.md"},
            {"source_file": "data/static/downtown_plaza/features.md"},
        ],
        parking_id="downtown_plaza",
    )

    evaluator = RAGEvaluator(DummyRetriever(result))
    ids = evaluator._extract_doc_ids(result)

    assert ids == [
        "data/static/downtown_plaza/location.md",
        "data/static/downtown_plaza/features.md",
        "downtown_plaza",
    ]


def test_rag_evaluator_metrics_are_calculated_correctly():
    evaluator = RAGEvaluator(DummyRetriever(SimpleNamespace(static_chunks=[], parking_id=None)))

    retrieved = ["a", "b", "c", "d"]
    relevant = ["b", "e"]

    assert evaluator.calculate_recall_at_k(retrieved, relevant, k=3) == 0.5
    assert evaluator.calculate_precision_at_k(retrieved, relevant, k=3) == (1 / 3)
    assert evaluator.calculate_mrr(retrieved, relevant) == 0.5


def test_performance_tester_works_with_injected_invoker():
    def fake_invoke(_state):
        return {"ok": True}

    tester = PerformanceTester(invoke_fn=fake_invoke)
    latency = tester.run_latency_benchmark(["q1", "q2"], num_iterations=2)
    throughput = tester.run_throughput_benchmark(["q1", "q2", "q3"], num_concurrent=2)

    assert latency["total_requests"] == 4
    assert latency["p50"] >= 0
    assert throughput["num_queries"] == 3
    assert throughput["qps"] > 0


def test_rag_evaluator_handles_non_structured_retriever_output():
    evaluator = RAGEvaluator(DummyRetriever("no context"))
    result = evaluator.evaluate_query("q", relevant_ids=["doc"], k=5)

    assert result["retrieved_ids"] == []
    assert result["recall@5"] == 0.0
    assert result["precision@5"] == 0.0
    assert result["mrr"] == 0.0


def test_retriever_returns_structured_empty_result_when_requested():
    retriever = ParkingRetriever(
        vector_store=SimpleNamespace(),
        sql_store=SimpleNamespace(),
        embedding_generator=SimpleNamespace(),
    )
    retriever._infer_parking_id = lambda _query: None
    retriever._retrieve_static_content = lambda _query, _parking_id: []

    result = retriever.retrieve("Where is parking?", return_format="structured")

    assert isinstance(result, RetrievalResult)
    assert result.static_chunks == []
    assert result.metadata["num_static_chunks"] == 0
