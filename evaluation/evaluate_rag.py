"""
RAG retrieval quality evaluation.

Metrics:
- Recall@k: Coverage of relevant documents
- Precision@k: Accuracy of retrieved documents
- MRR: Ranking quality
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluate RAG retrieval quality."""

    def __init__(self, retriever):
        self.retriever = retriever

    def calculate_recall_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int = 5,
    ) -> float:
        if not relevant_ids:
            return 0.0

        top_k = retrieved_ids[:k]
        relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        return relevant_retrieved / len(relevant_ids)

    def calculate_precision_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int = 5,
    ) -> float:
        if k <= 0:
            return 0.0

        top_k = retrieved_ids[:k]
        if not top_k:
            return 0.0

        relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        return relevant_retrieved / len(top_k)

    def calculate_mrr(self, retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                return 1.0 / rank
        return 0.0

    def evaluate_query(self, query: str, relevant_ids: List[str], k: int = 5) -> Dict[str, Any]:
        result = self.retriever.retrieve(query=query, return_format="structured")
        retrieved_ids = self._extract_doc_ids(result)

        recall = self.calculate_recall_at_k(retrieved_ids, relevant_ids, k)
        precision = self.calculate_precision_at_k(retrieved_ids, relevant_ids, k)
        mrr = self.calculate_mrr(retrieved_ids, relevant_ids)

        return {
            "query": query,
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_ids,
            f"recall@{k}": recall,
            f"precision@{k}": precision,
            "mrr": mrr,
        }

    def _extract_doc_ids(self, result) -> List[str]:
        """Extract document IDs from structured retrieval output in rank order."""
        doc_ids: List[str] = []

        for chunk in result.static_chunks:
            source_file = chunk.get("source_file")
            if source_file:
                doc_ids.append(source_file)

        # include inferred parking_id as a facility-level relevance signal
        if result.parking_id:
            doc_ids.append(result.parking_id)

        # dedupe while preserving order
        seen = set()
        ordered_unique = []
        for item in doc_ids:
            if item not in seen:
                seen.add(item)
                ordered_unique.append(item)

        return ordered_unique

    def evaluate_dataset(self, test_cases: List[Dict[str, Any]], k: int = 5) -> Dict[str, Any]:
        per_query_results = []

        for test_case in test_cases:
            result = self.evaluate_query(test_case["query"], test_case["relevant_ids"], k)
            per_query_results.append(result)

        num_queries = len(per_query_results)
        avg_recall = sum(r[f"recall@{k}"] for r in per_query_results) / num_queries
        avg_precision = sum(r[f"precision@{k}"] for r in per_query_results) / num_queries
        avg_mrr = sum(r["mrr"] for r in per_query_results) / num_queries

        return {
            "num_queries": num_queries,
            f"avg_recall@{k}": avg_recall,
            f"avg_precision@{k}": avg_precision,
            "avg_mrr": avg_mrr,
            "per_query_results": per_query_results,
        }

    def save_results(self, results: Dict[str, Any], output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"rag_metrics_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info("Saved RAG evaluation results to %s", output_file)


def _load_cases(dataset: Path) -> List[Dict[str, Any]]:
    with open(dataset) as f:
        return json.load(f)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality")
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/datasets/rag_test_cases.json"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))

    args = parser.parse_args()

    test_cases = _load_cases(args.dataset)

    retriever = None
    try:
        from src.rag.tools import get_retriever

        retriever = get_retriever()
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize retriever. Ensure OPENAI_API_KEY is set and backing services are running."
        ) from exc

    evaluator = RAGEvaluator(retriever)
    results = evaluator.evaluate_dataset(test_cases, k=args.k)

    print("\n=== RAG Evaluation Results ===")
    print(f"Number of queries: {results['num_queries']}")
    print(f"Avg Recall@{args.k}: {results[f'avg_recall@{args.k}']:.3f}")
    print(f"Avg Precision@{args.k}: {results[f'avg_precision@{args.k}']:.3f}")
    print(f"Avg MRR: {results['avg_mrr']:.3f}")

    evaluator.save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
