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

from src.rag.retriever import ParkingRetriever, RetrievalResult

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Evaluate RAG retrieval quality.
    """

    def __init__(self, retriever: ParkingRetriever):
        self.retriever = retriever

    def calculate_recall_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int = 5
    ) -> float:
        """
        Calculate recall@k.

        Args:
            retrieved_ids: List of retrieved document IDs (in rank order)
            relevant_ids: List of ground-truth relevant document IDs
            k: Number of top results to consider

        Returns:
            Recall score (0-1)
        """
        if not relevant_ids:
            return 0.0

        top_k = retrieved_ids[:k]
        relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant_ids)

        return relevant_retrieved / len(relevant_ids)

    def calculate_precision_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int = 5
    ) -> float:
        """
        Calculate precision@k.

        Args:
            retrieved_ids: List of retrieved document IDs (in rank order)
            relevant_ids: List of ground-truth relevant document IDs
            k: Number of top results to consider

        Returns:
            Precision score (0-1)
        """
        top_k = retrieved_ids[:k]
        relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant_ids)

        return relevant_retrieved / k if k > 0 else 0.0

    def calculate_mrr(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (for single query).

        Args:
            retrieved_ids: List of retrieved document IDs (in rank order)
            relevant_ids: List of ground-truth relevant document IDs

        Returns:
            Reciprocal rank (0-1)
        """
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                return 1.0 / rank

        return 0.0  # No relevant doc found

    def evaluate_query(
        self,
        query: str,
        relevant_ids: List[str],
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval for a single query.

        Args:
            query: User query
            relevant_ids: Ground-truth relevant document IDs
            k: Number of top results to consider

        Returns:
            {
                'query': str,
                'retrieved_ids': List[str],
                'relevant_ids': List[str],
                'recall@k': float,
                'precision@k': float,
                'mrr': float,
            }
        """
        # Retrieve documents
        result: RetrievalResult = self.retriever.retrieve(
            query=query,
            return_format="structured"
        )

        # Extract document IDs from result
        # For static content: use chunk metadata
        # For dynamic data: use facility IDs
        retrieved_ids = self._extract_doc_ids(result)

        # Calculate metrics
        recall = self.calculate_recall_at_k(retrieved_ids, relevant_ids, k)
        precision = self.calculate_precision_at_k(retrieved_ids, relevant_ids, k)
        mrr = self.calculate_mrr(retrieved_ids, relevant_ids)

        return {
            'query': query,
            'retrieved_ids': retrieved_ids,
            'relevant_ids': relevant_ids,
            f'recall@{k}': recall,
            f'precision@{k}': precision,
            'mrr': mrr,
        }

    def _extract_doc_ids(self, result: RetrievalResult) -> List[str]:
        """
        Extract document IDs from retrieval result.

        Returns list of IDs in rank order.
        """
        doc_ids = []

        # From static content (Weaviate chunks)
        for chunk in result.static_results:
            # Use source_file as doc ID
            if 'source_file' in chunk:
                doc_ids.append(chunk['source_file'])

        # From dynamic data (PostgreSQL facilities)
        for facility in result.dynamic_results:
            # Use facility name as doc ID
            if 'name' in facility:
                doc_ids.append(facility['name'])

        return doc_ids

    def evaluate_dataset(
        self,
        test_cases: List[Dict[str, Any]],
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval on entire test dataset.

        Args:
            test_cases: List of test cases [{'query': str, 'relevant_ids': List[str]}, ...]
            k: Number of top results to consider

        Returns:
            {
                'num_queries': int,
                'avg_recall@k': float,
                'avg_precision@k': float,
                'avg_mrr': float,
                'per_query_results': List[Dict],
            }
        """
        per_query_results = []

        for test_case in test_cases:
            query = test_case['query']
            relevant_ids = test_case['relevant_ids']

            result = self.evaluate_query(query, relevant_ids, k)
            per_query_results.append(result)

        # Calculate averages
        num_queries = len(per_query_results)
        avg_recall = sum(r[f'recall@{k}'] for r in per_query_results) / num_queries
        avg_precision = sum(r[f'precision@{k}'] for r in per_query_results) / num_queries
        avg_mrr = sum(r['mrr'] for r in per_query_results) / num_queries

        return {
            'num_queries': num_queries,
            f'avg_recall@{k}': avg_recall,
            f'avg_precision@{k}': avg_precision,
            'avg_mrr': avg_mrr,
            'per_query_results': per_query_results,
        }

    def save_results(self, results: Dict[str, Any], output_dir: Path):
        """Save evaluation results to JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"rag_metrics_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved RAG evaluation results to {output_file}")


def main():
    """CLI entry point for RAG evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality")
    parser.add_argument(
        '--dataset',
        type=Path,
        default=Path('evaluation/datasets/rag_test_cases.json'),
        help='Path to test dataset'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='Number of top results to consider'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('evaluation/results'),
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Load test cases
    with open(args.dataset) as f:
        test_cases = json.load(f)

    # Initialize retriever and evaluator
    retriever = ParkingRetriever()
    evaluator = RAGEvaluator(retriever)

    # Run evaluation
    logger.info(f"Evaluating {len(test_cases)} test cases...")
    results = evaluator.evaluate_dataset(test_cases, k=args.k)

    # Print summary
    print(f"\n=== RAG Evaluation Results ===")
    print(f"Number of queries: {results['num_queries']}")
    print(f"Avg Recall@{args.k}: {results[f'avg_recall@{args.k}']:.3f}")
    print(f"Avg Precision@{args.k}: {results[f'avg_precision@{args.k}']:.3f}")
    print(f"Avg MRR: {results['avg_mrr']:.3f}")

    # Save results
    evaluator.save_results(results, args.output_dir)


if __name__ == '__main__':
    main()
