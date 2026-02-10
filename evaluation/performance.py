"""
Performance benchmarking.

Metrics:
- Latency: Response time (p50, p95, p99)
- Throughput: Queries per second
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import HumanMessage

from src.chatbot.graph import graph
from src.chatbot.state import ChatbotState

logger = logging.getLogger(__name__)


class PerformanceTester:
    """
    Benchmark system performance.
    """

    def measure_latency(self, query: str) -> Dict[str, float]:
        """
        Measure latency for a single query.

        Returns:
            {
                'total_time': float (seconds),
            }
        """
        start = time.time()

        state: ChatbotState = {
            'messages': [HumanMessage(content=query)],
            'mode': 'info',
            'intent': None,
            'context': None,
            'reservation': {'completed_fields': [], 'validation_errors': {}},
            'error': None,
            'iteration_count': 0,
        }

        result = graph.invoke(state)

        total_time = time.time() - start

        return {
            'total_time': total_time,
        }

    def run_latency_benchmark(
        self,
        queries: List[str],
        num_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Run latency benchmark on multiple queries.

        Args:
            queries: List of test queries
            num_iterations: Number of times to run each query

        Returns:
            {
                'num_queries': int,
                'num_iterations': int,
                'latencies': List[float],
                'p50': float,
                'p95': float,
                'p99': float,
                'avg': float,
            }
        """
        latencies = []

        for query in queries:
            for _ in range(num_iterations):
                result = self.measure_latency(query)
                latencies.append(result['total_time'])

        # Calculate percentiles
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)

        p50 = latencies_sorted[int(n * 0.50)]
        p95 = latencies_sorted[int(n * 0.95)]
        p99 = latencies_sorted[int(n * 0.99)]
        avg = sum(latencies) / n

        return {
            'num_queries': len(queries),
            'num_iterations': num_iterations,
            'total_requests': n,
            'latencies': latencies,
            'p50': p50,
            'p95': p95,
            'p99': p99,
            'avg': avg,
        }

    def run_throughput_benchmark(
        self,
        queries: List[str],
        num_concurrent: int = 10
    ) -> Dict[str, Any]:
        """
        Run throughput benchmark with concurrent requests.

        Args:
            queries: List of test queries
            num_concurrent: Number of concurrent workers

        Returns:
            {
                'num_queries': int,
                'num_concurrent': int,
                'total_time': float,
                'qps': float (queries per second),
            }
        """
        start = time.time()

        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(self.measure_latency, q) for q in queries]
            for future in futures:
                future.result()  # Wait for completion

        total_time = time.time() - start
        qps = len(queries) / total_time

        return {
            'num_queries': len(queries),
            'num_concurrent': num_concurrent,
            'total_time': total_time,
            'qps': qps,
        }

    def save_results(self, results: Dict[str, Any], output_dir: Path):
        """Save benchmark results to JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"performance_{timestamp}.json"

        # Remove latencies list for cleaner output (can be large)
        results_to_save = {k: v for k, v in results.items() if k != 'latencies'}

        with open(output_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)

        logger.info(f"Saved performance results to {output_file}")


def main():
    """CLI entry point for performance benchmarks."""
    import argparse

    parser = argparse.ArgumentParser(description="Run performance benchmarks")
    parser.add_argument(
        '--dataset',
        type=Path,
        default=Path('evaluation/datasets/answer_test_cases.json'),
        help='Path to test dataset (uses questions)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=10,
        help='Number of iterations per query for latency test'
    )
    parser.add_argument(
        '--concurrent',
        type=int,
        default=10,
        help='Number of concurrent workers for throughput test'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('evaluation/results'),
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Load test queries
    with open(args.dataset) as f:
        test_cases = json.load(f)
    queries = [tc['question'] for tc in test_cases]

    # Initialize tester
    tester = PerformanceTester()

    # Run latency benchmark
    logger.info("Running latency benchmark...")
    latency_results = tester.run_latency_benchmark(queries, args.iterations)
    print(f"\n=== Latency Results ===")
    print(f"Total requests: {latency_results['total_requests']}")
    print(f"p50: {latency_results['p50']:.3f}s")
    print(f"p95: {latency_results['p95']:.3f}s")
    print(f"p99: {latency_results['p99']:.3f}s")
    print(f"avg: {latency_results['avg']:.3f}s")

    # Run throughput benchmark
    logger.info("Running throughput benchmark...")
    throughput_results = tester.run_throughput_benchmark(queries, args.concurrent)
    print(f"\n=== Throughput Results ===")
    print(f"QPS: {throughput_results['qps']:.2f}")
    print(f"Total time: {throughput_results['total_time']:.2f}s")

    # Save results
    results = {
        'latency': latency_results,
        'throughput': throughput_results,
    }
    tester.save_results(results, args.output_dir)


if __name__ == '__main__':
    main()
