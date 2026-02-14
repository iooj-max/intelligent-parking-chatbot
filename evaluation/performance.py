"""
Performance benchmarking.

Metrics:
- Latency: Response time (p50, p95, p99)
- Throughput: Queries per second
"""

import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import HumanMessage

from src.chatbot.state import ChatbotState

logger = logging.getLogger(__name__)


class PerformanceTester:
    """Benchmark chatbot performance."""

    def __init__(self, invoke_fn: Optional[Callable[[ChatbotState], Any]] = None):
        if invoke_fn is not None:
            self._invoke = invoke_fn
        else:
            from src.chatbot.graph import graph

            self._invoke = graph.invoke

    def measure_latency(self, query: str) -> Dict[str, float]:
        start = time.time()

        state: ChatbotState = {
            "messages": [HumanMessage(content=query)],
            "mode": "info",
            "intent": None,
            "context": None,
            "reservation": {"completed_fields": [], "validation_errors": {}},
            "error": None,
            "iteration_count": 0,
        }

        self._invoke(state)
        return {"total_time": time.time() - start}

    def run_latency_benchmark(self, queries: List[str], num_iterations: int = 10) -> Dict[str, Any]:
        latencies: List[float] = []

        for query in queries:
            for _ in range(num_iterations):
                latencies.append(self.measure_latency(query)["total_time"])

        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)

        def percentile(p: float) -> float:
            idx = min(math.ceil(p * n) - 1, n - 1)
            return latencies_sorted[max(idx, 0)]

        return {
            "num_queries": len(queries),
            "num_iterations": num_iterations,
            "total_requests": n,
            "latencies": latencies,
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "avg": sum(latencies) / n if n else 0.0,
        }

    def run_throughput_benchmark(self, queries: List[str], num_concurrent: int = 10) -> Dict[str, Any]:
        start = time.time()

        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(self.measure_latency, query) for query in queries]
            for future in futures:
                future.result()

        total_time = time.time() - start
        qps = (len(queries) / total_time) if total_time > 0 else 0.0

        return {
            "num_queries": len(queries),
            "num_concurrent": num_concurrent,
            "total_time": total_time,
            "qps": qps,
        }

    def save_results(self, results: Dict[str, Any], output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"performance_{timestamp}.json"

        results_to_save = {k: v for k, v in results.items() if k != "latencies"}
        with open(output_file, "w") as f:
            json.dump(results_to_save, f, indent=2)

        logger.info("Saved performance results to %s", output_file)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run performance benchmarks")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/datasets/answer_test_cases.json"),
        help="Path to test dataset (uses questions)",
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--concurrent", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))

    args = parser.parse_args()

    with open(args.dataset) as f:
        test_cases = json.load(f)
    queries = [tc["question"] for tc in test_cases]

    tester = PerformanceTester()

    logger.info("Running latency benchmark...")
    latency_results = tester.run_latency_benchmark(queries, args.iterations)
    print("\n=== Latency Results ===")
    print(f"Total requests: {latency_results['total_requests']}")
    print(f"p50: {latency_results['p50']:.3f}s")
    print(f"p95: {latency_results['p95']:.3f}s")
    print(f"p99: {latency_results['p99']:.3f}s")
    print(f"avg: {latency_results['avg']:.3f}s")

    logger.info("Running throughput benchmark...")
    throughput_results = tester.run_throughput_benchmark(queries, args.concurrent)
    print("\n=== Throughput Results ===")
    print(f"Total queries: {throughput_results['num_queries']}")
    print(f"Concurrent workers: {throughput_results['num_concurrent']}")
    print(f"Total time: {throughput_results['total_time']:.3f}s")
    print(f"QPS: {throughput_results['qps']:.2f}")

    combined_results = {"latency": latency_results, "throughput": throughput_results}
    tester.save_results(combined_results, args.output_dir)


if __name__ == "__main__":
    main()
