"""Single-user latency evaluation for unified graph responses."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langsmith import traceable, tracing_context
from langsmith.run_helpers import get_current_run_tree

from parking_agent.eval.retrieval_eval import DEFAULT_REPORT_DIR, _load_dataset
from parking_agent.graph import build_graph
from parking_agent.utils import message_content_to_text

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STATIC_DATASET = PROJECT_ROOT / "eval" / "static_retrieval_eval_dataset.jsonl"
DEFAULT_DYNAMIC_DATASET = PROJECT_ROOT / "eval" / "dynamic_retrieval_eval_dataset.jsonl"

try:
    from pydantic.warnings import PydanticDeprecatedSince20

    warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)
except Exception:
    warnings.filterwarnings(
        "ignore",
        message=r".*The `dict` method is deprecated; use `model_dump` instead.*",
    )

# LangChain/LangGraph + Pydantic v2 can emit noisy serializer warnings
# for structured outputs during eval runs.
warnings.filterwarnings(
    "ignore",
    message=r"^Pydantic serializer warnings:.*",
    category=UserWarning,
    module=r"pydantic\.main",
)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * percentile
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return sorted_values[low]
    low_weight = high - rank
    high_weight = rank - low
    return sorted_values[low] * low_weight + sorted_values[high] * high_weight


@traceable(name="performance_eval_call", run_type="chain")
def _timed_call(
    query: str,
    dataset_type: str,
    app: Any,
    config: RunnableConfig,
) -> dict[str, Any]:
    error_type = None
    error_message = None
    response_text = ""
    start = time.perf_counter()
    try:
        result = app.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config,
        )
        response_text = _extract_latest_ai_text(result.get("messages", []))
        status = "ok" if response_text else "error"
        if status != "ok":
            error_type = "empty_response"
            error_message = "Graph returned no assistant response."
    except Exception as exc:
        status = "error"
        error_type = type(exc).__name__
        error_message = str(exc)
    latency_ms = (time.perf_counter() - start) * 1000
    current_run = get_current_run_tree()
    sample = {
        "query": query,
        "dataset_type": dataset_type,
        "latency_ms": latency_ms,
        "status": status,
        "response_preview": response_text[:300],
        "run_id": str(getattr(current_run, "id", "")) if current_run is not None else None,
        "trace_id": str(getattr(current_run, "trace_id", "")) if current_run is not None else None,
    }
    if error_type is not None:
        sample["error_type"] = error_type
    if error_message is not None:
        sample["error_message"] = error_message
    return sample


def _extract_latest_ai_text(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""

    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = message_content_to_text(message.content)
            if text:
                return text
            continue
        if isinstance(message, BaseMessage):
            text = message_content_to_text(message.content)
            if text:
                return text
            continue
        if isinstance(message, dict):
            role = str(message.get("role", "")).strip().lower()
            if role not in {"assistant", "ai"}:
                continue
            text = message_content_to_text(message.get("content"))
            if text:
                return text
    return ""


def _summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [sample["latency_ms"] for sample in samples]
    errors = sum(1 for sample in samples if sample.get("status") != "ok")
    count = len(samples)
    return {
        "sample_count": count,
        "error_count": errors,
        "error_rate": (errors / count) if count else 0.0,
        "latency_ms": {
            "min": min(latencies) if latencies else 0.0,
            "max": max(latencies) if latencies else 0.0,
            "mean": (sum(latencies) / count) if count else 0.0,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
        },
    }


def _print_progress(current: int, total: int) -> None:
    if total <= 0:
        return
    bar_width = 30
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(bar_width * ratio)
    bar = "#" * filled + "-" * (bar_width - filled)
    percent = ratio * 100
    print(
        f"\rProgress: [{bar}] {current}/{total} ({percent:5.1f}%)",
        end="",
        flush=True,
    )
    if current >= total:
        print()


@traceable(name="performance_eval_run", run_type="chain")
def run_performance_evaluation(
    static_dataset_path: Path,
    dynamic_dataset_path: Path,
    report_dir: Path,
    *,
    repeats: int,
    query_limit: int | None,
    inter_call_delay_ms: int,
    show_progress: bool,
) -> Path:
    static_records = _load_dataset(static_dataset_path)
    dynamic_records = _load_dataset(dynamic_dataset_path)

    records: list[dict[str, str]] = [
        {"query": record["query"], "dataset_type": "static"} for record in static_records
    ] + [{"query": record["query"], "dataset_type": "dynamic"} for record in dynamic_records]
    if query_limit is not None and query_limit > 0:
        records = records[:query_limit]

    samples: list[dict[str, Any]] = []
    app = build_graph(checkpointer=None)
    delay_seconds = max(0.0, inter_call_delay_ms / 1000)
    total_calls = repeats * len(records)
    completed_calls = 0
    progress_enabled = show_progress and sys.stdout.isatty() and total_calls > 0
    if progress_enabled:
        _print_progress(0, total_calls)

    interrupted = False
    try:
        for repeat_index in range(repeats):
            for record_index, record in enumerate(records, start=1):
                thread_id = f"perf-eval-{repeat_index + 1}-{record_index}-{uuid4().hex}"
                invoke_config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
                sample = _timed_call(
                    query=record["query"],
                    dataset_type=record["dataset_type"],
                    app=app,
                    config=invoke_config,
                )
                sample["repeat"] = repeat_index + 1
                samples.append(sample)
                completed_calls += 1
                if progress_enabled:
                    _print_progress(completed_calls, total_calls)
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
    except KeyboardInterrupt:
        interrupted = True
        if progress_enabled:
            print()
        print("Interrupted by user. Saving partial report...")

    timestamp = datetime.now(timezone.utc)
    report_payload: dict[str, Any] = {
        "meta": {
            "generated_at_utc": timestamp.isoformat(),
            "static_dataset_path": str(static_dataset_path),
            "dynamic_dataset_path": str(dynamic_dataset_path),
            "query_count": len(records),
            "static_query_count": len(static_records),
            "dynamic_query_count": len(dynamic_records),
            "repeats": repeats,
            "inter_call_delay_ms": inter_call_delay_ms,
            "total_calls": len(samples),
            "planned_total_calls": total_calls,
            "interrupted": interrupted,
        },
        "agent_response": {
            "summary": _summarize_samples(samples),
            "samples": samples,
        },
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"perf_eval_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    output_path = report_dir / output_name
    output_path.write_text(
        json.dumps(report_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure single-user latency for full parking agent graph responses."
    )
    parser.add_argument("--static-dataset", default=str(DEFAULT_STATIC_DATASET))
    parser.add_argument("--dynamic-dataset", default=str(DEFAULT_DYNAMIC_DATASET))
    parser.add_argument("--output-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--inter-call-delay-ms",
        type=int,
        default=2000,
        help="Pause between consecutive graph calls to reduce rate-limit errors.",
    )
    parser.add_argument("--langsmith-project", default="intelligent-parking-chatbot")
    parser.add_argument("--max-p95-ms", type=float)
    parser.add_argument("--max-error-rate", type=float)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable interactive progress bar output.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    static_dataset_path = Path(args.static_dataset)
    dynamic_dataset_path = Path(args.dynamic_dataset)
    report_dir = Path(args.output_dir)
    repeats = max(1, int(args.repeats))
    query_limit = int(args.limit) if int(args.limit) > 0 else None
    inter_call_delay_ms = max(0, int(args.inter_call_delay_ms))
    show_progress = not bool(args.no_progress)

    with tracing_context(
        project_name=args.langsmith_project,
        tags=["evaluation", "performance", "single_user"],
        metadata={
            "eval_type": "single_user_latency",
            "static_dataset_path": str(static_dataset_path),
            "dynamic_dataset_path": str(dynamic_dataset_path),
            "repeats": repeats,
            "inter_call_delay_ms": inter_call_delay_ms,
        },
    ):
        report_path = run_performance_evaluation(
            static_dataset_path=static_dataset_path,
            dynamic_dataset_path=dynamic_dataset_path,
            report_dir=report_dir,
            repeats=repeats,
            query_limit=query_limit,
            inter_call_delay_ms=inter_call_delay_ms,
            show_progress=show_progress,
        )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    print(f"Saved performance evaluation report: {report_path}")
    summary = payload["agent_response"]["summary"]
    latency_stats = summary["latency_ms"]
    error_rate = summary["error_rate"]
    print(
        (
            "Unified agent p50/p95/p99 (ms): "
            f"{latency_stats['p50']:.2f}/{latency_stats['p95']:.2f}/{latency_stats['p99']:.2f}\n"
            f"Unified agent error rate: {error_rate:.4f}"
        )
    )

    failures: list[str] = []
    if args.max_p95_ms is not None and latency_stats["p95"] > args.max_p95_ms:
        failures.append(
            f"p95 {latency_stats['p95']:.2f}ms > required {args.max_p95_ms:.2f}ms"
        )
    if args.max_error_rate is not None and error_rate > args.max_error_rate:
        failures.append(
            f"error rate {error_rate:.4f} > required {args.max_error_rate:.4f}"
        )
    if failures:
        print("Threshold check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

