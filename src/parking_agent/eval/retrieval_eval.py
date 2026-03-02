"""Offline retrieval evaluation for Recall@K and Precision@K."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langsmith import traceable, tracing_context
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate.vectorstores import WeaviateVectorStore
from pydantic import SecretStr

from parking_agent.clients import build_weaviate_client
from parking_agent.retrieval import build_weaviate_retriever, deduplicate_documents
from src.config import settings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STATIC_DATASET = PROJECT_ROOT / "eval" / "static_retrieval_eval_dataset.jsonl"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "eval" / "reports"
DEFAULT_COLLECTION = settings.weaviate_collection
DEFAULT_K = settings.weaviate_top_k
DEFAULT_ALPHA = settings.weaviate_query_alpha
DEFAULT_CANDIDATE_K = settings.weaviate_candidate_k
DEFAULT_MAX_CHUNKS_PER_SOURCE = settings.weaviate_max_chunks_per_source


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Dataset line {line_index} must be a JSON object.")
            query = payload.get("query")
            relevant_doc_ids = payload.get("relevant_doc_ids")
            parking_id = payload.get("parking_id")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"Dataset line {line_index} has invalid 'query'.")
            if (
                not isinstance(relevant_doc_ids, list)
                or not relevant_doc_ids
                or not all(isinstance(item, str) and item.strip() for item in relevant_doc_ids)
            ):
                raise ValueError(
                    f"Dataset line {line_index} must include non-empty 'relevant_doc_ids'."
                )
            parking_id_str = (parking_id or "").strip() if isinstance(parking_id, str) else ""
            records.append(
                {
                    "query": query.strip(),
                    "parking_id": parking_id_str,
                    "relevant_doc_ids": [item.strip() for item in relevant_doc_ids],
                    "notes": payload.get("notes", ""),
                }
            )
    if not records:
        raise ValueError("Dataset is empty.")
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8192)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_doc_id(parking_id: Any, source_file: Any, chunk_index: Any) -> str:
    parking = str(parking_id).strip() if parking_id is not None else "unknown_parking"
    source = str(source_file).strip() if source_file is not None else "unknown_source"
    base = f"{parking}::{source}"
    if chunk_index is None:
        return base
    if isinstance(chunk_index, int):
        return f"{base}#chunk-{chunk_index}"
    chunk_text = str(chunk_index).strip()
    if not chunk_text:
        return base
    return f"{base}#chunk-{chunk_text}"


def _extract_source_file(doc_id: str) -> str:
    without_chunk = doc_id.split("#chunk-", maxsplit=1)[0]
    if "::" in without_chunk:
        return without_chunk.split("::", maxsplit=1)[1]
    return without_chunk


def _relevant_pairs(parking_id: str, relevant_doc_ids: list[str]) -> set[tuple[str, str]]:
    """Build set of (parking_id, source_file) for relevant documents."""
    pid = parking_id.strip().lower() if parking_id else ""
    return {(pid, sf.strip()) for sf in relevant_doc_ids if sf.strip()}


def _retrieved_pairs(documents: list) -> list[tuple[str, str]]:
    """Build list of (parking_id, source_file) for retrieved documents, preserving order."""
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for doc in documents:
        meta = doc.metadata or {}
        pid = str(meta.get("parking_id", "")).strip().lower()
        sf = str(meta.get("source_file", "")).strip()
        key = (pid, sf)
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    return pairs


def _ordered_unique_pairs(documents: list) -> list[tuple[str, str]]:
    """Build ordered list of unique (parking_id, source_file) from documents."""
    return _retrieved_pairs(documents)


def _source_file_to_doc_type(source_file: str) -> str:
    """Extract doc_type from source_file (e.g. features.md -> features)."""
    return source_file.replace(".md", "").replace("_", " ") if source_file else ""


@traceable(name="retrieval_eval_query", run_type="chain")
def _evaluate_query(
    vector_store: WeaviateVectorStore,
    query: str,
    parking_id: str,
    relevant_doc_ids: list[str],
    k: int,
    alpha: float,
    candidate_k: int,
    max_chunks_per_source: int,
) -> dict[str, Any]:
    normalized_ids = [parking_id.strip().lower()] if parking_id else []
    retriever = build_weaviate_retriever(
        vector_store,
        k=max(k, candidate_k),
        alpha=alpha,
        parking_ids=normalized_ids,
    )
    candidate_documents = retriever.invoke(query)
    documents = deduplicate_documents(
        candidate_documents,
        top_k=k,
        max_chunks_per_source_file=max_chunks_per_source,
    )
    retrieved_chunk_ids: list[str] = []
    for document in documents:
        metadata = document.metadata or {}
        doc_id = _normalize_doc_id(
            parking_id=metadata.get("parking_id"),
            source_file=metadata.get("source_file"),
            chunk_index=metadata.get("chunk_index"),
        )
        retrieved_chunk_ids.append(doc_id)

    relevant_set = _relevant_pairs(parking_id, relevant_doc_ids)
    retrieved_pairs = _retrieved_pairs(documents)
    unique_retrieved_pairs = list(dict.fromkeys(retrieved_pairs))
    true_positive_pairs = sorted(relevant_set.intersection(unique_retrieved_pairs))
    precision_at_k = len(true_positive_pairs) / max(len(unique_retrieved_pairs), 1)
    recall_at_k = len(true_positive_pairs) / len(relevant_set) if relevant_set else 0.0
    hit_at_k = int(bool(true_positive_pairs))

    first_relevant_rank: int | None = None
    for idx, pair in enumerate(unique_retrieved_pairs):
        if pair in relevant_set:
            first_relevant_rank = idx + 1
            break
    reciprocal_rank = 1.0 / first_relevant_rank if first_relevant_rank else 0.0

    all_candidate_pairs = _ordered_unique_pairs(candidate_documents)
    first_relevant_rank_in_candidates: int | None = None
    for idx, pair in enumerate(all_candidate_pairs):
        if pair in relevant_set:
            first_relevant_rank_in_candidates = idx + 1
            break

    relevant_doc_types = sorted(
        {_source_file_to_doc_type(sf) for _, sf in relevant_set}
    )

    relevant_files = sorted({_extract_source_file(item) for item in relevant_doc_ids})
    unique_retrieved_files = [f"{p}::{s}" for p, s in unique_retrieved_pairs]
    true_positive_file_ids = [f"{p}::{s}" for p, s in true_positive_pairs]

    return {
        "query": query,
        "relevant_doc_ids": relevant_doc_ids,
        "relevant_file_ids": relevant_files,
        "relevant_doc_types": relevant_doc_types,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "retrieved_file_ids": unique_retrieved_files,
        "true_positive_file_ids": true_positive_file_ids,
        "true_positive_count": len(true_positive_pairs),
        "retrieved_file_count": len(unique_retrieved_pairs),
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "hit_at_k": hit_at_k,
        "reciprocal_rank": reciprocal_rank,
        "first_relevant_rank": first_relevant_rank,
        "first_relevant_rank_in_candidates": first_relevant_rank_in_candidates,
    }


def _aggregate(query_reports: list[dict[str, Any]], k: int) -> dict[str, Any]:
    query_count = len(query_reports)
    macro_precision = (
        sum(report["precision_at_k"] for report in query_reports) / query_count
        if query_reports
        else 0.0
    )
    macro_recall = (
        sum(report["recall_at_k"] for report in query_reports) / query_count
        if query_reports
        else 0.0
    )

    total_true_positives = sum(report["true_positive_count"] for report in query_reports)
    total_retrieved = sum(report.get("retrieved_file_count", 0) for report in query_reports)
    total_relevant = sum(len(report["relevant_file_ids"]) for report in query_reports)
    micro_precision = (
        total_true_positives / total_retrieved if total_retrieved else 0.0
    )
    micro_recall = total_true_positives / total_relevant if total_relevant else 0.0
    hit_count = sum(int(report.get("hit_at_k", 0)) for report in query_reports)
    zero_recall_count = sum(1 for report in query_reports if report["recall_at_k"] == 0.0)

    mrr = (
        sum(report.get("reciprocal_rank", 0.0) for report in query_reports)
        / query_count
        if query_reports
        else 0.0
    )

    doc_type_counts: dict[str, dict[str, int | float]] = {}
    for report in query_reports:
        for doc_type in report.get("relevant_doc_types", []):
            if doc_type not in doc_type_counts:
                doc_type_counts[doc_type] = {
                    "query_count": 0,
                    "hit_count": 0,
                    "zero_recall_count": 0,
                }
            doc_type_counts[doc_type]["query_count"] += 1
            if report.get("hit_at_k"):
                doc_type_counts[doc_type]["hit_count"] += 1
            if report["recall_at_k"] == 0.0:
                doc_type_counts[doc_type]["zero_recall_count"] += 1

    for doc_type, counts in doc_type_counts.items():
        qc = counts["query_count"]
        counts["hit_rate"] = counts["hit_count"] / qc if qc else 0.0

    zero_recall_queries = [
        {
            "query": r["query"],
            "first_relevant_rank_in_candidates": r.get(
                "first_relevant_rank_in_candidates"
            ),
        }
        for r in query_reports
        if r["recall_at_k"] == 0.0
    ]

    return {
        "k": k,
        "query_count": query_count,
        "macro_precision_at_k": macro_precision,
        "macro_recall_at_k": macro_recall,
        "micro_precision_at_k": micro_precision,
        "micro_recall_at_k": micro_recall,
        "total_true_positives": total_true_positives,
        "total_retrieved_files": total_retrieved,
        "total_relevant": total_relevant,
        "hit_rate_at_k": (hit_count / query_count) if query_count else 0.0,
        "zero_recall_query_count": zero_recall_count,
        "mrr": mrr,
        "by_doc_type": doc_type_counts,
        "zero_recall_queries": zero_recall_queries,
    }


@traceable(name="retrieval_eval_run", run_type="chain")
def run_evaluation(
    static_dataset_path: Path,
    report_dir: Path,
    *,
    collection: str,
    k: int,
    alpha: float,
    candidate_k: int,
    max_chunks_per_source: int,
) -> Path:
    records = _load_dataset(static_dataset_path)
    dataset_hash = _sha256(static_dataset_path)
    report_dir.mkdir(parents=True, exist_ok=True)

    client = build_weaviate_client()
    try:
        vector_store = WeaviateVectorStore(
            client=client,
            index_name=collection,
            text_key="content",
            embedding=OpenAIEmbeddings(api_key=SecretStr(settings.openai_api_key)),
            attributes=[
                "parking_id",
                "content_type",
                "source_file",
                "chunk_index",
                "metadata",
            ],
        )
        query_reports: list[dict[str, Any]] = []
        for record in records:
            query_report = _evaluate_query(
                vector_store=vector_store,
                query=record["query"],
                parking_id=record["parking_id"],
                relevant_doc_ids=record["relevant_doc_ids"],
                k=k,
                alpha=alpha,
                candidate_k=candidate_k,
                max_chunks_per_source=max_chunks_per_source,
            )
            query_reports.append(query_report)
    finally:
        client.close()

    aggregate = _aggregate(query_reports=query_reports, k=k)
    timestamp = datetime.now(timezone.utc)
    report_payload = {
        "meta": {
            "generated_at_utc": timestamp.isoformat(),
            "static_dataset_path": str(static_dataset_path),
            "dataset_sha256": dataset_hash,
            "collection": collection,
            "k": k,
            "alpha": alpha,
            "candidate_k": candidate_k,
            "max_chunks_per_source": max_chunks_per_source,
            "query_count": len(records),
        },
        "metrics": aggregate,
        "queries": query_reports,
    }

    output_name = f"retrieval_eval_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    output_path = report_dir / output_name
    output_path.write_text(
        json.dumps(report_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure retrieval quality with Recall@K and Precision@K."
    )
    parser.add_argument("--static-dataset", default=str(DEFAULT_STATIC_DATASET))
    parser.add_argument("--output-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument(
        "--max-chunks-per-source", type=int, default=DEFAULT_MAX_CHUNKS_PER_SOURCE
    )
    parser.add_argument("--langsmith-project", default="parking-agent-evaluations")
    parser.add_argument("--min-macro-recall", type=float)
    parser.add_argument("--min-macro-precision", type=float)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    k = max(1, min(int(args.k), 50))
    candidate_k = max(k, min(int(args.candidate_k), 100))
    max_chunks_per_source = max(1, min(int(args.max_chunks_per_source), 10))
    alpha = float(args.alpha)
    static_dataset_path = Path(args.static_dataset)
    report_dir = Path(args.output_dir)

    with tracing_context(
        project_name=args.langsmith_project,
        tags=["evaluation", "retrieval", "offline"],
        metadata={
            "eval_type": "retrieval_quality",
            "k": k,
            "alpha": alpha,
            "static_dataset_path": str(static_dataset_path),
        },
    ):
        report_path = run_evaluation(
            static_dataset_path=static_dataset_path,
            report_dir=report_dir,
            collection=args.collection,
            k=k,
            alpha=alpha,
            candidate_k=candidate_k,
            max_chunks_per_source=max_chunks_per_source,
        )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    print(
        (
            f"Saved retrieval evaluation report: {report_path}\n"
            f"Macro Precision@{k}: {metrics.get('macro_precision_at_k', 0.0):.4f}\n"
            f"Macro Recall@{k}: {metrics.get('macro_recall_at_k', 0.0):.4f}\n"
            f"Micro Precision@{k}: {metrics.get('micro_precision_at_k', 0.0):.4f}\n"
            f"Micro Recall@{k}: {metrics.get('micro_recall_at_k', 0.0):.4f}\n"
            f"Hit Rate@{k}: {metrics.get('hit_rate_at_k', 0.0):.4f}\n"
            f"MRR: {metrics.get('mrr', 0.0):.4f}\n"
            f"Zero Recall Queries: {metrics.get('zero_recall_query_count', 0)}"
        )
    )
    failures: list[str] = []
    macro_precision = float(metrics.get("macro_precision_at_k", 0.0))
    macro_recall = float(metrics.get("macro_recall_at_k", 0.0))
    if args.min_macro_precision is not None and macro_precision < args.min_macro_precision:
        failures.append(
            "macro_precision_at_k "
            f"{macro_precision:.4f} < required {args.min_macro_precision:.4f}"
        )
    if args.min_macro_recall is not None and macro_recall < args.min_macro_recall:
        failures.append(
            f"macro_recall_at_k {macro_recall:.4f} < required {args.min_macro_recall:.4f}"
        )
    if failures:
        print("Threshold check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

