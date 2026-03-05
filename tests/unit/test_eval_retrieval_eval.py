"""Tests for retrieval evaluation helpers."""

from __future__ import annotations

import json

import pytest

from parking_agent.eval import retrieval_eval as retrieval_eval_mod


def test_load_dataset_parses_valid_jsonl(tmp_path) -> None:
    """Dataset loader returns normalized record list."""
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "query": "What are parking hours?",
                "parking_id": "airport_parking",
                "relevant_doc_ids": ["general_info.md", "faq.md"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    records = retrieval_eval_mod._load_dataset(dataset)
    assert len(records) == 1
    assert records[0]["query"] == "What are parking hours?"
    assert records[0]["relevant_doc_ids"] == ["general_info.md", "faq.md"]


def test_load_dataset_rejects_invalid_payload(tmp_path) -> None:
    """Loader fails if required fields are malformed."""
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        json.dumps({"query": "", "relevant_doc_ids": []}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid 'query'"):
        retrieval_eval_mod._load_dataset(dataset)


def test_aggregate_computes_macro_micro_and_hit_rate() -> None:
    """Aggregate combines per-query metrics into summary scores."""
    reports = [
        {
            "query": "q1",
            "precision_at_k": 1.0,
            "recall_at_k": 0.5,
            "true_positive_count": 1,
            "retrieved_file_count": 1,
            "relevant_file_ids": ["a", "b"],
            "hit_at_k": 1,
            "reciprocal_rank": 1.0,
            "relevant_doc_types": ["faq"],
            "first_relevant_rank_in_candidates": 1,
        },
        {
            "query": "q2",
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "true_positive_count": 0,
            "retrieved_file_count": 1,
            "relevant_file_ids": ["c"],
            "hit_at_k": 0,
            "reciprocal_rank": 0.0,
            "relevant_doc_types": ["policies"],
            "first_relevant_rank_in_candidates": None,
        },
    ]
    result = retrieval_eval_mod._aggregate(query_reports=reports, k=5)
    assert result["query_count"] == 2
    assert result["macro_precision_at_k"] == 0.5
    assert result["macro_recall_at_k"] == 0.25
    assert result["hit_rate_at_k"] == 0.5
