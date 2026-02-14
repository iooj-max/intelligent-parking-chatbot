# Evaluation Datasets

## RAG Test Cases (`rag_test_cases.json`)

Test cases for evaluating retrieval quality:
- Recall@K
- Precision@K
- MRR

**Format:**
```json
{
  "query": "User question",
  "relevant_ids": ["list", "of", "relevant", "document", "ids"],
  "description": "Optional test case description"
}
```

**Document ID formats:**
- Static content: `data/static/{facility}/{filename}.md`
- Dynamic facility signal: `{facility_id}` (e.g., `downtown_plaza`, `airport_parking`)

## Answer Test Cases (`answer_test_cases.json`)

Test cases for evaluating answer quality:
- Faithfulness
- Answer relevancy

**Format:**
```json
{
  "question": "User question",
  "reference_answer": "Expected answer content (for human verification)"
}
```

Reference answers are for documentation only; RAGAS uses model-based scoring, not exact matching.

## How to Run Testing + Evaluation

### Prerequisites
1. Activate your environment and install deps:
   ```bash
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```
2. Ensure infra is up:
   ```bash
   docker compose up -d
   ```
3. Ensure data is loaded:
   ```bash
   python -m src.data.loader
   ```
4. Ensure `OPENAI_API_KEY` is available in `.env` (or exported in shell).

### 1) Regression tests (fast)
```bash
OPENAI_API_KEY=sk-test-key PYTHONPATH=. pytest -q tests/test_output_filter.py tests/test_evaluation_metrics.py
```

### 2) Retrieval quality (Recall@K / Precision@K / MRR)
```bash
python -m evaluation.evaluate_rag --dataset evaluation/datasets/rag_test_cases.json --k 5
```

### 3) Answer quality (RAGAS)
```bash
python -m evaluation.evaluate_answers --dataset evaluation/datasets/answer_test_cases.json
```

### 4) Performance benchmarking
```bash
python -m evaluation.performance --dataset evaluation/datasets/answer_test_cases.json --iterations 3 --concurrent 5
```

All generated reports are written to `evaluation/results/`.

## Adding Test Cases

When adding test cases:
1. Cover all content types (hours, pricing, security, booking, etc.)
2. Test both facilities (`downtown_plaza`, `airport_parking`)
3. Include dynamic-data scenarios (availability/pricing)
4. Ensure `relevant_ids` match actual files/facility IDs used by retrieval output
