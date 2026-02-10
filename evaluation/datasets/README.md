# Evaluation Datasets

## RAG Test Cases (`rag_test_cases.json`)

Test cases for evaluating retrieval quality (recall, precision, MRR).

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
- Dynamic data: `{facility_name}` (e.g., "downtown_plaza")

## Answer Test Cases (`answer_test_cases.json`)

Test cases for evaluating answer quality (faithfulness, relevance).

**Format:**
```json
{
  "question": "User question",
  "reference_answer": "Expected answer content (for human verification)"
}
```

Reference answers are for documentation only - RAGAS uses LLM-based evaluation, not exact matching.

## Adding Test Cases

When adding test cases:
1. Cover all content types (hours, pricing, security, etc.)
2. Test both facilities (Downtown Plaza, Airport Parking)
3. Include edge cases (real-time availability, reservations)
4. Ensure relevant_ids match actual data files/facilities
