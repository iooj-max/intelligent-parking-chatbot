# Development Plan — Stage 1: RAG System & Chatbot

## Overview

**Goal:** Build an intelligent parking chatbot using RAG (Retrieval-Augmented Generation) that provides parking information, collects reservation data, and protects sensitive information.

**Developer:** Solo developer, local development on macOS, committing directly to `main`.

**Tech Stack:**
- Python 3.11+
- LangChain + LangGraph + LangSmith
- Weaviate (vector database)
- PostgreSQL (SQL database for dynamic data)

---

## Project Structure (Target)

```
intelligent-parking-chatbot/
├── src/
│   ├── __init__.py
│   ├── main.py                  # Application entry point
│   ├── config.py                # Settings and env configuration
│   ├── chatbot/
│   │   ├── __init__.py
│   │   ├── graph.py             # LangGraph chatbot workflow
│   │   ├── nodes.py             # Graph nodes (retrieve, generate, collect_input)
│   │   ├── state.py             # Chatbot state definition
│   │   └── prompts.py           # System prompts and templates
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── retriever.py         # Unified retriever (vector + SQL)
│   │   ├── vector_store.py      # Weaviate integration
│   │   └── sql_store.py         # PostgreSQL integration for dynamic data
│   ├── guardrails/
│   │   ├── __init__.py
│   │   ├── input_filter.py      # Input validation and filtering
│   │   └── output_filter.py     # Sensitive data detection in responses
│   └── data/
│       ├── __init__.py
│       └── loader.py            # Data ingestion scripts
├── data/
│   ├── static/                  # Static parking info (for vector DB)
│   └── dynamic/                 # Sample dynamic data (prices, availability)
├── tests/
│   ├── __init__.py
│   ├── test_rag.py
│   ├── test_chatbot.py
│   ├── test_guardrails.py
│   └── test_evaluation.py
├── evaluation/
│   ├── evaluate_rag.py          # RAG evaluation script (recall@k, precision)
│   └── results/                 # Evaluation reports
├── docker-compose.yml           # Weaviate + PostgreSQL for local dev
├── langgraph.json               # LangGraph deployment config (for LangSmith Studio)
├── pyproject.toml               # Project config and dependencies
├── .env.example                 # Environment variables template
├── .gitignore
└── README.md
```

---

## Development Steps

### Step 1: Project Setup

**What to do:**
- Initialize Python project with `pyproject.toml` (dependencies, linting, testing config).
- Create `.gitignore` (Python, .env, IDE files).
- Create `.env.example` with required env vars (API keys, DB connection strings).
- Set up `docker-compose.yml` for local Weaviate and PostgreSQL.
- Create `src/config.py` to load settings from environment.

**Dependencies:**
```
langchain, langchain-openai, langgraph, langsmith
langgraph-cli[inmem]          # Local LangGraph server for LangSmith Studio
weaviate-client
psycopg2-binary, sqlalchemy
python-dotenv
pydantic, pydantic-settings
pytest, pytest-asyncio
```

- Create `langgraph.json` — LangGraph deployment config pointing to the compiled graph.
  This file tells `langgraph dev` where to find the graph, enabling interactive testing in LangSmith Studio.
  ```json
  {
    "graphs": {
      "parking_chatbot": "./src/chatbot/graph.py:graph"
    },
    "env": ".env"
  }
  ```

**Acceptance criteria:**
- `docker compose up` starts Weaviate and PostgreSQL locally.
- Python environment is set up and imports work.

---

### Step 2: Data Preparation & Ingestion ✓ COMPLETED

**What to do:**
- ✓ Design Weaviate schema (`ParkingContent` collection)
- ✓ Design PostgreSQL schema (5 tables: facilities, working_hours, special_hours, pricing_rules, space_availability)
- ✓ Prepare test data for 2 parking facilities (downtown_plaza, airport_parking)
- ✓ Implement `src/rag/vector_store.py` — Weaviate client wrapper
- ✓ Implement `src/rag/sql_store.py` — PostgreSQL ORM models and query layer
- ✓ Implement `src/data/loader.py` — Idempotent data loading script
- ✓ Implement `src/data/chunker.py` — Markdown chunking utilities
- ✓ Implement `src/data/embeddings.py` — OpenAI embeddings wrapper
- ✓ Create test data files (12 markdown + 10 CSV files)

**Acceptance criteria:**
- ✓ Static data is indexed in Weaviate with embeddings
- ✓ Dynamic data is queryable from PostgreSQL
- ✓ Loader script is idempotent (can re-run safely)
- ✓ Test data available for 2 parking lots (downtown_plaza, airport_parking)

---

### Step 3: RAG Retriever ✓ COMPLETED

**What to do:**
- ✓ Implement `src/rag/retriever.py`:
  - ✓ Accept a user query.
  - ✓ Classify query intent (STATIC, DYNAMIC, HYBRID, RESERVATION).
  - ✓ Infer parking_id from query using pattern matching.
  - ✓ Retrieve from Weaviate (similarity search) for static info.
  - ✓ Query PostgreSQL for dynamic info (availability, prices, hours).
  - ✓ Merge and return context formatted as markdown string.
  - ✓ Implement error handling with graceful degradation.
- ✓ Write comprehensive tests in `tests/test_rag.py`.

**Implementation details:**
- Created `ParkingRetriever` class with keyword-based intent classification
- Supports parking ID inference from natural language (e.g., "downtown plaza", "airport")
- Returns structured `RetrievalResult` with metadata or formatted string
- Graceful fallback when Weaviate or PostgreSQL unavailable
- 41 passing unit tests with mocked dependencies
- 3 integration tests (require running databases)

**Acceptance criteria:**
- ✓ Retriever returns relevant context for different query types.
- ✓ Tests pass with mocked databases (41/41 unit tests passing).
- ✓ Integration tests work with real Weaviate + PostgreSQL.

---

### Step 4: Chatbot with LangGraph ✓ COMPLETED

**What to do:**
- ✓ Define chatbot state in `src/chatbot/state.py` (messages, context, reservation data)
- ✓ Define system prompts in `src/chatbot/prompts.py` (info mode, reservation mode)
- ✓ Build the LangGraph workflow in `src/chatbot/graph.py`:
  - ✓ **Router node:** Classify user intent (info request vs. reservation)
  - ✓ **Retrieve node:** Call RAG retriever for context
  - ✓ **Generate node:** LLM generates response using context
  - ✓ **Collect input node:** Guide user through reservation data collection
  - ✓ **Validate input node:** Validate reservation fields with retry logic
  - ✓ **Check completion node:** Verify all fields collected
  - ✓ **Confirm reservation node:** Show summary and ask for confirmation
- ✓ Implement all 7 nodes in `src/chatbot/nodes.py`
- ✓ Create `src/main.py` — CLI REPL interface
- ✓ Export compiled graph as `graph` in `src/chatbot/graph.py` for `langgraph.json`
- ✓ Write comprehensive test suite in `tests/test_chatbot.py`

**Implementation details:**
- StateGraph with dual modes: info (RAG-based Q&A) and reservation (step-by-step collection)
- 7 nodes with conditional routing based on state
- Field validation with graceful error messages and retry logic
- Mock-based unit tests (43 passing) and integration tests (marked separately)
- CLI interface with pretty printing and error handling

**Acceptance criteria:**
- ✓ Chatbot answers parking-related questions using retrieved context
- ✓ Chatbot collects reservation details (name, parking_id, date, start_time, duration_hours)
- ✓ Conversation flow works through StateGraph with proper routing
- ✓ Graph exported as `graph` variable for LangSmith Studio compatibility
- ✓ Unit tests pass (43/43 tests passing)
- ✓ CLI interface provides user-friendly interaction

---

### Step 5: Guardrails

**What to do:**
- Implement `src/guardrails/input_filter.py`:
  - Detect prompt injection attempts.
  - Validate input is parking-related (topic filtering).
- Implement `src/guardrails/output_filter.py`:
  - Scan LLM output for PII / sensitive data patterns (emails, phone numbers, credit cards).
  - Use regex + optionally a pre-trained NLP model (e.g., Presidio) for entity detection.
  - Mask or block responses containing sensitive data.
- Integrate guardrails into the LangGraph workflow (as pre/post-processing nodes).
- Write tests in `tests/test_guardrails.py`.

**Acceptance criteria:**
- Off-topic or malicious inputs are rejected gracefully.
- Responses containing PII patterns are masked before reaching the user.
- Tests cover common attack vectors and sensitive data patterns.

---

### Step 6: Evaluation

**What to do:**
- Create `evaluation/evaluate_rag.py`:
  - Build a test dataset: pairs of (question, expected_answer / expected_source_docs).
  - Measure retrieval quality: recall@k, precision@k, MRR.
  - Measure answer quality: faithfulness, relevance (using LangSmith evaluation or RAGAS).
- Run performance tests: response latency, throughput.
- Generate evaluation report in `evaluation/results/`.

**Acceptance criteria:**
- Evaluation script runs end-to-end and produces a report.
- Baseline metrics are documented.
- Report identifies areas for improvement.

---

**Note:** For local development workflow instructions, see [README.md](README.md).

---

## Deliverables (Stage 1)

| # | Deliverable | Step |
|---|-------------|------|
| 1 | Working chatbot with RAG | Steps 1-4 |
| 2 | Data split: static (Weaviate) + dynamic (PostgreSQL) | Step 2 |
| 3 | Interactive features: info + reservation collection | Step 4 |
| 4 | Guardrails: input/output filtering | Step 5 |
| 5 | Evaluation report with metrics | Step 6 |
