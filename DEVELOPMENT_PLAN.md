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

### Step 2: Data Preparation & Ingestion

**What to do:**
- Prepare sample parking data (JSON/markdown files in `data/`).
  - Static: parking locations, general info, booking process, rules.
  - Dynamic: space availability, working hours, prices.
- Implement `src/data/loader.py`:
  - Load and chunk static data, embed and store in Weaviate.
  - Load dynamic data into PostgreSQL tables.
- Implement `src/rag/vector_store.py` — Weaviate client wrapper.
- Implement `src/rag/sql_store.py` — PostgreSQL query layer for dynamic data.

**Acceptance criteria:**
- Static data is indexed in Weaviate with embeddings.
- Dynamic data is queryable from PostgreSQL.
- Loader script is idempotent (can re-run safely).

---

### Step 3: RAG Retriever

**What to do:**
- Implement `src/rag/retriever.py`:
  - Accept a user query.
  - Classify query intent (static vs. dynamic data need).
  - Retrieve from Weaviate (similarity search) for static info.
  - Query PostgreSQL for dynamic info (availability, prices).
  - Merge and return context for LLM.
- Write tests in `tests/test_rag.py`.

**Acceptance criteria:**
- Retriever returns relevant context for different query types.
- Tests pass with mocked databases.

---

### Step 4: Chatbot with LangGraph

**What to do:**
- Define chatbot state in `src/chatbot/state.py` (messages, context, user_data).
- Define system prompts in `src/chatbot/prompts.py`.
- Build the LangGraph workflow in `src/chatbot/graph.py`:
  - **Router node:** Classify user intent (info request vs. reservation).
  - **Retrieve node:** Call RAG retriever for context.
  - **Generate node:** LLM generates response using context.
  - **Collect input node:** Guide user through reservation data collection.
- Implement nodes in `src/chatbot/nodes.py`.
- Create `src/main.py` — CLI interface to interact with the chatbot.
- Connect LangSmith for tracing (set env vars).
- Ensure the compiled graph is exported in `src/chatbot/graph.py` as `graph` so that
  `langgraph.json` can reference it (`graph = workflow.compile()`).
- Verify the graph works in LangSmith Studio via `langgraph dev`.

**Acceptance criteria:**
- Chatbot answers parking-related questions using retrieved context.
- Chatbot can collect reservation details (name, date, time, parking spot).
- Conversation flow is visible in LangSmith.
- `langgraph dev` starts the local server and the graph is testable interactively in LangSmith Studio.

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

## Local Development Workflow (macOS)

### First-time setup

```bash
# 1. Clone the repository
git clone https://github.com/iooj-max/intelligent-parking-chatbot.git
cd intelligent-parking-chatbot

# 2. Create Python virtual environment (requires Python 3.11+)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install the project and all dependencies
pip install -e ".[dev]"

# 4. Set up environment variables
cp .env.example .env
# Open .env in your editor and fill in:
#   - OPENAI_API_KEY
#   - LANGSMITH_API_KEY
#   - Database connection strings (defaults work with docker-compose)

# 5. Start Weaviate and PostgreSQL
docker compose up -d

# 6. Load parking data into databases
python -m src.data.loader
```

### Daily workflow: pull changes and run

```bash
cd intelligent-parking-chatbot
source .venv/bin/activate

# Pull latest code from main
git pull origin main

# Make sure infrastructure is running
docker compose up -d

# Run the chatbot (CLI mode)
python -m src.main

# — OR — Run in LangSmith Studio (interactive graph testing)
langgraph dev
# Opens a local LangGraph server. Go to LangSmith Studio in your browser —
# the graph will appear for interactive testing, step-by-step node debugging,
# and state inspection.
```

### After code changes: update dependencies

```bash
# If pyproject.toml changed — reinstall the project
pip install -e ".[dev]"

# If data files changed — reload data into databases
python -m src.data.loader
```

### After docker-compose.yml changes: rebuild containers

```bash
# Pull fresh images (Weaviate, PostgreSQL) if versions were bumped
docker compose pull

# Recreate containers with updated config (preserves data in volumes)
docker compose up -d

# If you need a clean start (DESTROYS ALL DATA in volumes):
# docker compose down -v
# docker compose up -d
# python -m src.data.loader   # re-load data after volume wipe
```

### Running tests and evaluation

```bash
# Run all tests
pytest

# Run a specific test file
pytest tests/test_rag.py

# Run evaluation and generate report
python -m evaluation.evaluate_rag
```

---

## Deliverables (Stage 1)

| # | Deliverable | Step |
|---|-------------|------|
| 1 | Working chatbot with RAG | Steps 1-4 |
| 2 | Data split: static (Weaviate) + dynamic (PostgreSQL) | Step 2 |
| 3 | Interactive features: info + reservation collection | Step 4 |
| 4 | Guardrails: input/output filtering | Step 5 |
| 5 | Evaluation report with metrics | Step 6 |
