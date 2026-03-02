# Parking Agent (Telegram Bot)

A RAG-based parking assistant that answers questions about facilities, pricing, and availability, guides users through reservations, and routes completed bookings to an administrator for approval (human-in-the-loop).

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| **Agent & orchestration** | LangChain, LangGraph |
| **LLM** | OpenAI (GPT) |
| **Vector store** | Weaviate |
| **Relational DB** | PostgreSQL |
| **Messaging** | python-telegram-bot |
| **Tracing & observability** | LangSmith |
| **Infrastructure** | Docker, Docker Compose |

---

## Requirements

- Python **3.11+**
- Docker and Docker Compose
- (Optional) LangSmith account for tracing and evaluation visibility

## First-Time Setup

```bash
# 1) Clone repository
git clone https://github.com/iooj-max/intelligent-parking-chatbot.git
cd intelligent-parking-chatbot

# 2) Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3) Install project with dependencies
pip install -e .

# 4) Create environment file
cp .env.example .env
# Edit .env: set TELEGRAM_BOT_TOKEN, OPENAI_API_KEY, etc.
```

## Project Structure

```
src/
├── config.py                 # Settings (env vars)
├── parking_agent/            # LangGraph agent, Telegram bot
│   ├── main.py               # Entry point (long polling)
│   ├── graph.py              # LangGraph workflow (scope, info, reservation)
│   ├── tools.py              # RAG, facility validation, SQL tools
│   ├── agent_runners.py      # ReAct agent for info retrieval
│   ├── retrieval.py          # Weaviate retriever
│   ├── facility_validation.py # DB + LLM facility matching
│   ├── prompts.py            # LLM prompts
│   ├── schemas.py            # Pydantic models
│   ├── clients.py            # Postgres, Weaviate clients
│   ├── fetch_trace.py        # LangSmith trace export
│   └── eval/                 # Retrieval & performance eval scripts
└── data/                     # Data loading (Python modules)
    ├── loader.py             # CLI: load static + dynamic data
    ├── sql_store.py          # PostgreSQL models
    ├── vector_store.py       # Weaviate store
    └── chunker.py            # Text chunking for RAG

data/static/                   # Static markdown per facility (policies, FAQ, etc.)
data/dynamic/                  # CSV fixtures per facility
eval/                          # Eval datasets (JSONL)
runtime/                       # Reservation status files (output)
scripts/                       # Debug utilities
```

## Running the Project

All infrastructure runs in Docker. The data loader and LangGraph Studio run locally and connect to Docker services.

### Services

| Service      | Image / Build | Purpose                                                                 |
|-------------|---------------|-------------------------------------------------------------------------|
| **mcp-filesystem** | node:20-alpine (`mcp-proxy` + `@modelcontextprotocol/server-filesystem`) | MCP filesystem endpoint for reservation status file writes |
| **postgres** | postgres:16-alpine | PostgreSQL: parking facilities, availability, pricing, working hours |
| **weaviate** | semitechnologies/weaviate:1.28.4 | Vector DB: static content (policies, FAQ, location, features) |
| **parking-bot** | Built from project | Telegram bot: LangGraph agent, connects to postgres and weaviate   |

Ports: MCP filesystem `8081` (internal endpoint `/mcp`), Postgres `5432`, Weaviate REST `8080`, Weaviate gRPC `50051`.

### Initial Startup

```bash
# 1) Start infrastructure
docker compose up -d

# 2) Load test data (runs locally, connects to Docker)
.venv/bin/python -m src.data.loader

# 3) Start or refresh all services, including MCP filesystem
docker compose up -d

# 4) Start the bot (rebuild when code changed)
docker compose up -d parking-bot --build
```

View bot logs:

```bash
docker compose logs -f parking-bot
```

### Updating After Code Changes

```bash
# Rebuild and restart the bot
docker compose up -d parking-bot --build
```

### Updating After docker-compose.yml Changes

```bash
# Pull fresh images if versions were bumped
docker compose pull

# Recreate containers (preserves data in volumes)
docker compose up -d
```

### Clean Start (Destroys All Data)

```bash
docker compose down -v
docker compose up -d
.venv/bin/python -m src.data.loader   # re-load data
```

### LangGraph Studio (Local Development)

Run the graph in LangSmith Studio for debugging:

```bash
langgraph dev
```

The CLI prints a Studio URL, e.g. `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`.

**Note:** Studio runs locally and uses `.env`. Ensure `POSTGRES_HOST=localhost` and `WEAVIATE_HTTP_HOST=localhost` so it connects to Docker services.

### Run Bot Locally (Long Polling)

Instead of the Docker container:

```bash
source .venv/bin/activate
.venv/bin/python -m parking_agent.main
```

Requires `POSTGRES_HOST=localhost` and `WEAVIATE_HTTP_HOST=localhost` in `.env`.

---

## Environment Variables

Create `.env` in the project root; use `.env.example` as reference.

Required for runtime:
- `TELEGRAM_BOT_TOKEN` — token from BotFather
- `OPENAI_API_KEY` — for LLM and embeddings

MCP storage integration:
- `MCP_FILESYSTEM_URL` — MCP streamable HTTP endpoint for filesystem tool calls (example: `http://localhost:8081/mcp`)
- `MCP_FILESYSTEM_TIMEOUT_SECONDS` — timeout for MCP tool calls
- `MCP_RESERVATION_STATUS_DIR` — allowed directory inside MCP filesystem server for reservation status logs

---

## Loading Test Data

**MVP only.** Test data for two facilities: `downtown_plaza`, `airport_parking`.

| Command | Description |
|---------|-------------|
| `.venv/bin/python -m src.data.loader` | Full load (idempotent) |
| `... --reset` | Delete all data, then reload (asks for confirmation) |
| `... --parking-id downtown_plaza` | Load specific facility |
| `... --static-only` | Weaviate only |
| `... --dynamic-only` | PostgreSQL only |
| `... --static-only --reset` | Reset and reload Weaviate only |
| `... --dynamic-only --reset` | Reset and reload PostgreSQL only |
| `... --verbose` | Verbose logging |

What the loader does:
1. Creates Weaviate `ParkingContent` collection (if needed)
2. Loads and embeds static markdown content
3. Creates PostgreSQL tables (if needed)
4. Loads dynamic CSV data

---

## Verify Loaded Data

### Weaviate

```bash
curl http://localhost:8080/v1/schema | jq
curl "http://localhost:8080/v1/objects?class=ParkingContent&limit=5" | jq
curl "http://localhost:8080/v1/objects?class=ParkingContent" | jq '.objects | length'
```

### PostgreSQL

```bash
docker compose exec postgres psql -U parking -d parking
# SELECT * FROM parking_facilities;
# SELECT * FROM space_availability;
# \q
```

---

## Tests

### Scope Guardrail

```bash
.venv/bin/pytest tests/test_scope_guardrail.py -v
```

Add cases in `tests/test_scope_guardrail.py` → `SCOPE_TEST_CASES`.

### Facility Validation

```bash
.venv/bin/pytest tests/test_facility_validation.py -v
```

---

## Export Trace from LangSmith

```bash
.venv/bin/python -m parking_agent.fetch_trace --trace-id <TRACE_ID>
```

Trace files are saved to `traces/<trace_id>.json`.

---

## RAG Evaluation (Retrieval-Only)

Offline scripts for retrieval quality and latency.

### Quality Metrics: Recall@K and Precision@K

```bash
set -a; source .env; set +a
.venv/bin/python -m parking_agent.eval.retrieval_eval \
  --static-dataset eval/static_retrieval_eval_dataset.jsonl \
  --output-dir eval/reports
```

Defaults: k=5, alpha=0.5, candidate_k=20, max_chunks_per_source=1.

Optional: `--min-macro-recall`, `--min-macro-precision` for CI thresholds.

**Goals (k=5):** Hit Rate@5 ≥ 95%, Macro Recall@5 ≥ 90%, MRR ≥ 0.5.

### Performance (Single-User Latency)

```bash
set -a; source .env; set +a
.venv/bin/python -m parking_agent.eval.performance_eval \
  --static-dataset eval/static_retrieval_eval_dataset.jsonl \
  --dynamic-dataset eval/dynamic_retrieval_eval_dataset.jsonl \
  --repeats 1 \
  --inter-call-delay-ms 5000 \
  --max-p95-ms 2500 \
  --max-error-rate 0.01 \
  --output-dir eval/reports
```

Outputs: p50/p95/p99, error rate, JSON report. Use `--inter-call-delay-ms` to reduce 429 rate-limit errors. Use `--no-progress` to disable the progress bar. Both eval scripts publish runs to LangSmith when tracing is enabled.
