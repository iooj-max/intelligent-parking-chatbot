# Local Development

Run the parking chatbot locally with Docker Compose. All infrastructure (PostgreSQL, Weaviate, MCP filesystem, parking-bot) runs in containers.

## Prerequisites

- Python **3.11+**
- Docker and Docker Compose
- (Optional) LangSmith account for tracing

## First-Time Setup

```bash
# 1) Clone repository
git clone https://github.com/iooj-max/intelligent-parking-chatbot.git
cd intelligent-parking-chatbot

# 2) Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3) Install project with dependencies
pip install -e .

# 4) Create environment file
cp .env.example .env
# Edit .env: set TELEGRAM_BOT_TOKEN, OPENAI_API_KEY, etc.
```

## Docker Services

| Service | Image / Build | Purpose |
|---------|---------------|---------|
| **mcp-filesystem** | node:20-alpine | MCP filesystem endpoint for reservation status file writes |
| **postgres** | postgres:16-alpine | PostgreSQL: parking facilities, availability, pricing, working hours |
| **weaviate** | semitechnologies/weaviate:1.28.4 | Vector DB: static content (policies, FAQ, location, features) |
| **parking-bot** | Built from project | Telegram bot: LangGraph agent |

Ports: MCP filesystem `8081` (endpoint `/mcp`), Postgres `5432`, Weaviate REST `8080`, Weaviate gRPC `50051`.

## Commands to Run

### Initial Startup

```bash
# 1) Start all services (build local bot image on first run)
docker compose up -d --build

# 2) Load test data (runs locally, connects to Docker services)
.venv/bin/python -m src.data.loader
```

### View Bot Logs

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

## Run Bot Locally (Without Docker Container)

Instead of running the bot in Docker, run it as a local Python process:

```bash
source .venv/bin/activate
.venv/bin/python -m parking_agent.main
```

Requires `POSTGRES_HOST=localhost` and `WEAVIATE_HTTP_HOST=localhost` in `.env` (Docker services must be running).

## LangGraph Studio (Debugging)

Run the graph in LangSmith Studio for debugging:

```bash
langgraph dev
```

The CLI prints a Studio URL, e.g. `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`.

**Note:** Studio runs locally and uses `.env`. Ensure `POSTGRES_HOST=localhost` and `WEAVIATE_HTTP_HOST=localhost` so it connects to Docker services.

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

## Environment Variables (Local)

Create `.env` in the project root; use `.env.example` as reference.

Required for runtime:
- `TELEGRAM_BOT_TOKEN` — token from BotFather
- `OPENAI_API_KEY` — for LLM and embeddings

For local Docker:
- `POSTGRES_HOST=localhost` (or `postgres` when running inside Docker)
- `WEAVIATE_HTTP_HOST=localhost` (or `weaviate` when running inside Docker)
- `MCP_FILESYSTEM_URL=http://localhost:8081/mcp` (or `http://mcp-filesystem:8081/mcp` inside Docker)

**Weaviate authentication:** `WEAVIATE_API_KEY` is not required for local Weaviate. When unset or empty, the client connects without authentication (local Docker Weaviate has `AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true`). For Weaviate Cloud (production), set `WEAVIATE_API_KEY` in your environment.
