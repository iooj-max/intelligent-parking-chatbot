# intelligent-parking-chatbot

An intelligent chatbot that can interact with users, provide information about parking spaces, handle the reservation process, and involve a human administrator for confirmation ("human-in-the-loop"). Based on Retrieval-Augmented Generation.

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

## Loading Test Data

⚠️ **FOR TESTING PURPOSES ONLY**

This project includes test data for 2 parking facilities (`downtown_plaza` and `airport_parking`) to help you get started with development. This data loader is NOT production-ready and should only be used for local testing.

### First-time data load

After starting the infrastructure with `docker compose up -d`, load the test data:

```bash
python -m src.data.loader
```

This script will:
1. Create the Weaviate `ParkingContent` collection
2. Load and embed static content from markdown files
3. Create PostgreSQL tables with constraints
4. Load dynamic data from CSV files

The loading process is **idempotent** - you can run it multiple times safely without creating duplicates.

### Resetting test data

To clear and reload all test data:

```bash
python -m src.data.loader --reset
```

⚠️ **Warning**: This will DELETE all existing data in both Weaviate and PostgreSQL.

### Loading specific parking facilities

```bash
# Load only downtown_plaza
python -m src.data.loader --parking-id downtown_plaza

# Load only airport_parking
python -m src.data.loader --parking-id airport_parking
```

### Verifying data load

**Check Weaviate:**
```bash
# View schema
curl http://localhost:8080/v1/schema | jq

# Query objects (requires jq)
curl "http://localhost:8080/v1/objects?class=ParkingContent&limit=5" | jq

# Count total objects
curl "http://localhost:8080/v1/objects?class=ParkingContent" | jq '.objects | length'
```

**Check PostgreSQL:**
```bash
# Connect to database
docker compose exec postgres psql -U parking -d parking

# Query tables
SELECT * FROM parking_facilities;
SELECT * FROM space_availability;
SELECT * FROM working_hours;
SELECT * FROM pricing_rules;

# Exit PostgreSQL
\q
```

### Troubleshooting

**Connection refused error:**
- Ensure `docker compose up -d` is running
- Check containers: `docker compose ps`
- View logs: `docker compose logs weaviate` or `docker compose logs postgres`

**OpenAI API error:**
- Check that `OPENAI_API_KEY` is set in `.env`
- Verify API key is valid at https://platform.openai.com/api-keys

**Duplicate data or inconsistent state:**
- Run `python -m src.data.loader --reset` to clean and reload all data

**Import errors:**
- Reinstall the project: `pip install -e ".[dev]"`
- Verify you're in the virtual environment: `which python` should show `.venv` 
