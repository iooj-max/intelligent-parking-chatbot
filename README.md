# Parking Agent (Telegram Bot)

A RAG-based parking assistant that answers parking-related questions and supports reservation requests with administrator approval.

## Documentation

Start here for system usage and internals:

| Guide | Description |
|-------|-------------|
| [Project Presentation](docs/presentations/Parking%20Chatbot%20-%20Presentation.pdf) | Product and system overview slides |
| [System Architecture](docs/system-architecture.md) | High-level architecture, components, data flow, and state boundaries |
| [Agent and Server Logic](docs/agent-server-logic.md) | Routing flow, execution flow, thread model, and admin approval logic |
| [Setup and Deployment](docs/setup-and-deployment.md) | End-to-end setup path with links to local and cloud deployment guides |
| [Local Development](docs/local-development.md) | Docker Compose setup, local run, data loading, and LangGraph Studio |
| [Cloud Deployment](docs/cloud-deployment.md) | Terraform and manual deployment to Google Cloud |
| [Testing Guide](docs/testing.md) | Test strategy, scope, and run commands |

## Technology Stack

| Layer | Technology |
|-------|------------|
| Agent and orchestration | LangChain, LangGraph |
| LLM | OpenAI GPT |
| Vector store | Weaviate |
| Relational DB | PostgreSQL |
| Messaging | Telegram (python-telegram-bot) |
| Tracing and observability | LangSmith |
| Cloud platform | Google Cloud Platform (Cloud Run, Cloud SQL) |
| Infrastructure as code | Terraform |
| Infrastructure | Docker, Docker Compose |

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env

docker compose up -d --build
.venv/bin/python -m src.data.loader
```

For complete setup and deployment instructions, use [docs/setup-and-deployment.md](docs/setup-and-deployment.md).
