# Setup and Deployment

This page is the single starting point for environment setup and deployment.

## Choose Your Path

- Local development and debugging:
  - [Local Development](local-development.md)

- Production or cloud deployment:
  - [Cloud Deployment (Google Cloud Platform)](cloud-deployment.md)

## Recommended Order

1. Read [System Architecture](system-architecture.md).
2. Complete [Local Development](local-development.md).
3. Validate behavior with [Testing Guide](testing.md).
4. Deploy using [Cloud Deployment](cloud-deployment.md) when ready.

## Minimum Local Command Set

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env

docker compose up -d --build
.venv/bin/python -m src.data.loader
```

## Notes

- Keep secrets only in `.env` or secret managers.
- Do not commit runtime artifacts from `runtime/`.
- Use LangGraph Studio (`langgraph dev`) for graph-level debugging in local environment.
