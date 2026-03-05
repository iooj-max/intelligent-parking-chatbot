# Terraform: Deploy Parking Bot to GCP

Deploys the full parking chatbot stack to Google Cloud Platform (equivalent to docker-compose):

- **Cloud Run** — parking-bot, Weaviate, MCP filesystem
- **Cloud SQL** — PostgreSQL 16
- **Artifact Registry** — Docker image storage
- **VPC Access Connector** — Cloud Run → Cloud SQL private connectivity

## Prerequisites

- **Google account** with a GCP project
- **Billing enabled** for the project — Cloud Run, Artifact Registry, Cloud SQL, and other services require a linked billing account. Enable at [console.cloud.google.com/billing](https://console.cloud.google.com/billing)
- [Terraform](https://www.terraform.io/downloads) >= 1.0
- [gcloud](https://cloud.google.com/sdk/docs/install) CLI installed and authenticated

## Quick Start

1. Copy and fill variables:

```bash
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values
```

2. Authenticate and set project:

```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

> **Note:** `gcloud auth application-default login` creates Application Default Credentials used by Terraform. Run this before `terraform plan` or `terraform apply`.

3. Build and push Docker images:

```bash
# From project root
gcloud auth configure-docker REGION-docker.pkg.dev

# Parking bot
docker build -t REGION-docker.pkg.dev/PROJECT_ID/parking-bot/parking-bot:latest .
docker push REGION-docker.pkg.dev/PROJECT_ID/parking-bot/parking-bot:latest

# MCP filesystem (pre-installed packages, avoids cold start)
docker build -f terraform/mcp-filesystem/Dockerfile -t REGION-docker.pkg.dev/PROJECT_ID/parking-bot/mcp-filesystem:latest .
docker push REGION-docker.pkg.dev/PROJECT_ID/parking-bot/mcp-filesystem:latest
```

Or use Cloud Build:

```bash
gcloud builds submit --tag REGION-docker.pkg.dev/PROJECT_ID/parking-bot/parking-bot:latest .
gcloud builds submit -f terraform/mcp-filesystem/Dockerfile --tag REGION-docker.pkg.dev/PROJECT_ID/parking-bot/mcp-filesystem:latest .
```

4. Apply Terraform:

```bash
cd terraform
terraform init
terraform plan
terraform apply
```

5. Load data into PostgreSQL and Weaviate (use Cloud SQL Proxy + Weaviate URL from outputs):

```bash
# After terraform apply, use outputs:
# weaviate_url — for WEAVIATE_HTTP_HOST (host only) and WEAVIATE_URL
# Cloud SQL: use Cloud SQL Proxy or connect via private IP from a VM in same VPC
.venv/bin/python -m src.data.loader
```

### Local access to Cloud SQL via Cloud SQL Proxy

To connect to Cloud SQL from your local machine (e.g. for data loading, debugging, or `psql`), use Cloud SQL Proxy. It forwards `localhost:5432` to the Cloud SQL instance.

1. Install Cloud SQL Proxy:

```bash
brew install cloud-sql-proxy
```

2. Start the proxy (run in a separate terminal and keep it running):

```bash
cloud-sql-proxy $(terraform output -raw cloud_sql_connection_name) --port=5432
```

Or with an explicit connection name (`PROJECT_ID:REGION:INSTANCE`):

```bash
cloud-sql-proxy YOUR_PROJECT_ID:YOUR_REGION:parking-db --port=5432
```

3. Configure `.env` for Cloud SQL:

```
POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5432
POSTGRES_DB=parking
POSTGRES_USER=parking
POSTGRES_PASSWORD=<db_password from terraform.tfvars>
```

4. Run the data loader or any tool that uses PostgreSQL; it will connect through the proxy to Cloud SQL.

## Variables

| Variable | Description |
|---------|-------------|
| `project_id` | GCP project ID |
| `region` | Region (default: us-central1) |
| `telegram_bot_token` | Token from BotFather |
| `openai_api_key` | OpenAI API key |
| `telegram_admin_chat_id` | Admin Telegram chat ID |
| `db_password` | PostgreSQL password |
| `image_tag` | Docker image tag |
| `min_instances` | Min Cloud Run instances (0 = scale to zero) |
| `max_instances` | Max Cloud Run instances |

## Outputs

- `artifact_registry_repository` — Image registry path
- `cloud_run_url` — Parking-bot Cloud Run URL
- `cloud_sql_private_ip` — Cloud SQL private IP
- `cloud_sql_connection_name` — Connection name for Cloud SQL Proxy (`PROJECT:REGION:INSTANCE`)
- `weaviate_url` — Weaviate Cloud Run URL (for data loader)
- `mcp_filesystem_url` — MCP filesystem Cloud Run URL

## Security

- Add `terraform.tfvars` to `.gitignore` (contains secrets)
- For production, use Secret Manager instead of plain env vars
- Restrict `allUsers` on Cloud Run if using authenticated webhooks
