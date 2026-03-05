variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for Cloud Run and Cloud SQL"
  type        = string
  default     = "us-central1"
}

variable "telegram_bot_token" {
  description = "Telegram bot token from BotFather"
  type        = string
  sensitive   = true
}

variable "openai_api_key" {
  description = "OpenAI API key for LLM and embeddings"
  type        = string
  sensitive   = true
}

variable "telegram_admin_chat_id" {
  description = "Telegram chat ID for admin approval notifications"
  type        = string
  default     = ""
}

variable "weaviate_url" {
  description = "External Weaviate URL (e.g. Weaviate Cloud). When set, Weaviate Cloud Run is not deployed. Leave empty to deploy Weaviate on Cloud Run (experimental, may fail)."
  type        = string
  default     = ""
}

variable "weaviate_api_key" {
  description = "Weaviate Cloud API key (required when using weaviate_url with Weaviate Cloud)"
  type        = string
  default     = ""
  sensitive   = true
}

variable "image_tag" {
  description = "Docker image tag for parking-bot (e.g. latest or commit SHA)"
  type        = string
  default     = "latest"
}

variable "min_instances" {
  description = "Minimum number of Cloud Run instances (0 = scale to zero)"
  type        = number
  default     = 0
}

variable "max_instances" {
  description = "Maximum number of Cloud Run instances"
  type        = number
  default     = 1
}

variable "db_password" {
  description = "PostgreSQL password for parking database"
  type        = string
  sensitive   = true
}

variable "cloud_sql_authorized_networks" {
  description = "Authorized networks for Cloud SQL public IP (required for Cloud SQL Proxy from local machine). Restrict in production."
  type = list(object({
    name  = string
    value = string
  }))
  default = [
    { name = "allow-all-dev", value = "0.0.0.0/0" }
  ]
}
