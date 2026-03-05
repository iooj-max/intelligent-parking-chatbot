# Enable required APIs
resource "google_project_service" "run" {
  project = var.project_id
  service = "run.googleapis.com"
}

resource "google_project_service" "artifactregistry" {
  project = var.project_id
  service = "artifactregistry.googleapis.com"
}

resource "google_project_service" "sqladmin" {
  project = var.project_id
  service = "sqladmin.googleapis.com"
}

resource "google_project_service" "servicenetworking" {
  project = var.project_id
  service = "servicenetworking.googleapis.com"
}

resource "google_project_service" "vpcaccess" {
  project = var.project_id
  service = "vpcaccess.googleapis.com"
}

resource "google_project_service" "compute" {
  project = var.project_id
  service = "compute.googleapis.com"
}

# Artifact Registry for Docker images
resource "google_artifact_registry_repository" "parking_bot" {
  project       = var.project_id
  location      = var.region
  repository_id = "parking-bot"
  format        = "DOCKER"

  depends_on = [google_project_service.artifactregistry]
}

# Private IP range for Cloud SQL
resource "google_compute_global_address" "private_ip_range" {
  project       = var.project_id
  name          = "cloud-sql-private-range"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = "projects/${var.project_id}/global/networks/default"

  depends_on = [google_project_service.compute]
}

resource "google_service_networking_connection" "private_vpc" {
  network                 = "projects/${var.project_id}/global/networks/default"
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_range.name]

  depends_on = [google_project_service.servicenetworking]
}

# Cloud SQL (PostgreSQL)
resource "google_sql_database_instance" "parking" {
  project          = var.project_id
  name             = "parking-db"
  database_version = "POSTGRES_16"
  region           = var.region

  settings {
    tier = "db-f1-micro"

    ip_configuration {
      ipv4_enabled    = true
      private_network = google_service_networking_connection.private_vpc.network
      # Required for Cloud SQL Proxy from local machine; restrict in production
      dynamic "authorized_networks" {
        for_each = var.cloud_sql_authorized_networks
        content {
          name  = authorized_networks.value.name
          value = authorized_networks.value.value
        }
      }
    }

    database_flags {
      name  = "cloudsql.iam_authentication"
      value = "off"
    }
  }

  deletion_protection = false

  depends_on = [
    google_project_service.sqladmin,
    google_service_networking_connection.private_vpc,
  ]
}

resource "google_sql_database" "parking" {
  project  = var.project_id
  name     = "parking"
  instance = google_sql_database_instance.parking.name
}

resource "google_sql_user" "parking" {
  project  = var.project_id
  name     = "parking"
  instance = google_sql_database_instance.parking.name
  password = var.db_password
}

# Locals for Weaviate: use external URL or deployed Cloud Run
locals {
  weaviate_deploy = var.weaviate_url == ""
  weaviate_url    = var.weaviate_url != "" ? var.weaviate_url : google_cloud_run_v2_service.weaviate[0].uri
  weaviate_host   = var.weaviate_url != "" ? split("/", replace(replace(var.weaviate_url, "https://", ""), "http://", ""))[0] : replace(google_cloud_run_v2_service.weaviate[0].uri, "https://", "")
}

# Weaviate (Cloud Run) - vector store for RAG. Skipped when weaviate_url is set (use Weaviate Cloud).
resource "google_cloud_run_v2_service" "weaviate" {
  count = local.weaviate_deploy ? 1 : 0
  project  = var.project_id
  name     = "weaviate"
  location = var.region

  template {
    scaling {
      min_instance_count = 1
      max_instance_count = 1
    }

    containers {
      image = "docker.io/semitechnologies/weaviate:1.36.2"

      args = ["--host", "0.0.0.0", "--port", "8080", "--scheme", "http"]

      ports {
        container_port = 8080
      }

      startup_probe {
        tcp_socket {
          port = 8080
        }
        initial_delay_seconds = 90
        period_seconds       = 10
        timeout_seconds      = 5
        failure_threshold    = 36
      }

      env {
        name  = "QUERY_DEFAULTS_LIMIT"
        value = "25"
      }
      env {
        name  = "AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED"
        value = "true"
      }
      env {
        name  = "PERSISTENCE_DATA_PATH"
        value = "/var/lib/weaviate"
      }
      env {
        name  = "DEFAULT_VECTORIZER_MODULE"
        value = "none"
      }
      env {
        name  = "ENABLE_MODULES"
        value = "none"
      }
      env {
        name  = "CLUSTER_HOSTNAME"
        value = "node1"
      }
      env {
        name  = "RAFT_BOOTSTRAP_EXPECT"
        value = "1"
      }
      env {
        name  = "TRANSFORMERS_WAIT_FOR_STARTUP"
        value = "false"
      }
      env {
        name  = "CLIP_WAIT_FOR_STARTUP"
        value = "false"
      }
      env {
        name  = "GOMEMLIMIT"
        value = "1536MiB"
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "2Gi"
        }
        startup_cpu_boost = true
      }
    }
  }

  depends_on = [google_project_service.run]
}

resource "google_cloud_run_v2_service_iam_member" "weaviate_invoker" {
  count = local.weaviate_deploy ? 1 : 0

  project   = google_cloud_run_v2_service.weaviate[0].project
  location  = google_cloud_run_v2_service.weaviate[0].location
  name      = google_cloud_run_v2_service.weaviate[0].name
  role      = "roles/run.invoker"
  member    = "allUsers"
}

# MCP filesystem (Cloud Run) - reservation status storage
resource "google_cloud_run_v2_service" "mcp_filesystem" {
  project  = var.project_id
  name     = "mcp-filesystem"
  location = var.region

  template {
    scaling {
      min_instance_count = 1
      max_instance_count = 1
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.parking_bot.repository_id}/mcp-filesystem:latest"

      ports {
        container_port = 8080
      }

      startup_probe {
        tcp_socket {
          port = 8080
        }
        initial_delay_seconds = 60
        period_seconds       = 10
        timeout_seconds      = 5
        failure_threshold    = 24
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
      }
    }
  }

  depends_on = [google_project_service.run]
}

resource "google_cloud_run_v2_service_iam_member" "mcp_invoker" {
  project   = google_cloud_run_v2_service.mcp_filesystem.project
  location  = google_cloud_run_v2_service.mcp_filesystem.location
  name      = google_cloud_run_v2_service.mcp_filesystem.name
  role      = "roles/run.invoker"
  member    = "allUsers"
}

# VPC Access connector for Cloud Run -> Cloud SQL
resource "google_vpc_access_connector" "connector" {
  project       = var.project_id
  name          = "parking-connector"
  region        = var.region
  network       = "default"
  ip_cidr_range = "10.8.0.0/28"

  depends_on = [google_project_service.vpcaccess]
}

# Cloud Run service
resource "google_cloud_run_v2_service" "parking_bot" {
  project  = var.project_id
  name     = "parking-bot"
  location = var.region

  template {
    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    vpc_access {
      connector = google_vpc_access_connector.connector.id
      egress    = "PRIVATE_RANGES_ONLY"
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.parking_bot.repository_id}/parking-bot:${var.image_tag}"

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
      }

      env {
        name  = "TELEGRAM_BOT_TOKEN"
        value = var.telegram_bot_token
      }
      env {
        name  = "OPENAI_API_KEY"
        value = var.openai_api_key
      }
      env {
        name  = "TELEGRAM_ADMIN_CHAT_ID"
        value = var.telegram_admin_chat_id
      }
      env {
        name  = "POSTGRES_HOST"
        value = google_sql_database_instance.parking.private_ip_address
      }
      env {
        name  = "POSTGRES_PORT"
        value = "5432"
      }
      env {
        name  = "POSTGRES_DB"
        value = google_sql_database.parking.name
      }
      env {
        name  = "POSTGRES_USER"
        value = google_sql_user.parking.name
      }
      env {
        name  = "POSTGRES_PASSWORD"
        value = var.db_password
      }
      env {
        name  = "WEAVIATE_URL"
        value = local.weaviate_url
      }
      env {
        name  = "WEAVIATE_HTTP_HOST"
        value = local.weaviate_host
      }
      env {
        name  = "WEAVIATE_HTTP_PORT"
        value = "443"
      }
      env {
        name  = "WEAVIATE_HTTP_SECURE"
        value = "true"
      }
      env {
        name  = "WEAVIATE_GRPC_HOST"
        value = local.weaviate_host
      }
      env {
        name  = "WEAVIATE_GRPC_PORT"
        value = "443"
      }
      env {
        name  = "WEAVIATE_GRPC_SECURE"
        value = "true"
      }
      dynamic "env" {
        for_each = var.weaviate_api_key != "" ? [1] : []
        content {
          name  = "WEAVIATE_API_KEY"
          value = var.weaviate_api_key
        }
      }
      env {
        name  = "MCP_FILESYSTEM_URL"
        value = "${google_cloud_run_v2_service.mcp_filesystem.uri}/mcp"
      }
    }
  }

  depends_on = [
    google_project_service.run,
    google_vpc_access_connector.connector,
    google_cloud_run_v2_service.weaviate,
    google_cloud_run_v2_service.mcp_filesystem,
  ]
}

# Allow unauthenticated access (for Telegram webhook or long-polling; adjust for production)
resource "google_cloud_run_v2_service_iam_member" "public" {
  project   = google_cloud_run_v2_service.parking_bot.project
  location  = google_cloud_run_v2_service.parking_bot.location
  name      = google_cloud_run_v2_service.parking_bot.name
  role      = "roles/run.invoker"
  member    = "allUsers"
}
