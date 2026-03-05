output "artifact_registry_repository" {
  description = "Artifact Registry repository for parking-bot images"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.parking_bot.repository_id}"
}

output "cloud_run_url" {
  description = "Cloud Run service URL"
  value       = google_cloud_run_v2_service.parking_bot.uri
}

output "cloud_sql_private_ip" {
  description = "Cloud SQL private IP (for reference)"
  value       = google_sql_database_instance.parking.private_ip_address
}

output "cloud_sql_connection_name" {
  description = "Cloud SQL connection name for Cloud SQL Proxy (PROJECT:REGION:INSTANCE)"
  value       = google_sql_database_instance.parking.connection_name
}

output "weaviate_url" {
  description = "Weaviate URL (Cloud Run or external, for data loader)"
  value       = local.weaviate_url
}

output "mcp_filesystem_url" {
  description = "MCP filesystem Cloud Run URL"
  value       = "${google_cloud_run_v2_service.mcp_filesystem.uri}/mcp"
}
