from pathlib import Path
from urllib.parse import quote_plus

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # LLM
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    parking_agent_model: str = Field(default="gpt-4o-mini", validation_alias="PARKING_AGENT_MODEL")
    telegram_bot_token: str = Field(default="", validation_alias="TELEGRAM_BOT_TOKEN")
    telegram_admin_chat_id: str = Field(default="", validation_alias="TELEGRAM_ADMIN_CHAT_ID")

    @field_validator('openai_api_key')
    @classmethod
    def validate_openai_key(cls, v: str) -> str:
        """Validate OpenAI API key is present and correctly formatted."""
        if not v or v.strip() == "":
            raise ValueError("OpenAI API key is required")
        if not v.startswith("sk-"):
            raise ValueError("Invalid OpenAI API key format (must start with 'sk-')")
        return v

    # LangSmith
    langsmith_api_key: str = Field(default="", validation_alias="LANGSMITH_API_KEY")
    langsmith_tracing: bool = Field(default=True, validation_alias="LANGSMITH_TRACING")
    langsmith_project: str = Field(default="intelligent-parking-chatbot", validation_alias="LANGSMITH_PROJECT")

    # PostgreSQL
    postgres_host: str = Field(default="localhost", validation_alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, validation_alias="POSTGRES_PORT")
    postgres_db: str = Field(default="parking", validation_alias="POSTGRES_DB")
    postgres_user: str = Field(default="parking", validation_alias="POSTGRES_USER")
    postgres_password: str = Field(default="parking", validation_alias="POSTGRES_PASSWORD")

    # Weaviate
    weaviate_url: str = Field(default="http://localhost:8080", validation_alias="WEAVIATE_URL")
    weaviate_http_host: str = Field(default="localhost", validation_alias="WEAVIATE_HTTP_HOST")
    weaviate_http_port: int = Field(default=8080, validation_alias="WEAVIATE_HTTP_PORT")
    weaviate_http_secure: bool = Field(default=False, validation_alias="WEAVIATE_HTTP_SECURE")
    weaviate_grpc_host: str = Field(default="localhost", validation_alias="WEAVIATE_GRPC_HOST")
    weaviate_grpc_port: int = Field(default=50051, validation_alias="WEAVIATE_GRPC_PORT")
    weaviate_grpc_secure: bool = Field(default=False, validation_alias="WEAVIATE_GRPC_SECURE")
    weaviate_collection: str = Field(default="ParkingContent", validation_alias="WEAVIATE_COLLECTION")
    weaviate_top_k: int = Field(default=5, validation_alias="WEAVIATE_TOP_K")
    weaviate_candidate_k: int = Field(default=20, validation_alias="WEAVIATE_CANDIDATE_K")
    weaviate_max_chunks_per_source: int = Field(default=1, validation_alias="WEAVIATE_MAX_CHUNKS_PER_SOURCE")
    weaviate_query_alpha: float = Field(default=0.5, validation_alias="WEAVIATE_QUERY_ALPHA")

    @property
    def postgres_dsn(self) -> str:
        encoded_user = quote_plus(self.postgres_user)
        encoded_password = quote_plus(self.postgres_password)
        return (
            f"postgresql+psycopg://{encoded_user}:{encoded_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()