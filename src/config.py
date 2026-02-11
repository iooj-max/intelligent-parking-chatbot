from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    openai_api_key: str

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
    langsmith_api_key: str = ""
    langsmith_tracing: bool = True
    langsmith_project: str = "intelligent-parking-chatbot"

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "parking"
    postgres_user: str = "parking"
    postgres_password: str = "parking"

    # Weaviate
    weaviate_url: str = "http://localhost:8080"

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
