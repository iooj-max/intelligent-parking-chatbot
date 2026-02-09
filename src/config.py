from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    openai_api_key: str = ""

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
