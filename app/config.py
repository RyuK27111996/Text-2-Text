from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and `.env`."""

    app_name: str = "fastapi-gemma-api"
    app_env: str = "dev"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_debug: bool = False

    gemini_api_key: str | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com"
    gemini_api_version: str = "v1beta"
    gemini_default_model: str = "gemma-3-27b-it"

    request_timeout_seconds: float = 30.0
    max_input_chars: int = 16000
    max_output_tokens_limit: int = 2048
    default_max_output_tokens: int = 512
    max_concurrent_provider_calls: int = 10
    gemini_max_retries: int = 2
    gemini_retry_base_delay_seconds: float = 0.5

    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )




@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings instance."""
    return Settings()
