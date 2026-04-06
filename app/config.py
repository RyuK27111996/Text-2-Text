from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "fastapi-gemini-async-api"
    app_env: str = "dev"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_debug: bool = False

    gemini_api_key: str | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com"
    gemini_api_version: str = "v1beta"
    gemini_default_model: str = "gemini-2.5-flash"

    # Ollama (local) settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_default_model: str = "gemma3"

    request_timeout_seconds: float = 30.0
    max_input_chars: int = 16000
    max_output_tokens_limit: int = 2048
    default_max_output_tokens: int = 512
    max_concurrent_provider_calls: int = 10

    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )




@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
