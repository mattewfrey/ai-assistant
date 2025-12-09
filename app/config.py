from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration sourced from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    env: str = Field(default="dev", alias="APP_ENV")
    debug: bool = Field(default=False, alias="APP_DEBUG")

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: Optional[str] = Field(default=None, alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    openai_temperature: float = Field(default=0.2, alias="OPENAI_TEMPERATURE")
    assistant_min_confidence: float = Field(default=0.6, alias="ASSISTANT_MIN_CONFIDENCE")

    platform_base_url: str = Field(default="https://platform.local", alias="PLATFORM_BASE_URL")
    http_timeout_seconds: float = Field(default=15.0, alias="HTTP_TIMEOUT_SECONDS")

    enable_data_explanations: bool = Field(default=False, alias="ENABLE_DATA_EXPLANATIONS")
    use_langchain: bool = Field(default=False, alias="USE_LANGCHAIN")
    langchain_tracing: bool = Field(default=False, alias="LANGCHAIN_TRACING")
    langchain_cache: bool = Field(default=False, alias="LANGCHAIN_CACHE")

    assistant_system_prompt_version: str = Field(default="v1")

    # LangSmith / LangChain tracing
    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_endpoint: str | None = Field(default=None, alias="LANGSMITH_ENDPOINT")
    langsmith_project: str | None = Field(default=None, alias="LANGSMITH_PROJECT")
    langsmith_tracing_v2: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")
    enable_beautify_reply: bool = Field(default=True, alias="ENABLE_BEAUTIFY_REPLY")
    enable_request_tracing: bool = Field(default=True, alias="ENABLE_REQUEST_TRACING")
    enable_local_router: bool = Field(default=True, alias="ENABLE_LOCAL_ROUTER")

    # Rate limiting settings
    llm_rate_limit_window_seconds: int = Field(default=60, alias="LLM_RATE_LIMIT_WINDOW")
    llm_rate_limit_max_calls: int = Field(default=20, alias="LLM_RATE_LIMIT_MAX_CALLS")

    # Price validation bounds
    price_max_upper_bound: float = Field(default=500_000.0, alias="PRICE_MAX_UPPER_BOUND")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
