from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[1] / ".env", extra="ignore")
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    world_file: Path | None = None
    llm_timeout_seconds: float = Field(default=120, gt=0)
    task_timeout_seconds: float = Field(default=1800, gt=0)
    temperature: float = Field(default=0.1, ge=0, le=1)
    context_tokens: int = Field(default=8192, ge=2048, le=32768)
    max_coordinator_cycles: int = Field(default=20, ge=1, le=100)
    max_explorer_actions: int = Field(default=8, ge=1, le=50)
    max_tool_calls: int = Field(default=50, ge=1, le=200)
    max_critic_reviews: int = Field(default=8, ge=1, le=20)
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
