from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_BACKEND_DIR / ".env", extra="ignore")
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"
    world_file: Path | None = None
    memory_file: Path = _BACKEND_DIR / "memory" / "persistent.json"
    graph_file: Path = _BACKEND_DIR / "memory" / "beliefs.json"
    llm_timeout_seconds: float = Field(default=120, gt=0)
    task_timeout_seconds: float = Field(default=1800, gt=0)
    temperature: float = Field(default=0.1, ge=0, le=1)
    context_tokens: int = Field(default=8192, ge=2048, le=32768)
    decision_thinking: bool = False
    decision_output_tokens: int = Field(default=4096, ge=1024, le=16384)
    max_coordinator_cycles: int = Field(default=20, ge=1, le=100)
    # A hard safety ceiling, not the normal way a delegation ends. Healthy work
    # stops on completion, blockage or semantic stall long before this.
    max_explorer_actions: int = Field(default=40, ge=1, le=200)
    max_tool_calls: int = Field(default=80, ge=1, le=400)
    max_critic_reviews: int = Field(default=8, ge=1, le=20)
    # Physical actions yielding no semantic progress before the strategy is replanned.
    max_semantic_stall: int = Field(default=3, ge=1, le=20)
    # Actions after which a room scan is treated as stale and worth repeating.
    search_stale_after: int = Field(default=12, ge=1, le=200)
    log_level: str = "INFO"
    # Simulated world clock
    simulation_mode: Literal["realtime", "action_driven"] = "realtime"
    clock_speed: float = Field(default=60.0, gt=0)   # 1 real second = 1 simulated minute
    clock_start_hour: float = Field(default=8.0, ge=0, lt=24)
    action_seconds: float = Field(default=60.0, gt=0)  # simulated seconds per action
    # Idle autonomous loop
    idle_tick_seconds: float = Field(default=5.0, gt=0)
    idle_reflection_interval: float = Field(default=120.0, gt=0)
    intention_threshold: float = Field(default=0.6, ge=0, le=1)
    drive_helpfulness: float = Field(default=0.8, ge=0, le=1)
    distillation_task_interval: int = Field(default=5, ge=1)  # run distillation every N completed tasks


@lru_cache
def get_settings() -> Settings:
    return Settings()
