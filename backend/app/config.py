from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_BACKEND_DIR / ".env", extra="ignore")
    llm_provider: Literal["colibri", "ollama"] = "colibri"
    colibri_base_url: str = "http://127.0.0.1:8001/v1"
    colibri_model: str = "qwen3.6-colibri"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"
    world_file: Path | None = None
    # Legacy JSON stores: imported into memory_db_file once, then never read or written.
    memory_file: Path = _BACKEND_DIR / "memory" / "persistent.json"
    graph_file: Path = _BACKEND_DIR / "memory" / "beliefs.json"
    llm_timeout_seconds: float = Field(default=120, gt=0)
    task_timeout_seconds: float = Field(default=1800, gt=0)
    temperature: float = Field(default=0.1, ge=0, le=1)
    context_tokens: int = Field(default=8192, ge=2048, le=32768)
    decision_thinking: bool = False
    direct_initial_delegation: bool = False
    digital_tools_enabled: bool = True
    decision_output_tokens: int = Field(default=4096, ge=1024, le=16384)
    max_coordinator_cycles: int = Field(default=20, ge=1, le=100)
    # A hard safety ceiling, not the normal way a delegation ends. Healthy work
    # stops on completion, blockage or semantic stall long before this.
    max_explorer_actions: int = Field(default=40, ge=1, le=200)
    max_tool_calls: int = Field(default=80, ge=1, le=400)
    max_critic_reviews: int = Field(default=8, ge=1, le=20)
    # Physical actions yielding no semantic progress before the strategy is replanned.
    max_semantic_stall: int = Field(default=3, ge=1, le=20)
    # Reasoning budgets. They stop loops that make no progress; long tasks that keep
    # producing new evidence are bounded only by the wall-clock timeout.
    max_task_llm_calls: int = Field(default=60, ge=5, le=500)
    max_consecutive_rejections: int = Field(default=6, ge=2, le=50)
    max_cycles_without_progress: int = Field(default=5, ge=2, le=50)
    max_critic_disagreements: int = Field(default=3, ge=1, le=20)
    # Actions after which a room scan is treated as stale and worth repeating.
    search_stale_after: int = Field(default=12, ge=1, le=200)
    log_level: str = "INFO"
    # Simulated world clock
    simulation_mode: Literal["realtime", "action_driven"] = "realtime"
    clock_speed: float = Field(default=60.0, gt=0)   # 1 real second = 1 simulated minute
    clock_start_hour: float = Field(default=8.0, ge=0, lt=24)
    action_seconds: float = Field(default=60.0, gt=0)  # simulated seconds per action
    # Idle autonomous loop
    idle_reflection_enabled: bool = True
    idle_tick_seconds: float = Field(default=5.0, gt=0)
    idle_reflection_interval: float = Field(default=120.0, gt=0)
    intention_threshold: float = Field(default=0.6, ge=0, le=1)
    drive_helpfulness: float = Field(default=0.8, ge=0, le=1)
    # Conversation with Kip. Leave chat_* unset to chat with the main model; set e.g.
    # CHAT_PROVIDER=ollama CHAT_MODEL=qwen3:4b for a fast, separate chat model.
    chat_provider: Literal["colibri", "ollama"] | None = None
    chat_model: str | None = None
    chat_base_url: str | None = None
    chat_max_tokens: int = Field(default=300, ge=50, le=2000)
    chat_temperature: float = Field(default=0.6, ge=0, le=1.5)
    # confirm: errands start only when asked or after a yes to Kip's offer.
    # act: any errand the chat model proposes starts immediately (less safe).
    chat_autonomy: Literal["confirm", "act"] = "confirm"
    # Seconds after the last chat activity before the idle mind may use the model again.
    chat_quiet_seconds: float = Field(default=90.0, ge=0)
    # How long an offer Kip made can still be accepted with a plain "yes".
    chat_offer_seconds: float = Field(default=600.0, gt=0)
    # Person id assumed to be talking when the UI does not say (empty = unknown speaker).
    chat_default_speaker: str = ""
    # Idle initiative: may the idle mind start errands on its own?
    autonomous_errands: bool = True
    # Kip's durable store: searchable memories (FTS5 + sqlite-vec) and structured state
    memory_db_file: Path = _BACKEND_DIR / "memory" / "kip.db"
    embeddings_enabled: bool = True
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_cache_dir: Path = _BACKEND_DIR / "memory" / "models"
    # Persistent personality layer (no extra model calls)
    persona_enabled: bool = True
    persona_name: str = Field(default="Kip", min_length=1, max_length=40)


@lru_cache
def get_settings() -> Settings:
    return Settings()
