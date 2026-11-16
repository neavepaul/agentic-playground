from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_BACKEND_DIR / ".env", extra="ignore")
    # The one local model, built for a Raspberry Pi 4 (8 GB): Ollama with a 4-bit Qwen3.5 2B.
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:2b-q4_K_M"
    # Groq: a hosted model for a few bounded, high-value calls (goal planning and
    # web-research tasks). Free-tier limits are per model, so several are listed and
    # the first with quota left is used. Without a key, everything stays on the local model.
    groq_api_key: SecretStr | None = None
    # CricketData.org (CricAPI): cricket results, live scores and fixtures for a week or more either
    # side, 100 requests a day free. Unset: ESPN's scoreboard (no key, unofficial, a few days) only.
    cricketdata_api_key: SecretStr | None = None
    # football-data.org: ten days of matches across the competitions the key covers, 10 requests a minute
    # free. Unset: ESPN's league scoreboards (this round only) answer football.
    footballdata_api_key: SecretStr | None = None
    # Groww Trade API, a TOTP key: "how is my portfolio doing" reads their holdings (read-only; Kip never calls
    # order, funds or positions endpoints). The secret is the TOTP seed. Unset: no portfolio answers.
    groww_api_key: SecretStr | None = None
    groww_api_secret: SecretStr | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_models: str = "openai/gpt-oss-120b,qwen/qwen3.8-27b,openai/gpt-oss-20b"
    groq_timeout_seconds: float = Field(default=60, gt=0)
    # When every Groq model is rate-limited and one frees within this long, a task call waits
    # for it (once) instead of falling back to the local model, which can take minutes.
    groq_max_wait_seconds: float = Field(default=20, ge=0, le=120)
    # A chat reply waits for Groq this long in all (repeatedly, if it is limited again) before the
    # local model takes it: a Pi's first word from a 1,000-token prompt takes minutes.
    groq_chat_wait_seconds: float = Field(default=30, ge=0, le=120)
    # Per-minute tokens an errand call leaves free on the models chat also uses, so a long lookup
    # (several 4,000-token calls) cannot leave the next reply to the local model. 0 turns it off.
    groq_chat_reserve_tokens: int = Field(default=3000, ge=0, le=32000)
    groq_max_output_tokens: int = Field(default=4096, ge=256, le=32768)
    # Hard caps: Groq calls per goal plan, and per web-research task. Past a cap, or
    # whenever Groq cannot serve a call, the local model carries on.
    groq_plan_calls: int = Field(default=2, ge=0, le=4)
    groq_research_calls: int = Field(default=4, ge=0, le=10)
    # Conversation on Groq while it has quota, streamed, falling back to the local chat model.
    # Qwen first: web search runs only on the gpt-oss models, so chat leaves their quota alone.
    chat_hosted: bool = True
    # How hard Groq's browser search thinks. Each extra step re-reads every page read so far.
    groq_search_reasoning: Literal["low", "medium", "high"] = "low"
    groq_chat_models: str = "qwen/qwen3.8-27b,openai/gpt-oss-20b"
    # How long Ollama keeps the model loaded after a call: -1 always (a Pi reloading ~2 GB from a
    # microSD card takes tens of seconds), "30m" and so on, or empty for Ollama's 5 minutes.
    ollama_keep_alive: str = "-1"
    # A small model on a Pi can take minutes over a long call.
    llm_timeout_seconds: float = Field(default=600, gt=0)
    task_timeout_seconds: float = Field(default=1800, gt=0)
    temperature: float = Field(default=0.1, ge=0, le=1)
    # Ollama's context window: smaller needs less memory and is quicker to set up per call.
    context_tokens: int = Field(default=4096, ge=2048, le=32768)
    decision_thinking: bool = False
    digital_tools_enabled: bool = True
    decision_output_tokens: int = Field(default=4096, ge=1024, le=16384)
    max_coordinator_cycles: int = Field(default=20, ge=1, le=100)
    # A hard safety ceiling, not the normal way a delegation ends. Healthy work
    # stops on completion or blockage long before this.
    max_explorer_actions: int = Field(default=40, ge=1, le=200)
    max_tool_calls: int = Field(default=80, ge=1, le=400)
    max_critic_reviews: int = Field(default=8, ge=1, le=20)
    # Reasoning budgets. They stop loops that make no progress; long tasks that keep
    # producing new evidence are bounded only by the wall-clock timeout.
    max_task_llm_calls: int = Field(default=60, ge=5, le=500)
    max_consecutive_rejections: int = Field(default=6, ge=2, le=50)
    max_cycles_without_progress: int = Field(default=5, ge=2, le=50)
    max_critic_disagreements: int = Field(default=3, ge=1, le=20)
    log_level: str = "INFO"
    # Every time Kip says or reasons with is in this zone, whatever the device's clock is set to
    # (a Raspberry Pi often runs on UTC). An IANA name.
    timezone: str = "Asia/Kolkata"
    # A spoken reply plus its action line; a Pi writes only a few tokens a second.
    chat_max_tokens: int = Field(default=300, ge=50, le=16384)
    # One ongoing conversation. Separate IDs at the ChatSession boundary isolate
    # transcripts; they are not user authentication.
    chat_session_id: str = Field(default="home", min_length=1, max_length=100)
    # Who may talk to the API. There is no login, so Kip trusts whoever reaches it: keep it on
    # loopback. Host names the server answers to (a page on another site that points its own name at
    # this machine is refused), and web pages allowed to open the WebSocket or send changes, comma-separated.
    allowed_hosts: str = "127.0.0.1,localhost"
    allowed_origins: str = "http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:8000,http://localhost:8000"
    # false: a reply written by the local model keeps its memories, but its task and offer are dropped;
    # errands then start only from the person's own words (tool requests, live questions, a yes to an
    # offer). A 2B model's own errands can be nonsense; Groq-written replies are unaffected.
    local_chat_errands: bool = False
    # Earlier turns each reply carries: less is faster on a Pi, more remembers the thread better.
    # A long message itself is never refused for it.
    chat_history_tokens: int = Field(default=300, ge=0, le=8192)
    # When a long chat outgrows that, fold its older turns into a short note (one background model
    # call, never blocking a reply) that rides in each prompt. false: older turns are only dropped.
    chat_compaction: bool = True
    chat_checkpoint_tokens: int = Field(default=200, ge=50, le=1024)
    # What is worth remembering is decided after each reply that could hold a lasting fact, in its own
    # short model call, so the chat model only talks. false: no new facts are learned from chat unless
    # the model volunteers them in its action line.
    chat_extraction: bool = True
    # When Groq answers chat it can read far more than the Pi's 4K window, but every token counts
    # against its per-minute quota, so the extra is modest. The local model keeps CONTEXT_TOKENS and
    # CHAT_HISTORY_TOKENS; a reply that falls back to it is rebuilt at that size. Set both equal to the
    # local values to give Groq the same prompt.
    chat_hosted_context_tokens: int = Field(default=8192, ge=2048, le=131072)
    chat_hosted_history_tokens: int = Field(default=1600, ge=0, le=16384)
    chat_temperature: float = Field(default=0.6, ge=0, le=1.5)
    # confirm: errands start only when asked or after a yes to Kip's offer.
    # act: any errand the chat model proposes starts immediately (less safe).
    chat_autonomy: Literal["confirm", "act"] = "confirm"
    # How long an offer Kip made can still be accepted with a plain "yes".
    chat_offer_seconds: float = Field(default=600.0, gt=0)
    # Who is talking when nobody has said (e.g. "neave"); empty = unknown until they give a name.
    chat_default_speaker: str = ""
    # Local intent router (app/chat/router.py). shadow: it reads every message and records what
    # it would have done next to what the usual path did, changing nothing. on: it decides which
    # live-information questions go straight to the web lookup, and resolves follow-ups.
    intent_router: Literal["off", "shadow", "on"] = "on"
    # Kip's durable store: searchable memories (FTS5 + sqlite-vec) and structured state
    memory_db_file: Path = _BACKEND_DIR / "memory" / "kip.db"
    # The built page the backend serves at / when present (pnpm build); none in development (Vite).
    frontend_dist: Path = _BACKEND_DIR.parent / "frontend" / "dist"
    embeddings_enabled: bool = True
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_cache_dir: Path = _BACKEND_DIR / "memory" / "models"
    # Persistent personality layer (no extra model calls)
    persona_enabled: bool = True
    persona_name: str = Field(default="Kip", min_length=1, max_length=40)
    # Who built Kip, as he should know it ("Neave"). Empty: he says he does not know.
    persona_maker: str = Field(default="", max_length=80)


@lru_cache
def get_settings() -> Settings:
    return Settings()
