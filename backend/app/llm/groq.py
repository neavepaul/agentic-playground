"""Groq's hosted OpenAI-compatible API, for a few bounded, high-value calls.

Colibri stays the engine for conversation and the task loop. Groq's free tier
limits each model separately, so a pool of models is tried in preference order and
a model whose quota is spent is skipped until its reset time. Each model is tried
at most once per call: nothing here loops, waits or retries.

Requests are plain chat completions: the system, user and history messages as given,
with no JSON mode, tools or reasoning parameters. Callers validate any JSON locally.
"""
import logging
import re
import time
from dataclasses import dataclass

import httpx

from app.config import Settings
from .base import ModelError

_log = logging.getLogger("agentic_friend.llm.groq")
_DURATION = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_UNIT_SECONDS = {"h": 3600, "m": 60, "s": 1, "ms": 0.001}
# Reasoning that some models (qwen) write inline, and a fence around the whole reply.
_THINK = re.compile(r"<think>.*?</think>\s*", re.S)
_FENCED = re.compile(r"^```[\w-]*\n(.*)\n```$", re.S)


def reset_seconds(value: str | None) -> float | None:
    """Groq's reset headers ("7.66s", "2m59.56s", "1h2m", "240ms") as seconds."""
    if not value:
        return None
    parts = _DURATION.findall(value.strip())
    if not parts or "".join(number + unit for number, unit in parts) != value.strip():
        return None
    return sum(float(number) * _UNIT_SECONDS[unit] for number, unit in parts)


@dataclass
class ModelQuota:
    """What Groq last reported for one model. Unknown means "assume available"."""
    blocked_until: float = 0.0
    remaining_requests: int | None = None
    requests_reset_at: float = 0.0
    remaining_tokens: int | None = None
    tokens_reset_at: float = 0.0

    def available(self, need_tokens: int, now: float) -> bool:
        if now < self.blocked_until:
            return False
        if self.remaining_requests is not None and self.remaining_requests < 1 and now < self.requests_reset_at:
            return False
        if self.remaining_tokens is not None and self.remaining_tokens < need_tokens and now < self.tokens_reset_at:
            return False
        return True

    def observe(self, headers: httpx.Headers, now: float) -> None:
        def number(name):
            try:
                return int(headers[name])
            except (KeyError, ValueError):
                return None
        requests, tokens = number("x-ratelimit-remaining-requests"), number("x-ratelimit-remaining-tokens")
        if requests is not None:
            self.remaining_requests = requests
            self.requests_reset_at = now + (reset_seconds(headers.get("x-ratelimit-reset-requests")) or 60)
        if tokens is not None:
            self.remaining_tokens = tokens
            self.tokens_reset_at = now + (reset_seconds(headers.get("x-ratelimit-reset-tokens")) or 60)


def estimate_tokens(messages: list[dict]) -> int:
    # Conservative: ~4 characters per token, plus room for the reply.
    return sum(len(message.get("content") or "") for message in messages) // 3 + 1024


class GroqClient:
    def __init__(self, settings: Settings, now=time.monotonic) -> None:
        if settings.groq_api_key is None:
            raise ValueError("GROQ_API_KEY is not set.")
        self.settings = settings
        self.models = [model.strip() for model in settings.groq_models.split(",") if model.strip()]
        self.quota = {model: ModelQuota() for model in self.models}
        self.now = now
        self.last_metrics: dict = {}
        self.http = httpx.AsyncClient(
            base_url=settings.groq_base_url.rstrip("/") + "/", timeout=settings.groq_timeout_seconds,
            headers={"Authorization": f"Bearer {settings.groq_api_key.get_secret_value()}"})

    def candidates(self, need_tokens: int) -> list[str]:
        now = self.now()
        return [model for model in self.models if self.quota[model].available(need_tokens, now)]

    def has_capacity(self, messages: list[dict] | None = None) -> bool:
        return bool(self.candidates(estimate_tokens(messages or [])))

    async def generate(self, messages: list[dict], response_schema: dict) -> str:
        return await self._complete(estimate_tokens(messages), lambda model: {
            "model": model, "messages": messages, "temperature": self.settings.temperature,
            "max_completion_tokens": self.settings.groq_max_output_tokens})

    async def _complete(self, need: int, body) -> str:
        """The reply from the first model in preference order that can answer."""
        self.last_metrics = {}
        tried = []
        for model in self.candidates(need):
            tried.append(model)
            started = time.monotonic()
            try:
                response = await self.http.post("chat/completions", json=body(model))
            except httpx.TimeoutException:
                raise ModelError("Groq timed out.") from None
            except httpx.RequestError:
                raise ModelError("Cannot reach Groq.") from None
            now = self.now()
            self.quota[model].observe(response.headers, now)
            status = response.status_code
            if status in (401, 403):
                raise ModelError("Groq rejected the API key. Check GROQ_API_KEY.")
            if status == 429:
                wait = reset_seconds(response.headers.get("retry-after")) or _seconds(response.headers.get("retry-after"))
                self.quota[model].blocked_until = now + (wait or 60)
                _log.info("groq model=%s rate-limited for %.0fs; trying the next model", model, wait or 60)
                continue
            code = _error_code(response) if status == 400 else ""
            if status == 400 and code != "tool_use_failed":  # a made-up tool call is one bad generation
                # A request this model refuses (e.g. an unknown model) will keep failing,
                # so it sits out for an hour instead of costing a call each time.
                self.quota[model].blocked_until = now + 3600
                _log.warning("groq model=%s refused the request (HTTP 400 %s); skipping it for an hour",
                             model, code or "no code")
                continue
            if status >= 400:
                _log.info("groq model=%s returned HTTP %d; trying the next model", model, status)
                continue
            try:
                envelope = response.json()
                choice = envelope["choices"][0]
                content = choice["message"]["content"]
                if isinstance(content, str):
                    content = _THINK.sub("", content).strip()
                    content = _FENCED.sub(r"\1", content).strip()
                if not isinstance(content, str) or not content:
                    raise ValueError("Missing content")
            except (ValueError, KeyError, TypeError, IndexError):
                _log.info("groq model=%s returned an unexpected envelope; trying the next model", model)
                continue
            if choice.get("finish_reason") == "length":
                _log.info("groq model=%s hit its output limit; trying the next model", model)
                continue
            usage = envelope.get("usage") or {}
            self.last_metrics = {"provider": "groq", "model": envelope.get("model") or model,
                                 "groq_seconds": round(time.monotonic() - started, 3),
                                 **{key: usage[key] for key in ("prompt_tokens", "completion_tokens")
                                    if isinstance(usage.get(key), int)}}
            return content
        reason = "every model failed" if tried else "no model has quota left"
        raise ModelError(f"Groq could not answer: {reason}.")

    async def close(self) -> None:
        await self.http.aclose()


def _seconds(value: str | None) -> float | None:
    try:
        return float(value) if value else None
    except ValueError:
        return None


def _error_code(response: httpx.Response) -> str:
    try:
        return str(response.json()["error"].get("code") or "")
    except (ValueError, KeyError, TypeError, AttributeError):
        return ""


class HostedFirst:
    """Up to `limit` Groq calls for one purpose, then the local model.

    The local model also takes over at once whenever Groq cannot serve a call
    (no quota, an error, a timeout), so the task never waits on hosted capacity.
    """

    def __init__(self, hosted: GroqClient, local, limit: int, purpose: str) -> None:
        self.hosted, self.local, self.limit, self.purpose = hosted, local, limit, purpose
        self.used = 0
        self.last_metrics: dict = {}

    async def generate(self, messages: list[dict], response_schema: dict) -> str:
        if self.used < self.limit and self.hosted.has_capacity(messages):
            self.used += 1
            try:
                raw = await self.hosted.generate(messages, response_schema)
                self.last_metrics = dict(self.hosted.last_metrics)
                return raw
            except ModelError as exc:
                _log.warning("Groq unavailable for %s (%s). Using the local model.", self.purpose, exc)
        raw = await self.local.generate(messages, response_schema)
        self.last_metrics = dict(getattr(self.local, "last_metrics", None) or {})
        return raw
