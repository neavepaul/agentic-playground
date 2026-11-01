"""Groq's hosted OpenAI-compatible API, for a few bounded, high-value calls.

Colibri stays the engine for conversation and the task loop. Groq's free tier
limits each model separately, so a pool of models is tried in preference order and
a model whose quota is spent is skipped until its reset time. Each model is tried
at most once per call: nothing here loops, waits or retries.

Requests are plain chat completions: the system, user and history messages as given,
with no JSON mode, tools or reasoning parameters. Callers validate any JSON locally.
The one exception is search(): Groq's built-in browser_search, on the gpt-oss models
that support it, which looks something up on the web and writes the answer in one call.
"""
import json
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
# browser_search answers cite what they read inline, like 【2†L26-L28】.
_CITATION = re.compile(r"\s*【[^】]*】")
# The models browser_search runs on, and the room its search results take in the prompt.
SEARCH_MODELS = ("openai/gpt-oss",)
SEARCH_TOKENS = 6000
# A 400 that is one bad generation, not a refused request: the same model gets one more try.
_BAD_GENERATION = {"output_parse_failed", "tool_use_failed"}


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

    def candidates(self, need_tokens: int, searching: bool = False) -> list[str]:
        now = self.now()
        return [model for model in self.models if self.quota[model].available(need_tokens, now)
                and (not searching or model.startswith(SEARCH_MODELS))]

    def has_capacity(self, messages: list[dict] | None = None) -> bool:
        return bool(self.candidates(estimate_tokens(messages or [])))

    def can_search(self, messages: list[dict] | None = None) -> bool:
        return bool(self.candidates(estimate_tokens(messages or []) + SEARCH_TOKENS, searching=True))

    async def generate(self, messages: list[dict], response_schema: dict) -> str:
        message = await self._complete(estimate_tokens(messages), lambda model: {
            "model": model, "messages": messages, "temperature": self.settings.temperature,
            "max_completion_tokens": self.settings.groq_max_output_tokens})
        return message["content"]

    async def search(self, messages: list[dict]) -> "WebAnswer":
        """Groq searches the web itself and answers from what it found.

        Sources are the pages it opened, else its top search results, at most three. The
        reply's closing SOURCE: line is split off, and everything it read is kept as evidence,
        so the caller can check the answer against it. An answer with nothing searched behind
        it has no sources, and is no evidence."""
        message = await self._complete(estimate_tokens(messages) + SEARCH_TOKENS, lambda model: {
            "model": model, "messages": messages, "temperature": self.settings.chat_temperature,
            "max_completion_tokens": self.settings.groq_max_output_tokens,
            # Low effort: fewer browsing steps, and each step re-reads everything read so far.
            "reasoning_effort": self.settings.groq_search_reasoning, "include_reasoning": False,
            "tools": [{"type": "browser_search"}]}, searching=True)
        from app.capabilities.grounding import split_source
        answer, quote = split_source(_CITATION.sub("", message["content"]))
        tools = [t for t in message.get("executed_tools") or [] if isinstance(t, dict)]
        found = WebAnswer(answer=" ".join(answer.split()), quote=" ".join(quote.split()),
                          sources=search_sources(tools), evidence=search_evidence(tools),
                          searches=sum("open" not in f"{t.get('type', '')} {t.get('name', '')}" for t in tools),
                          pages=sum("open" in f"{t.get('type', '')} {t.get('name', '')}" for t in tools))
        # Most of the prompt tokens are Groq's own browsing: each result page it reads is fed back in.
        _log.info("groq browser_search model=%s searches=%d pages_opened=%d read_chars=%d own_prompt_chars=%d "
                  "prompt_tokens=%s completion_tokens=%s", self.last_metrics.get("model"), found.searches,
                  found.pages, len(found.evidence), sum(len(m.get("content") or "") for m in messages),
                  self.last_metrics.get("prompt_tokens"), self.last_metrics.get("completion_tokens"))
        self.last_metrics.update(searches=found.searches, pages_opened=found.pages, read_chars=len(found.evidence))
        return found

    async def _complete(self, need: int, body, searching: bool = False) -> dict:
        """The reply message from the first model in preference order that can answer."""
        self.last_metrics = {}
        tried, retried = [], set()
        queue = self.candidates(need, searching)
        while queue:
            model = queue.pop(0)
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
            if status == 400 and code in _BAD_GENERATION and model not in retried:
                retried.add(model)
                queue.insert(0, model)
                _log.info("groq model=%s produced output it could not parse (%s); trying it once more", model, code)
                continue
            if status == 400 and code not in _BAD_GENERATION and not searching:  # a made-up tool call is one bad generation
                # A request this model refuses (e.g. an unknown model) will keep failing,
                # so it sits out for an hour instead of costing a call each time. A refused
                # search says nothing about plain calls, so it never benches the model.
                self.quota[model].blocked_until = now + 3600
                _log.warning("groq model=%s refused the request (HTTP 400 %s); skipping it for an hour",
                             model, code or "no code")
                continue
            if status >= 400:
                _log.info("groq model=%s returned HTTP %d (%s); trying the next model", model, status,
                          _error_detail(response))
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
            return {**choice["message"], "content": content}
        reason = "every model failed" if tried else "no model has quota left"
        raise ModelError(f"Groq could not answer: {reason}.")

    def chat_candidates(self, models: list[str], messages: list[dict], max_tokens: int) -> list[str]:
        # A spoken reply is a few dozen tokens: reserve for that, not the 1,200-token ceiling, or a
        # model's per-minute quota looks spent after two replies.
        now, need = self.now(), estimate_tokens(messages) - 1024 + min(max_tokens, 300)
        return [m for m in models if self.quota.setdefault(m, ModelQuota()).available(need, now)]

    async def stream_chat(self, messages: list[dict], models: list[str], max_tokens: int,
                          temperature: float | None = None):
        """The reply's text as Groq streams it, from the first of `models` that can answer.

        A model that fails before any text arrives (no quota, an error) hands over to the
        next; once text has been spoken, a failure is raised, never papered over."""
        self.last_metrics = {}
        for model in self.chat_candidates(models, messages, max_tokens):
            body = {"model": model, "messages": messages, "stream": True, "max_completion_tokens": max_tokens,
                    "temperature": self.settings.chat_temperature if temperature is None else temperature}
            # Conversation needs no visible reasoning: it only delays the first word.
            body |= ({"reasoning_effort": "none"} if model.startswith("qwen/") else
                     {"reasoning_effort": "low", "include_reasoning": False} if model.startswith("openai/gpt-oss")
                     else {})
            started, first, raw, sent, finish = time.monotonic(), None, "", 0, None
            try:
                async with self.http.stream("POST", "chat/completions", json=body) as response:
                    now = self.now()
                    self.quota[model].observe(response.headers, now)
                    if response.status_code == 429:
                        wait = reset_seconds(response.headers.get("retry-after")) or 60
                        self.quota[model].blocked_until = now + wait
                        _log.info("groq chat model=%s rate-limited for %.0fs; trying the next model", model, wait)
                        continue
                    if response.status_code in (401, 403):
                        raise ModelError("Groq rejected the API key. Check GROQ_API_KEY.")
                    if response.status_code >= 400:
                        await response.aread()
                        _log.info("groq chat model=%s returned HTTP %d (%s); trying the next model", model,
                                  response.status_code, _error_detail(response))
                        continue
                    async for line in response.aiter_lines():
                        if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                            continue
                        try:
                            choice = json.loads(line[6:])["choices"][0]
                        except (ValueError, KeyError, IndexError, TypeError):
                            continue
                        finish = choice.get("finish_reason") or finish
                        raw += (choice.get("delta") or {}).get("content") or ""
                        # Inline reasoning some models still write is never spoken.
                        visible = _THINK.sub("", raw)
                        if "<think>" in visible:
                            visible = visible[:visible.index("<think>")]
                        if len(visible) > sent:
                            first = first or time.monotonic() - started
                            yield visible[sent:]
                            sent = len(visible)
            except httpx.TimeoutException:
                if sent:
                    raise ModelError("Groq timed out mid-reply.") from None
                _log.info("groq chat model=%s timed out; trying the next model", model)
                continue
            except httpx.RequestError:
                if sent:
                    raise ModelError("Lost Groq mid-reply.") from None
                raise ModelError("Cannot reach Groq.") from None
            if not sent:
                _log.info("groq chat model=%s returned no text; trying the next model", model)
                continue
            self.last_metrics = {"provider": "groq", "model": model, "first_token_seconds": round(first or 0, 3),
                                 "total_seconds": round(time.monotonic() - started, 3),
                                 **({"truncated": True} if finish == "length" else {})}
            return
        raise ModelError("Groq could not hold the conversation: no chat model has quota left.")

    async def close(self) -> None:
        await self.http.aclose()


@dataclass
class WebAnswer:
    answer: str
    quote: str              # the words the answer says it rests on, copied from what was read
    sources: list[dict]
    evidence: str           # everything browser_search returned: result lists and page text
    searches: int = 0
    pages: int = 0


def search_evidence(executed_tools) -> str:
    """All text browser_search saw: its tool outputs, and result titles and contents."""
    parts = []
    for tool in executed_tools or []:
        parts.append(str(tool.get("output") or ""))
        for result in (tool.get("search_results") or {}).get("results") or []:
            if isinstance(result, dict):
                parts.append(f"{result.get('title') or ''} {result.get('content') or ''}")
    return "\n".join(part for part in parts if part.strip())


def search_sources(executed_tools, limit: int = 3) -> list[dict]:
    """The pages a browser_search answer rests on: those it opened, else its top results."""
    found, opened = [], []
    for tool in executed_tools or []:
        if not isinstance(tool, dict):
            continue
        page = "open" in f"{tool.get('type', '')} {tool.get('name', '')}"
        for result in (tool.get("search_results") or {}).get("results") or []:
            url = result.get("url") if isinstance(result, dict) else None
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                (opened if page else found).append({"title": str(result.get("title") or ""), "url": url})
    titles = {item["url"]: item["title"] for item in found if item["title"]}
    sources = []
    for item in opened or found:
        if any(source["url"] == item["url"] for source in sources):
            continue
        # An opened page is titled like "www.site.com - viewing lines [0 - 64] of 64".
        title = titles.get(item["url"]) or re.sub(r"\s+-\s+viewing lines.*$", "", item["title"]) or item["url"]
        sources.append({"title": title[:250], "url": item["url"], "snippet": ""})
        if len(sources) == limit:
            break
    return sources


def _seconds(value: str | None) -> float | None:
    try:
        return float(value) if value else None
    except ValueError:
        return None


def _error_detail(response: httpx.Response) -> str:
    """Groq's error code and message (never request content or credentials)."""
    try:
        error = response.json()["error"]
        return f"{error.get('code') or error.get('type') or 'no code'}: {str(error.get('message') or '')[:200]}"
    except (ValueError, KeyError, TypeError, AttributeError):
        return "no error body"


def _error_code(response: httpx.Response) -> str:
    try:
        return str(response.json()["error"].get("code") or "")
    except (ValueError, KeyError, TypeError, AttributeError):
        return ""


class HostedChat:
    """Conversation on Groq while it has quota (first words in about a second), streamed; the
    local chat model otherwise. Everything else the session asks of it goes to the local model."""

    def __init__(self, hosted: GroqClient, local, models: list[str]) -> None:
        self.hosted, self.local, self.models = hosted, local, models
        self.last_metrics: dict = {}

    async def stream_text(self, messages: list[dict], max_tokens: int = 300, temperature: float | None = None):
        spoke = False
        if self.hosted.chat_candidates(self.models, messages, max_tokens):
            try:
                async for chunk in self.hosted.stream_chat(messages, self.models, max_tokens, temperature):
                    spoke = True
                    yield chunk
                self.last_metrics = dict(self.hosted.last_metrics)
                return
            except ModelError as exc:
                if spoke:
                    raise
                _log.warning("Groq unavailable for conversation (%s); using the local model, which is slower.", exc)
        else:
            _log.warning("Groq has no quota left for conversation; using the local model, which is slower.")
        async for chunk in self.local.stream_text(messages, max_tokens=max_tokens, temperature=temperature):
            yield chunk
        self.last_metrics = {"provider": "local", **dict(getattr(self.local, "last_metrics", None) or {})}

    def __getattr__(self, name):
        return getattr(self.local, name)


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
        if self.used >= self.limit:
            _log.info("Groq call budget for %s used (%d); using the local model.", self.purpose, self.limit)
        elif not self.hosted.has_capacity(messages):
            # Every model is rate-limited: say so, since the local model can take minutes.
            _log.warning("Groq has no quota left for %s; using the local model, which is slower.", self.purpose)
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
