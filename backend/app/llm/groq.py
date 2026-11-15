"""Groq's hosted OpenAI-compatible API, for a few bounded, high-value calls.

The local model (Ollama) is the engine for conversation and the task loop. Groq's free tier
limits each model separately, so a pool of models is tried in preference order and
a model whose quota is spent is skipped until its reset time. Each model is tried
at most once per call, except that when every model is rate-limited and one frees within
GROQ_MAX_WAIT_SECONDS, the call waits for it once: seconds here beat minutes on the local model.

Requests are plain chat completions: the system, user and history messages as given,
with no JSON mode, tools or reasoning parameters. Callers validate any JSON locally.
The one exception is search(): Groq's built-in browser_search, on the gpt-oss models
that support it, which looks something up on the web and writes the answer in one call.
"""
import asyncio
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
    limit_tokens: int | None = None
    tokens_seen_at: float = 0.0

    def available(self, need_tokens: int, now: float) -> bool:
        return self.free_in(need_tokens, now) <= 0

    def free_in(self, need_tokens: int, now: float) -> float:
        """Seconds until this model can take a call this size; 0 if it can now."""
        waits = [self.blocked_until - now]
        if self.remaining_requests is not None and self.remaining_requests < 1:
            waits.append(self.requests_reset_at - now)
        if self.remaining_tokens is not None and self.remaining_tokens < need_tokens:
            # reset-tokens is when the per-minute bucket is full again; it refills steadily, so a
            # call that fits the bucket can go once enough has come back.
            full = self.tokens_reset_at - now
            if self.limit_tokens and need_tokens <= self.limit_tokens:
                refill = (need_tokens - self.remaining_tokens) * 60 / self.limit_tokens - (now - self.tokens_seen_at)
                full = min(full, refill)
            waits.append(full)
        return max(0.0, *waits)

    def rate_limited(self, wait: float, now: float) -> None:
        """A 429: its retry-after is Groq's answer for this model, over what the token headers suggest."""
        self.blocked_until = now + wait
        self.remaining_tokens = self.remaining_requests = None

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
            self.remaining_tokens, self.tokens_seen_at = tokens, now
            self.tokens_reset_at = now + (reset_seconds(headers.get("x-ratelimit-reset-tokens")) or 60)
            self.limit_tokens = number("x-ratelimit-limit-tokens") or self.limit_tokens


def estimate_tokens(messages: list[dict]) -> int:
    # Conservative: ~4 characters per token, plus room for the reply.
    return sum(len(message.get("content") or "") for message in messages) // 3 + 1024


class GroqClient:
    def __init__(self, settings: Settings, now=time.monotonic, sleep=None) -> None:
        if settings.groq_api_key is None:
            raise ValueError("GROQ_API_KEY is not set.")
        self.settings = settings
        self.models = [model.strip() for model in settings.groq_models.split(",") if model.strip()]
        self.quota = {model: ModelQuota() for model in self.models}
        self.now, self.sleep = now, sleep or asyncio.sleep
        self.last_metrics: dict = {}
        self.http = httpx.AsyncClient(
            base_url=settings.groq_base_url.rstrip("/") + "/", timeout=settings.groq_timeout_seconds,
            headers={"Authorization": f"Bearer {settings.groq_api_key.get_secret_value()}"})

    def candidates(self, need_tokens: int, searching: bool = False) -> list[str]:
        now = self.now()
        return [model for model in self.models if self.quota[model].available(need_tokens, now)
                and (not searching or model.startswith(SEARCH_MODELS))]

    def free_soon(self, need_tokens: int, searching: bool = False) -> dict[str, float]:
        return self._soon([model for model in self.models if not searching or model.startswith(SEARCH_MODELS)],
                          need_tokens)

    def _soon(self, models: list[str], need_tokens: int) -> dict[str, float]:
        """Rate-limited models that can take this call within GROQ_MAX_WAIT_SECONDS, with their wait."""
        now, limit = self.now(), self.settings.groq_max_wait_seconds
        waits = {model: self.quota.setdefault(model, ModelQuota()).free_in(need_tokens, now) for model in models}
        return {model: wait for model, wait in waits.items() if 0 < wait <= limit}

    def has_capacity(self, messages: list[dict] | None = None) -> bool:
        need = estimate_tokens(messages or [])
        return bool(self.candidates(need) or self.free_soon(need))

    def can_search(self, messages: list[dict] | None = None) -> bool:
        need = estimate_tokens(messages or []) + SEARCH_TOKENS
        return bool(self.candidates(need, searching=True) or self.free_soon(need, searching=True))

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
        tried, retried, waited = [], set(), False
        queue = self.candidates(need, searching)
        while True:
            if not queue and not waited and (soon := self.free_soon(need, searching)):
                # Every model is rate-limited, one only for seconds: the local model would take minutes.
                waited, wait = True, min(soon.values())
                _log.info("groq: every model rate-limited; waiting %.1fs for %s instead of the local model",
                          wait, min(soon, key=soon.get))
                await self.sleep(wait)
                queue = [model for model in self.candidates(need, searching) if model in soon]
            if not queue:
                break
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
                self.quota[model].rate_limited(wait or 60, now)
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

    @staticmethod
    def _chat_need(messages: list[dict], max_tokens: int) -> int:
        # A spoken reply is a few dozen tokens: reserve for that, not the 1,200-token ceiling, or a
        # model's per-minute quota looks spent after two replies.
        return estimate_tokens(messages) - 1024 + min(max_tokens, 300)

    def chat_candidates(self, models: list[str], messages: list[dict], max_tokens: int) -> list[str]:
        now, need = self.now(), self._chat_need(messages, max_tokens)
        return [m for m in models if self.quota.setdefault(m, ModelQuota()).available(need, now)]

    def chat_free_soon(self, models: list[str], messages: list[dict], max_tokens: int) -> dict[str, float]:
        return self._soon(models, self._chat_need(messages, max_tokens))

    async def stream_chat(self, messages: list[dict], models: list[str], max_tokens: int,
                          temperature: float | None = None):
        """The reply's text as Groq streams it, from the first of `models` that can answer.

        A model that fails before any text arrives (no quota, an error) hands over to the
        next; once text has been spoken, a failure is raised, never papered over. When every
        model is rate-limited for only seconds, the reply waits for one, once."""
        self.last_metrics = {}
        queue, waited, retried, tried = self.chat_candidates(models, messages, max_tokens), False, set(), False
        while True:
            if not queue and not waited and (soon := self.chat_free_soon(models, messages, max_tokens)):
                waited, wait = True, min(soon.values())
                _log.info("groq chat: every model rate-limited; waiting %.1fs for %s instead of the local model",
                          wait, min(soon, key=soon.get))
                await self.sleep(wait)
                queue = [m for m in self.chat_candidates(models, messages, max_tokens) if m in soon]
            if not queue:
                break
            model, tried = queue.pop(0), True
            body = {"model": model, "messages": messages, "stream": True, "max_completion_tokens": max_tokens,
                    "temperature": self.settings.chat_temperature if temperature is None else temperature}
            # Conversation needs no visible reasoning: it only delays the first word.
            body |= ({"reasoning_effort": "none"} if model.startswith("qwen/") else
                     {"reasoning_effort": "low", "include_reasoning": False} if model.startswith("openai/gpt-oss")
                     else {})
            started, first, raw, sent, finish, stream_error = time.monotonic(), None, "", 0, None, ""
            try:
                async with self.http.stream("POST", "chat/completions", json=body) as response:
                    now = self.now()
                    self.quota[model].observe(response.headers, now)
                    if response.status_code == 429:
                        retry = response.headers.get("retry-after")
                        wait = reset_seconds(retry) or _seconds(retry) or 60  # usually plain seconds: "8"
                        self.quota[model].rate_limited(wait, now)
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
                            event = json.loads(line[6:])
                            if isinstance(event, dict) and isinstance(event.get("error"), dict):
                                # Groq reports some failures inside a 200 stream, not as a status.
                                error = event["error"]
                                stream_error = f"{error.get('code') or error.get('type') or 'no code'}: " \
                                               f"{str(error.get('message') or '')[:200]}"
                                continue
                            choice = event["choices"][0]
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
                # One empty generation (often all its tokens spent reasoning) says little about the
                # next: the same model gets one more try, which costs seconds, not the local model's minutes.
                # A model that called a tool will do it again for this message (gpt-oss reaches for its
                # own Python tool when asked to "use Python"): the next model instead.
                again = model not in retried and not stream_error.startswith("tool_use_failed")
                retried.add(model)
                if again:
                    queue.insert(0, model)
                _log.info("groq chat model=%s returned no text (finish=%s, %d characters of hidden reasoning%s); %s",
                          model, finish, len(raw), f", stream error {stream_error}" if stream_error else "",
                          "trying it once more" if again else "trying the next model")
                continue
            self.last_metrics = {"provider": "groq", "model": model, "first_token_seconds": round(first or 0, 3),
                                 "total_seconds": round(time.monotonic() - started, 3),
                                 **({"truncated": True} if finish == "length" else {})}
            return
        reason = "no chat model answered" if tried else "no chat model has quota left"
        raise ModelError(f"Groq could not hold the conversation: {reason}.")

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
        if (self.hosted.chat_candidates(self.models, messages, max_tokens)
                or self.hosted.chat_free_soon(self.models, messages, max_tokens)):
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
