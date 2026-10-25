"""Colibri's local OpenAI-compatible chat API, with application JSON validation."""
import json
import time
from collections.abc import AsyncIterator

import httpx

from app.config import Settings
from .base import ModelError


class ColibriClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.last_metrics: dict = {}
        self.http = httpx.AsyncClient(
            base_url=settings.colibri_base_url.rstrip("/") + "/",
            timeout=settings.llm_timeout_seconds, trust_env=False)

    async def generate(self, messages: list[dict], response_schema: dict) -> str:
        self.last_metrics = {}
        thinking = self.settings.decision_thinking and response_schema.get("title") in {
            "ExplorerCommand", "CoordinatorDecision", "GoalPlan", "CriticReview"
        }
        # Qwen's Colibri engine rejects response_format grammars. structured()
        # supplies the schema in the prompt and validates/repairs the result.
        instructions = ("Return only one valid JSON object matching the supplied schema. "
                        "Do not include markdown fences, commentary or reasoning.\n")
        request_messages = [dict(message) for message in messages]
        if request_messages and request_messages[0].get("role") == "system":
            request_messages[0]["content"] = instructions + request_messages[0]["content"]
        else:
            request_messages.insert(0, {"role": "system", "content": instructions})
        try:
            response = await self.http.post("chat/completions", json={
                "model": self.settings.colibri_model,
                "messages": request_messages, "stream": False,
                "enable_thinking": thinking,
                "temperature": self.settings.temperature,
                "max_tokens": self.settings.decision_output_tokens if thinking else 1024,
            })
            response.raise_for_status()
            envelope = response.json()
            choice = envelope["choices"][0]
            content = choice["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("Missing content")
            usage = envelope.get("usage") or {}
            self.last_metrics = {"model": envelope.get("model") or self.settings.colibri_model,
                                 **{key: usage[key] for key in ("prompt_tokens", "completion_tokens")
                                    if isinstance(usage.get(key), int)}}
            if choice.get("finish_reason") == "length":
                raise ModelError("Colibri reached its output or context limit. Increase the server's "
                                 "output/context budget or shorten the task context.")
            # reasoning_content is deliberately ignored, like Ollama's thinking.
            return content
        except httpx.TimeoutException:
            raise ModelError("Colibri timed out. Check the server or increase LLM_TIMEOUT_SECONDS.") from None
        except httpx.HTTPStatusError as exc:
            raise ModelError(f"Colibri returned HTTP {exc.response.status_code}. Check its model and server configuration.") from None
        except httpx.RequestError:
            raise ModelError("Cannot reach Colibri. Start it on port 8001 and check COLIBRI_BASE_URL.") from None
        except (ValueError, KeyError, TypeError, IndexError, AttributeError):
            raise ModelError("Colibri returned an unexpected response envelope.") from None

    async def stream_text(self, messages: list[dict], max_tokens: int = 300,
                          temperature: float | None = None) -> AsyncIterator[str]:
        """Plain conversational text as server-sent events; no JSON, no thinking."""
        started, self.last_metrics = time.monotonic(), {}
        try:
            async with self.http.stream("POST", "chat/completions", json={
                "model": self.settings.colibri_model, "messages": messages, "stream": True,
                "enable_thinking": False, "max_tokens": max_tokens,
                "stream_options": {"include_usage": True},
                "temperature": self.settings.temperature if temperature is None else temperature,
            }) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    envelope = json.loads(data)
                    usage = envelope.get("usage") or {}
                    self.last_metrics.update({key: usage[key] for key in ("prompt_tokens", "completion_tokens")
                                              if isinstance(usage.get(key), int)})
                    choices = envelope["choices"]
                    if not choices:  # The optional final usage event has no choice.
                        continue
                    if choices[0].get("finish_reason") == "length":
                        raise ModelError("Colibri reached its reply limit before finishing. Please try a shorter request.")
                    delta = choices[0].get("delta", {}).get("content")
                    if delta:
                        self.last_metrics.setdefault("first_token_seconds", round(time.monotonic() - started, 3))
                        yield delta
            self.last_metrics["total_seconds"] = round(time.monotonic() - started, 3)
        except httpx.TimeoutException:
            raise ModelError("Colibri timed out while replying.") from None
        except httpx.HTTPStatusError as exc:
            raise ModelError(f"Colibri returned HTTP {exc.response.status_code} while replying.") from None
        except httpx.RequestError:
            raise ModelError("Cannot reach Colibri. Start it on port 8001 and check COLIBRI_BASE_URL.") from None
        except (ValueError, KeyError, TypeError, IndexError):
            raise ModelError("Colibri returned an unexpected stream.") from None

    async def health(self) -> dict:
        try:
            response = await self.http.get("models", timeout=3)
            response.raise_for_status()
            models = [model["id"] for model in response.json()["data"]]
            return {"reachable": True, "model": self.settings.colibri_model,
                    "model_available": self.settings.colibri_model in models}
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return {"reachable": False, "model": self.settings.colibri_model, "model_available": False}

    async def close(self) -> None:
        await self.http.aclose()
