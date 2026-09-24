import httpx

from app.config import Settings
from .base import ModelError


# Ollama reports durations in nanoseconds; only fields it actually sends are kept.
_DURATIONS = {"total_duration": "total_seconds", "load_duration": "load_seconds",
              "prompt_eval_duration": "prompt_eval_seconds", "eval_duration": "eval_seconds"}
_COUNTS = {"prompt_eval_count": "prompt_tokens", "eval_count": "completion_tokens"}


class OllamaClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.last_metrics: dict = {}
        self.http = httpx.AsyncClient(base_url=settings.ollama_base_url.rstrip("/"),
                                     timeout=settings.llm_timeout_seconds, trust_env=False)

    async def generate(self, messages: list[dict], response_schema: dict) -> str:
        thinking = self.settings.decision_thinking and response_schema.get("title") in {
            "ExplorerCommand", "CoordinatorDecision", "GoalPlan", "CriticReview"
        }
        try:
            response = await self.http.post("/api/chat", json={
                "model": self.settings.ollama_model, "messages": messages,
                "format": response_schema, "stream": False, "think": thinking,
                "options": {"temperature": self.settings.temperature,
                            "num_ctx": self.settings.context_tokens,
                            "num_predict": self.settings.decision_output_tokens if thinking else 1024},
            })
            response.raise_for_status()
            # The optional thinking field is deliberately ignored and never persisted.
            envelope = response.json()
            content = envelope["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("Missing content")
            self.last_metrics = {"model": envelope.get("model") or self.settings.ollama_model,
                                 **{name: round(envelope[key] / 1e9, 3)
                                    for key, name in _DURATIONS.items() if isinstance(envelope.get(key), (int, float))},
                                 **{name: envelope[key]
                                    for key, name in _COUNTS.items() if isinstance(envelope.get(key), int)}}
            return content
        except httpx.TimeoutException:
            raise ModelError("Ollama timed out. Try a smaller model or increase LLM_TIMEOUT_SECONDS.") from None
        except httpx.HTTPStatusError as exc:
            raise ModelError(f"Ollama returned HTTP {exc.response.status_code}. Check the configured model is installed.") from None
        except httpx.RequestError:
            raise ModelError("Cannot reach Ollama. Start ollama serve and check OLLAMA_BASE_URL.") from None
        except (ValueError, KeyError, TypeError):
            raise ModelError("Ollama returned an unexpected response envelope.") from None

    async def health(self) -> dict:
        try:
            response = await self.http.get("/api/tags", timeout=3)
            response.raise_for_status()
            models = [m["name"] for m in response.json()["models"]]
            return {"reachable": True, "model": self.settings.ollama_model,
                    "model_available": self.settings.ollama_model in models}
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return {"reachable": False, "model": self.settings.ollama_model, "model_available": False}

    async def close(self) -> None:
        await self.http.aclose()
