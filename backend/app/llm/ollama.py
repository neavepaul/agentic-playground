import httpx

from app.config import Settings
from .base import ModelError


class OllamaClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.http = httpx.AsyncClient(base_url=settings.ollama_base_url.rstrip("/"),
                                     timeout=settings.llm_timeout_seconds, trust_env=False)

    async def generate(self, messages: list[dict], response_schema: dict) -> str:
        try:
            response = await self.http.post("/api/chat", json={
                "model": self.settings.ollama_model, "messages": messages,
                "format": response_schema, "stream": False, "think": False,
                "options": {"temperature": self.settings.temperature,
                            "num_ctx": self.settings.context_tokens, "num_predict": 1024},
            })
            response.raise_for_status()
            # The optional thinking field is deliberately ignored and never persisted.
            content = response.json()["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("Missing content")
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
