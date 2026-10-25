from app.config import Settings
from .colibri import ColibriClient
from .ollama import OllamaClient


def create_client(settings: Settings) -> ColibriClient | OllamaClient:
    if settings.llm_provider == "colibri":
        return ColibriClient(settings)
    return OllamaClient(settings)
