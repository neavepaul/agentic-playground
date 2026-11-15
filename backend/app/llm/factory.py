from app.config import Settings
from .ollama import OllamaClient


def create_client(settings: Settings) -> OllamaClient:
    """The one local model: Ollama, for conversation and errands alike."""
    return OllamaClient(settings)


def create_hosted_client(settings: Settings):
    """Groq for bounded hosted calls, or None (no key) so everything stays local."""
    if settings.groq_api_key is None or not settings.groq_api_key.get_secret_value().strip():
        return None
    from .groq import GroqClient
    return GroqClient(settings)
