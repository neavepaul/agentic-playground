from app.config import Settings
from .colibri import ColibriClient
from .ollama import OllamaClient


def create_client(settings: Settings) -> ColibriClient | OllamaClient:
    if settings.llm_provider == "colibri":
        return ColibriClient(settings)
    return OllamaClient(settings)


def create_hosted_client(settings: Settings):
    """Groq for bounded hosted calls, or None (no key) so everything stays local."""
    if settings.groq_api_key is None or not settings.groq_api_key.get_secret_value().strip():
        return None
    from .groq import GroqClient
    return GroqClient(settings)


def create_chat_client(settings: Settings, default):
    """A separate (usually smaller, faster) model for conversation, or the main client."""
    if not (settings.chat_provider or settings.chat_model or settings.chat_base_url):
        return default
    provider = settings.chat_provider or settings.llm_provider
    overrides = {"llm_provider": provider}
    if settings.chat_model:
        overrides[f"{provider}_model"] = settings.chat_model
    if settings.chat_base_url:
        overrides[f"{provider}_base_url"] = settings.chat_base_url
    return create_client(settings.model_copy(update=overrides))
