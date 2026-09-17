"""Pick an answer provider by name.

Adding a new one means writing a Provider subclass and adding a line to
PROVIDERS below. Nothing else in the project needs to change.
"""

from backend.providers.anthropic_provider import AnthropicProvider
from backend.providers.base import Provider, ProviderError, ProviderNotReady
from backend.providers.gemini import GeminiProvider
from backend.providers.ollama import OllamaProvider
from utils import config

PROVIDERS: dict[str, type[Provider]] = {
    "gemini": GeminiProvider,
    "ollama": OllamaProvider,
    "anthropic": AnthropicProvider,
}


def get_provider(name: str | None = None, api_key: str | None = None) -> Provider:
    """Build a provider by name.

    `api_key` overrides whatever the environment holds, which is what lets a
    deployed copy ask each visitor for their own key instead of shipping one.
    Providers that need no key ignore it.
    """
    name = (name or config.PROVIDER).lower()
    if name not in PROVIDERS:
        raise ValueError(
            f"Unknown provider {name!r}. Available: {', '.join(PROVIDERS)}"
        )

    provider_class = PROVIDERS[name]
    if api_key and provider_class.key_variable:
        return provider_class(api_key=api_key)
    return provider_class()


def model_for(name: str) -> str:
    """The configured model name for a provider, for display purposes."""
    return {
        "gemini": config.GEMINI_MODEL,
        "ollama": config.OLLAMA_MODEL,
        "anthropic": config.ANTHROPIC_MODEL,
    }.get(name.lower(), "?")


__all__ = [
    "PROVIDERS",
    "Provider",
    "ProviderError",
    "ProviderNotReady",
    "get_provider",
    "model_for",
]
