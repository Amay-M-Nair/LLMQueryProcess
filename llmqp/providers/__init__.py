"""Pick an answer provider by name.

Adding a new one means writing a Provider subclass and adding a line to
PROVIDERS below. Nothing else in the project needs to change.
"""

from .. import config
from .anthropic_provider import AnthropicProvider
from .base import Provider, ProviderError, ProviderNotReady
from .gemini import GeminiProvider
from .ollama import OllamaProvider

PROVIDERS: dict[str, type[Provider]] = {
    "gemini": GeminiProvider,
    "ollama": OllamaProvider,
    "anthropic": AnthropicProvider,
}


def get_provider(name: str | None = None) -> Provider:
    name = (name or config.PROVIDER).lower()
    if name not in PROVIDERS:
        raise ValueError(
            f"Unknown provider {name!r}. Available: {', '.join(PROVIDERS)}"
        )
    return PROVIDERS[name]()


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
