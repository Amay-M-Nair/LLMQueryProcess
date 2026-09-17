"""Anthropic Claude - paid, and the strongest at following the citation rules."""

import os
from typing import Iterator

from backend.providers.base import Provider, ProviderError, ProviderNotReady
from utils import config


class AnthropicProvider(Provider):
    name = "anthropic"
    note = "Paid (needs credit). Best instruction-following of the three."
    key_variable = "ANTHROPIC_API_KEY"

    def __init__(self, model: str = config.ANTHROPIC_MODEL, api_key: str | None = None):
        self.model = model
        self.api_key = (api_key or "").strip() or None

    def _key(self) -> str | None:
        return self.api_key or os.environ.get("ANTHROPIC_API_KEY")

    def check_ready(self) -> None:
        if not self._key():
            raise ProviderNotReady(
                "No ANTHROPIC_API_KEY found. Add it to your .env file, or set "
                'PROVIDER = "gemini" in utils/config.py to use the free tier.'
            )

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        import anthropic

        self.check_ready()
        client = anthropic.Anthropic(api_key=self._key())

        request = dict(
            model=self.model,
            max_tokens=config.ANTHROPIC_MAX_TOKENS,
            system=system,
            thinking={"type": "adaptive"},
            output_config={"effort": config.ANTHROPIC_EFFORT},
            messages=[{"role": "user", "content": prompt}],
        )

        yielded = False
        try:
            if config.USE_REFUSAL_FALLBACKS:
                with client.beta.messages.stream(
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                    **request,
                ) as stream:
                    for text in stream.text_stream:
                        yielded = True
                        yield text
                    _check_refusal(stream.get_final_message())
                    return
        except anthropic.BadRequestError:
            # Fallback beta not enabled on this account - use the plain endpoint.
            if yielded:
                raise

        try:
            with client.messages.stream(**request) as stream:
                for text in stream.text_stream:
                    yield text
                _check_refusal(stream.get_final_message())
        except anthropic.APIStatusError as exc:
            raise ProviderError(f"Claude request failed: {exc}") from exc


def _check_refusal(message) -> None:
    """A declined request returns HTTP 200 with no text - say so explicitly."""
    if getattr(message, "stop_reason", None) == "refusal":
        details = getattr(message, "stop_details", None)
        category = getattr(details, "category", None) or "unspecified"
        raise ProviderError(
            f"The model declined to answer (category: {category}). Try rephrasing."
        )
