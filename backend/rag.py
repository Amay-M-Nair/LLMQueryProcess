"""Build the prompt and hand it to whichever provider is configured."""

from typing import Iterator

from backend.llm import Provider, ProviderError, ProviderNotReady, get_provider
from ingestion.chunker import Chunk
from utils import config

__all__ = [
    "ProviderError",
    "ProviderNotReady",
    "build_prompt",
    "stream_answer",
]


def build_prompt(question: str, retrieved: list[tuple[Chunk, float]]) -> str:
    """Lay the excerpts out as a numbered list the model can cite by number."""
    blocks = []
    for index, (chunk, _score) in enumerate(retrieved, start=1):
        blocks.append(f"[{index}] Source: {chunk.label}\n{chunk.text}")

    excerpts = "\n\n".join(blocks) if blocks else "(no excerpts were retrieved)"
    return (
        f"Here are the source excerpts:\n\n{excerpts}\n\n"
        f"---\n\nQuestion: {question}"
    )


def stream_answer(
    question: str,
    retrieved: list[tuple[Chunk, float]],
    provider: Provider | None = None,
) -> Iterator[str]:
    """Yield the answer text as it arrives, so the page can render it live."""
    provider = provider or get_provider()
    prompt = build_prompt(question, retrieved)
    yield from provider.stream(config.SYSTEM_PROMPT, prompt)
