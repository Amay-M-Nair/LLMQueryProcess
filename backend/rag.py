"""Generating an answer - with retrieved excerpts, or without.

Two prompts, two contracts. A retrieval answer is bound to the excerpts and
must cite them; a direct answer has nothing to cite and must not pretend
otherwise. Keeping them apart is what stops a general question picking up
citation machinery that has nothing behind it.

Nothing here decides *which* path a question takes - that is the router's
job - and nothing here touches Streamlit.
"""

from typing import Iterator

from backend.llm import Provider, ProviderError, ProviderNotReady, get_provider
from ingestion.chunker import Chunk
from utils import config, prompts

__all__ = [
    "ProviderError",
    "ProviderNotReady",
    "build_prompt",
    "stream_answer",
    "stream_direct_answer",
]


def build_prompt(
    question: str,
    retrieved: list[tuple[Chunk, float]],
    history: list[dict] | None = None,
) -> str:
    """Lay the excerpts out as a numbered list the model can cite by number.

    The conversation goes above them when there is one. Rewriting solves a
    different problem: it fixes what gets searched for, so "where does that
    figure come from?" retrieves the right excerpts and still reads as a
    non-sequitur, because the model writing the answer never saw what it was
    being asked about.

    It is kept in its own block, and the system prompt says plainly that it is
    not evidence. Given a conversation and sources run together, a model will
    cite the conversation.
    """
    blocks = []
    for index, (chunk, _score) in enumerate(retrieved, start=1):
        blocks.append(f"[{index}] Source: {chunk.label}\n{chunk.text}")

    excerpts = "\n\n".join(blocks) if blocks else "(no excerpts were retrieved)"

    preamble = ""
    if history:
        conversation = prompts.history_block(history, config.HISTORY_TURNS)
        preamble = f"Conversation so far:\n{conversation}\n\n---\n\n"

    return (
        f"{preamble}Here are the source excerpts:\n\n{excerpts}\n\n"
        f"---\n\nQuestion: {question}"
    )


def stream_answer(
    question: str,
    retrieved: list[tuple[Chunk, float]],
    provider: Provider | None = None,
    length: str = prompts.DEFAULT_LENGTH,
    history: list[dict] | None = None,
) -> Iterator[str]:
    """Answer from the excerpts, with citations. Yields text as it arrives."""
    provider = provider or get_provider()
    yield from provider.stream(
        prompts.rag_system(length), build_prompt(question, retrieved, history)
    )


def stream_direct_answer(
    question: str,
    history: list[dict] | None = None,
    provider: Provider | None = None,
    length: str = prompts.DEFAULT_LENGTH,
) -> Iterator[str]:
    """Answer from the model's own knowledge, with no excerpts and no citations.

    History goes in so a conversation stays coherent across turns - without
    it, "and why is that?" has nothing to attach to.
    """
    provider = provider or get_provider()
    system, user = prompts.direct_prompt(
        question, history or [], config.HISTORY_TURNS, length
    )
    yield from provider.stream(system, user)
