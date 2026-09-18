"""Make a follow-up question stand on its own.

"What about its limitations?" retrieves nothing useful: the words that decide
which chunks matter - Transformer, attention, whatever "it" was - are not in
the question. Retrieval can only match what it is given, so the reference has
to be resolved before the search, not after.

Rewriting is conditional. Most questions already stand alone, and sending
every one of them to a model would double the cost of the system to fix a
minority of cases. So a cheap test decides, and only questions that look
dependent on the conversation are rewritten.

The rewritten question is used for retrieval only. Generation always sees
what the user actually typed - their wording carries emphasis a paraphrase
loses, and an answer that quietly responds to a reworded question is
disconcerting to read.
"""

import re
from dataclasses import dataclass

from backend.llm import Provider, ProviderError, ProviderNotReady
from utils import config, logs, prompts

log = logs.get(__name__)

# Words that point at something said earlier rather than naming it.
DANGLING_REFERENCE = re.compile(
    r"\b(it|its|it's|that|this|these|those|they|them|their|"
    r"the former|the latter|the above|the same)\b",
    re.IGNORECASE,
)

# Openers that are almost always continuations.
CONTINUATION = re.compile(
    r"^\s*(and|but|so|also|what about|how about|why not|why|what else|"
    r"anything else|more|go on|continue|elaborate|expand)\b",
    re.IGNORECASE,
)


@dataclass
class Rewrite:
    """The query to search with, and whether it was changed at all."""

    query: str
    original: str
    changed: bool
    reason: str

    @property
    def cost_a_call(self) -> bool:
        return self.reason == "rewritten by the model"


def needs_rewriting(query: str, history: list[dict]) -> bool:
    """Cheap test for whether this question leans on the conversation."""
    if not history or not query.strip():
        return False
    if DANGLING_REFERENCE.search(query):
        return True
    if CONTINUATION.match(query):
        return True
    # A very short question is usually a fragment of the previous one.
    return len(query.split()) <= config.SHORT_FOLLOWUP_WORDS


def looks_degenerate(rewritten: str, original: str) -> bool:
    """Reject a rewrite that is obviously not one.

    Models asked to rewrite sometimes answer instead, or return the prompt,
    or explain what they did. Any of those would be searched for as if it
    were the question, which is worse than not rewriting at all.
    """
    if not rewritten:
        return True
    if len(rewritten) > max(len(original) * 4, 200):
        return True
    if "\n" in rewritten.strip():
        return True
    return False


def rewrite(
    query: str,
    history: list[dict],
    provider: Provider | None = None,
) -> Rewrite:
    """Resolve references in `query`, or hand it back untouched."""
    original = query

    if not needs_rewriting(query, history):
        return Rewrite(query, original, False, "stands on its own")

    if provider is None:
        return Rewrite(query, original, False, "no provider to rewrite with")

    system, user = prompts.rewrite_prompt(query, history, config.HISTORY_TURNS)
    try:
        reply = provider.complete(system, user)
    except (ProviderNotReady, ProviderError) as exc:
        log.info("rewrite unavailable, searching the original question: %s", exc)
        return Rewrite(query, original, False, "rewrite call failed")
    except Exception:  # noqa: BLE001 - a failed rewrite must not sink the question
        log.warning("rewrite failed, searching the original question", exc_info=True)
        return Rewrite(query, original, False, "rewrite call failed")

    cleaned = reply.strip().strip('"').strip("'").strip()
    if looks_degenerate(cleaned, original):
        log.info("discarded an unusable rewrite (%d chars)", len(cleaned))
        return Rewrite(query, original, False, "the rewrite was unusable")

    if cleaned.lower() == original.strip().lower():
        return Rewrite(original, original, False, "the model left it unchanged")

    return Rewrite(cleaned, original, True, "rewritten by the model")
