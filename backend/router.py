"""Turn an intent into the path the question takes.

This is deliberately the dullest file in the project. Routing is where a
system quietly grows a second copy of its business logic - a little retrieval
here, a little prompt building there - until no single place describes what
happens to a question. So this decides, and does nothing.
"""

from dataclasses import dataclass

from backend.intent_classifier import (
    CALCULATION,
    GENERAL,
    Intent,
    UNKNOWN,
)

RETRIEVAL = "retrieval"   # through the documents, answer cited
DIRECT = "direct"         # straight to the model, no retrieval
CALCULATE = "calculate"   # worked out locally, no model at all
CLARIFY = "clarify"       # nothing sensible to do; ask the user


@dataclass
class Route:
    name: str
    intent: Intent
    reason: str
    downgraded_from: str | None = None

    @property
    def uses_model(self) -> bool:
        return self.name in (RETRIEVAL, DIRECT)

    @property
    def cites_sources(self) -> bool:
        return self.name == RETRIEVAL


def route(intent: Intent, has_index: bool) -> Route:
    """Pick the path. `has_index` is checked here, not trusted from the intent.

    The classifier can ask for retrieval against an index that does not exist
    - the model does not know what is loaded, and a stale intent can outlive
    a cleared index. Sending that to the RAG path would produce an answer
    grounded in nothing, so it is downgraded and the downgrade is recorded
    rather than silently applied.
    """
    if intent.name == CALCULATION:
        return Route(CALCULATE, intent, "arithmetic, worked out locally")

    if intent.name == UNKNOWN:
        return Route(CLARIFY, intent, intent.reason)

    if intent.requires_retrieval:
        if not has_index:
            return Route(
                DIRECT,
                intent,
                "needs documents, but none are indexed - answering unaided",
                downgraded_from=intent.name,
            )
        return Route(RETRIEVAL, intent, "answerable from the indexed documents")

    if intent.name == GENERAL:
        return Route(DIRECT, intent, "general knowledge, no retrieval needed")

    return Route(DIRECT, intent, f"{intent.name} does not need retrieval")
