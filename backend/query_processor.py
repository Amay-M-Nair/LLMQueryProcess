"""The whole journey of a question, in one place.

Everything else in backend/ does one step and knows nothing about the others.
This is the only module that knows the order they go in, which is what keeps
the flow readable: to find out what happens to a question, read `process`.

It returns a plan and a trace rather than an answer, because the answer is
streamed and the caller decides how. The trace is the same object the debug
panel renders and the evaluation harness scores, so what you see while using
the app is exactly what gets measured.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator

from backend import calculator, intent_classifier, query_rewriter, rag, router
from backend.intent_classifier import Intent
from backend.llm import Provider
from backend.query_rewriter import Rewrite
from backend.router import Route
from ingestion.chunker import Chunk
from ingestion.pipeline import retrieve
from utils import config, prompts
from utils.preprocessing import Query, preprocess
from vectorstore.faiss_store import VectorStore


@dataclass
class QueryTrace:
    """What happened to one question, and what it cost.

    Written once and read twice: by the debug panel, and by the evaluation
    harness. Anything worth measuring later belongs here rather than in a log
    line nobody parses.
    """

    query: Query
    intent: Intent | None = None
    route: Route | None = None
    rewrite: Rewrite | None = None
    retrieved: list[tuple[Chunk, float]] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    api_calls: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def search_query(self) -> str:
        """What retrieval actually searched for."""
        return self.rewrite.query if self.rewrite else self.query.cleaned

    @property
    def total_seconds(self) -> float:
        return sum(self.timings.values())

    @contextmanager
    def timed(self, stage: str):
        started = time.monotonic()
        try:
            yield
        finally:
            self.timings[stage] = time.monotonic() - started


@dataclass
class Plan:
    """What to do about a question, and everything learned deciding it."""

    trace: QueryTrace
    answer: str | None = None          # already known, needs no model
    stream: object | None = None       # a callable returning an Iterator[str]
    message: str | None = None         # something to tell the user instead

    @property
    def sources(self) -> list[tuple[Chunk, float]]:
        return self.trace.retrieved


def process(
    question: str,
    store: VectorStore | None = None,
    history: list[dict] | None = None,
    provider: Provider | None = None,
    on_stage: Callable[[str], None] | None = None,
    length: str = prompts.DEFAULT_LENGTH,
) -> Plan:
    """Preprocess, classify, route, and prepare the answer.

    No model is called for the answer here - the caller streams it - but
    classification and rewriting may each spend one call, and the trace
    records how many were spent.

    `on_stage` is called with a short description before each step. Deciding
    how to answer takes several seconds, most of it classification, and a
    caller with no way to say what is happening can only show dead air.
    """
    history = history or []
    say = on_stage or (lambda _message: None)
    query = preprocess(question)
    trace = QueryTrace(query=query)

    has_index = bool(store is not None and len(store) > 0)
    sources = store.sources if has_index else []

    if query.is_empty:
        trace.intent = intent_classifier.classify_by_rule("", has_index)
        trace.route = router.route(trace.intent, has_index)
        return Plan(trace, message="Ask a question to get started.")

    # --- What kind of question is this? ---
    say("Working out what you are asking")
    with trace.timed("classify"):
        before = intent_classifier.classify_by_rule(query.cleaned, has_index)
        trace.intent = intent_classifier.classify(
            query.cleaned, has_index=has_index, sources=sources, provider=provider
        )
    if before is None and trace.intent.from_model:
        trace.api_calls += 1

    trace.route = router.route(trace.intent, has_index)
    if trace.route.downgraded_from:
        trace.notes.append(
            f"asked for {trace.route.downgraded_from}, but nothing is indexed"
        )

    # --- Arithmetic never reaches a model ---
    if trace.route.name == router.CALCULATE:
        say("Working it out")
        try:
            with trace.timed("calculate"):
                return Plan(trace, answer=calculator.answer(query.cleaned))
        except calculator.NotArithmetic as exc:
            # The rule said this was a sum and the parser disagreed. Rather
            # than fail, let the model have it.
            trace.notes.append(f"not arithmetic after all ({exc}); answering directly")
            trace.route = router.Route(
                router.DIRECT, trace.intent, "arithmetic parse failed"
            )

    if trace.route.name == router.CLARIFY:
        return Plan(
            trace,
            message=(
                "I could not tell what that is asking. Try rephrasing it, or "
                "upload a document if it is about one of your files."
            ),
        )

    # --- Retrieval path ---
    if trace.route.name == router.RETRIEVAL:
        if query_rewriter.needs_rewriting(query.cleaned, history):
            say("Working out what the question refers to")
        with trace.timed("rewrite"):
            trace.rewrite = query_rewriter.rewrite(
                query.cleaned, history, provider=provider
            )
        if trace.rewrite.cost_a_call:
            trace.api_calls += 1

        say(f"Searching {len(store)} passage{'' if len(store) == 1 else 's'}")
        with trace.timed("retrieve"):
            trace.retrieved = retrieve(store, trace.search_query, k=config.TOP_K)

        if not trace.retrieved:
            # Nothing matched. Rather than stopping, answer unaided and say so
            # - the honest boundary is more useful than a dead end.
            trace.notes.append("nothing in the documents matched; answering unaided")
            trace.route = router.Route(
                router.DIRECT, trace.intent, "no matching excerpts",
                downgraded_from=trace.intent.name,
            )
        else:
            trace.api_calls += 1
            say(f"Writing from {len(trace.retrieved)} excerpts")
            return Plan(
                trace,
                stream=lambda: rag.stream_answer(
                    query.original, trace.retrieved, provider=provider, length=length
                ),
            )

    # --- Direct path ---
    say("Writing")
    trace.api_calls += 1
    return Plan(
        trace,
        stream=lambda: rag.stream_direct_answer(
            query.original, history, provider=provider, length=length
        ),
    )


def answer_now(plan: Plan) -> str:
    """Run a plan to completion. For scripts and the evaluation harness."""
    if plan.answer is not None:
        return plan.answer
    if plan.message is not None:
        return plan.message
    if plan.stream is None:
        return ""
    return "".join(plan.stream())
