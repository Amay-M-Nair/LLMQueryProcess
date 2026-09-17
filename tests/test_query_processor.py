"""The orchestrator and the rewriter, with the model stood in for by fakes."""

import numpy as np
import pytest

from backend import router
from backend.query_processor import answer_now, process
from backend.query_rewriter import needs_rewriting, rewrite
from backend.providers.base import Provider
from ingestion.chunker import Chunk
from vectorstore.faiss_store import VectorStore
from vectorstore.metadata_store import DocumentRecord


class FakeProvider(Provider):
    """Returns queued replies in order, and records what it was asked."""

    name = "fake"

    def __init__(self, *replies, raises=None):
        self.replies = list(replies)
        self.raises = raises
        self.prompts = []

    def stream(self, system, prompt):
        self.prompts.append((system, prompt))
        if self.raises:
            raise self.raises
        yield self.replies.pop(0) if self.replies else ""

    def check_ready(self):
        return None

    @property
    def calls(self):
        return len(self.prompts)


@pytest.fixture
def store():
    """A tiny real store - two chunks, real vectors, real FAISS."""
    chunks = [
        Chunk("Refunds are available within 30 days of purchase.",
              "refunds.pdf", 1, "refunds_page1_chunk0", 0, 8),
        Chunk("Employees accrue 25 days of paid annual leave each year.",
              "leave.pdf", 1, "leave_page1_chunk0", 0, 9),
    ]
    vectors = np.eye(len(chunks), 384, dtype="float32")
    built = VectorStore()
    for index, chunk in enumerate(chunks):
        built.add(
            vectors[index : index + 1],
            [chunk],
            DocumentRecord(chunk.source, f"hash{index}", 1, 1),
        )
    return built


# --- Routing, end to end ---------------------------------------------------

def test_arithmetic_costs_no_api_call(store):
    provider = FakeProvider()
    plan = process("what is 25% of 800?", store=store, provider=provider)
    assert plan.trace.route.name == router.CALCULATE
    assert plan.trace.api_calls == 0
    assert provider.calls == 0
    assert "200" in answer_now(plan)


def test_general_question_works_without_any_documents():
    """The whole point of routing: this used to be impossible."""
    provider = FakeProvider("Transformers are a neural architecture.")
    plan = process("explain transformers", store=None, provider=provider)
    assert plan.trace.route.name == router.DIRECT
    assert plan.sources == []
    assert answer_now(plan) == "Transformers are a neural architecture."


def test_direct_answer_prompt_forbids_citations():
    provider = FakeProvider("...")
    plan = process("explain transformers", store=None, provider=provider)
    answer_now(plan)
    system, _ = provider.prompts[-1]
    assert "nothing to cite" in system.lower()


def test_document_question_retrieves_and_cites(store):
    provider = FakeProvider('{"intent":"document_query","requires_retrieval":true}',
                            "Refunds within 30 days [1].")
    plan = process("what is the refund window?", store=store, provider=provider)
    assert plan.trace.route.name == router.RETRIEVAL
    assert plan.sources, "should have retrieved something"
    answer_now(plan)
    system, prompt = provider.prompts[-1]
    assert "[1]" in prompt, "excerpts must be numbered for citation"
    assert "cit" in system.lower()


def test_empty_question_asks_for_one(store):
    plan = process("   ", store=store, provider=FakeProvider())
    assert plan.message
    assert plan.stream is None


def test_generation_sees_the_original_wording(store):
    """Cleaning is for retrieval; the model gets what the user typed."""
    provider = FakeProvider("Transformers are...")
    plan = process("  explain   TRANSFORMERS  ", store=None, provider=provider)
    answer_now(plan)
    _, prompt = provider.prompts[-1]
    assert "TRANSFORMERS" in prompt


# --- The trace is the record both the UI and evaluation read ---------------

def test_trace_records_how_it_decided(store):
    provider = FakeProvider('{"intent":"document_query","requires_retrieval":true}', "x")
    plan = process("what is the refund window?", store=store, provider=provider)
    trace = plan.trace
    assert trace.intent.method in ("heuristic", "llm", "fallback")
    assert trace.route.reason
    assert "classify" in trace.timings
    assert trace.total_seconds >= 0


def test_api_calls_are_counted(store):
    provider = FakeProvider('{"intent":"document_query","requires_retrieval":true}', "x")
    plan = process("what is the refund window?", store=store, provider=provider)
    answer_now(plan)
    # One to classify, one to answer.
    assert plan.trace.api_calls == 2


def test_heuristic_route_counts_one_call_only(store):
    provider = FakeProvider("A summary.")
    plan = process("summarize this document", store=store, provider=provider)
    assert plan.trace.intent.method == "heuristic"
    assert plan.trace.api_calls == 1, "no classification call should have been spent"


# --- When retrieval finds nothing -----------------------------------------

def test_empty_retrieval_answers_unaided_and_says_so(store):
    provider = FakeProvider('{"intent":"document_query","requires_retrieval":true}',
                            "I do not have that.")
    plan = process(
        "what is the airspeed velocity of an unladen swallow?",
        store=store, provider=provider,
    )
    if not plan.sources:
        assert plan.trace.route.name == router.DIRECT
        assert any("nothing in the documents" in n for n in plan.trace.notes)


# --- Rewriting -------------------------------------------------------------

HISTORY = [{"question": "Explain Transformers.", "answer": "They use attention."}]


@pytest.mark.parametrize(
    "query, expected",
    [
        ("what about its limitations?", True),
        ("and why is that?", True),
        ("why?", True),
        ("tell me more about that", True),
        ("what is the refund policy for digital goods?", False),
        ("explain how attention works in detail", False),
    ],
)
def test_only_dependent_questions_are_rewritten(query, expected):
    assert needs_rewriting(query, HISTORY) is expected


def test_no_history_means_no_rewrite():
    assert not needs_rewriting("what about its limitations?", [])


def test_rewrite_resolves_the_reference():
    provider = FakeProvider("What are the limitations of Transformers?")
    result = rewrite("what about its limitations?", HISTORY, provider=provider)
    assert result.changed
    assert "Transformers" in result.query
    assert result.original == "what about its limitations?"


def test_rewrite_that_looks_like_an_answer_is_rejected():
    """A model that answers instead of rewriting must not poison the search."""
    provider = FakeProvider(
        "Transformers have several limitations. " * 40
    )
    result = rewrite("what about its limitations?", HISTORY, provider=provider)
    assert not result.changed
    assert result.query == "what about its limitations?"
    assert "unusable" in result.reason


def test_multiline_rewrite_is_rejected():
    provider = FakeProvider("Here is the rewrite:\nWhat are the limitations?")
    result = rewrite("what about its limitations?", HISTORY, provider=provider)
    assert not result.changed


def test_rewrite_failure_falls_back_to_the_original():
    provider = FakeProvider(raises=RuntimeError("network gone"))
    result = rewrite("what about its limitations?", HISTORY, provider=provider)
    assert not result.changed
    assert result.query == "what about its limitations?"


def test_standalone_question_never_costs_a_call():
    provider = FakeProvider("something")
    result = rewrite("what is the refund policy?", HISTORY, provider=provider)
    assert provider.calls == 0
    assert not result.changed
