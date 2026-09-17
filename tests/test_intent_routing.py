"""Classification and routing, including the table from the project plan.

Nothing here reaches the network: the model is stood in for by fakes, so the
tests state exactly what a good reply, a bad reply and an outage each do.
"""

import pytest

from backend import router
from backend.intent_classifier import (
    CALCULATION,
    DOCUMENT_QUERY,
    GENERAL,
    SUMMARIZATION,
    UNKNOWN,
    classify,
    classify_by_rule,
    parse_reply,
)
from backend.llm import ProviderError, ProviderNotReady
from backend.providers.base import Provider

SOURCES = ["employee_handbook.pdf"]


class FakeProvider(Provider):
    """Replies with whatever it was given, and counts how often it was asked."""

    name = "fake"

    def __init__(self, reply="", raises=None):
        self.reply, self.raises, self.calls = reply, raises, 0

    def stream(self, system, prompt):
        self.calls += 1
        if self.raises:
            raise self.raises
        yield self.reply

    def check_ready(self):
        return None


def json_reply(intent, retrieval):
    return '{"intent": "%s", "requires_retrieval": %s}' % (
        intent, "true" if retrieval else "false"
    )


# --- The project plan's intent table (doc section 7) -----------------------

@pytest.mark.parametrize(
    "query, expected_intent, expects_retrieval, expected_route",
    [
        ("Explain transformers", GENERAL, False, router.DIRECT),
        ("What is 25% of 800?", CALCULATION, False, router.CALCULATE),
        ("What is our leave policy?", DOCUMENT_QUERY, True, router.RETRIEVAL),
        ("Summarize this PDF", SUMMARIZATION, True, router.RETRIEVAL),
    ],
)
def test_plan_intent_table(query, expected_intent, expects_retrieval, expected_route):
    provider = FakeProvider(json_reply(expected_intent, expects_retrieval))
    intent = classify(query, has_index=True, sources=SOURCES, provider=provider)
    assert intent.name == expected_intent
    assert intent.requires_retrieval is expects_retrieval
    assert router.route(intent, has_index=True).name == expected_route


# --- Heuristics settle the easy ones without paying for a call ------------

@pytest.mark.parametrize(
    "query, expected",
    [
        ("what is 25% of 800?", CALCULATION),
        ("2 + 2", CALCULATION),
        ("summarize this document", SUMMARIZATION),
        ("Summarise the PDF", SUMMARIZATION),
        ("tl;dr", SUMMARIZATION),
        ("give me an overview", SUMMARIZATION),
        ("what is this document about?", SUMMARIZATION),
        ("", UNKNOWN),
        ("   ", UNKNOWN),
    ],
)
def test_rules_decide_without_the_model(query, expected):
    provider = FakeProvider(json_reply(GENERAL, False))
    intent = classify(query, has_index=True, sources=SOURCES, provider=provider)
    assert intent.name == expected
    assert intent.method == "heuristic"
    assert provider.calls == 0, "a rule should have settled this for free"


def test_ambiguous_query_does_reach_the_model():
    provider = FakeProvider(json_reply(DOCUMENT_QUERY, True))
    intent = classify("what is the refund window?", has_index=True,
                      sources=SOURCES, provider=provider)
    assert provider.calls == 1
    assert intent.method == "llm"


def test_no_index_means_no_call_and_no_retrieval():
    """With nothing indexed there is nothing to decide - the model can't help."""
    provider = FakeProvider(json_reply(DOCUMENT_QUERY, True))
    intent = classify("what is our refund policy?", has_index=False, provider=provider)
    assert provider.calls == 0
    assert intent.name == GENERAL
    assert intent.requires_retrieval is False


def test_summary_request_with_nothing_indexed_is_unknown():
    intent = classify_by_rule("summarize this document", has_index=False)
    assert intent.name == UNKNOWN
    assert router.route(intent, has_index=False).name == router.CLARIFY


# --- Malformed model replies must still produce a usable route ------------

@pytest.mark.parametrize(
    "reply, expected",
    [
        ('{"intent": "general", "requires_retrieval": false}', (GENERAL, False)),
        ('```json\n{"intent": "general", "requires_retrieval": false}\n```',
         (GENERAL, False)),
        ('Here you go: {"intent": "document_query", "requires_retrieval": true}',
         (DOCUMENT_QUERY, True)),
        ('{"intent": "GENERAL", "requires_retrieval": false}', (GENERAL, False)),
        # Names the intent but garbles the flag - keep the judgement, derive
        # the rest.
        ('{"intent": "document_query", "requires_retrieval": "yes"}',
         (DOCUMENT_QUERY, True)),
    ],
)
def test_parses_imperfect_json(reply, expected):
    assert parse_reply(reply) == expected


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "I think this is a document query.",
        "{not json at all",
        '{"intent": "banana", "requires_retrieval": true}',
        '["general"]',
    ],
)
def test_unusable_replies_are_rejected(reply):
    assert parse_reply(reply) is None


def test_garbage_reply_falls_back_to_retrieval_when_documents_exist():
    """Grounded and cited is the recoverable mistake; answering from memory is not."""
    provider = FakeProvider("no idea, sorry")
    intent = classify("what is the refund window?", has_index=True,
                      sources=SOURCES, provider=provider)
    assert intent.name == DOCUMENT_QUERY
    assert intent.method == "fallback"
    assert router.route(intent, has_index=True).name == router.RETRIEVAL


@pytest.mark.parametrize("failure", [ProviderNotReady("no key"),
                                     ProviderError("503"),
                                     RuntimeError("socket closed")])
def test_provider_failure_never_sinks_the_question(failure):
    provider = FakeProvider(raises=failure)
    intent = classify("what is the refund window?", has_index=True,
                      sources=SOURCES, provider=provider)
    assert intent.method == "fallback"
    assert intent.name == DOCUMENT_QUERY


def test_no_provider_at_all_still_routes():
    intent = classify("what is the refund window?", has_index=True, provider=None)
    assert intent.method == "fallback"
    assert intent.name == DOCUMENT_QUERY


# --- The router's own guard -----------------------------------------------

def test_retrieval_is_downgraded_when_the_index_vanished():
    """A model can ask for documents that are no longer there."""
    provider = FakeProvider(json_reply(DOCUMENT_QUERY, True))
    intent = classify("what is the refund window?", has_index=True,
                      sources=SOURCES, provider=provider)
    decision = router.route(intent, has_index=False)
    assert decision.name == router.DIRECT
    assert decision.downgraded_from == DOCUMENT_QUERY


def test_model_asking_for_retrieval_without_an_index_is_corrected():
    provider = FakeProvider(json_reply(DOCUMENT_QUERY, True))
    intent = classify("anything at all here", has_index=False, provider=provider)
    assert intent.requires_retrieval is False


def test_route_flags_describe_the_path():
    provider = FakeProvider(json_reply(DOCUMENT_QUERY, True))
    retrieval = router.route(
        classify("refund window?", has_index=True, sources=SOURCES, provider=provider),
        has_index=True,
    )
    assert retrieval.cites_sources and retrieval.uses_model

    calculation = router.route(classify_by_rule("2+2", has_index=True), has_index=True)
    assert not calculation.uses_model, "arithmetic must not cost an API call"
    assert not calculation.cites_sources
