"""The offered questions must do what their labels say.

Each example is captioned with the route it takes - "worked out locally, no
API call", "general knowledge", "from your documents". If a question did not
actually route that way the app would be advertising behaviour it does not
have, on the first screen a new visitor sees.

So every question in every pool is routed, and the label checked against
where it went.
"""

import random

import pytest

from backend import intent_classifier, router
from utils import examples

DOCUMENT_NAME = "handbook.pdf"


@pytest.mark.parametrize("question", examples.CALCULATION)
def test_every_calculation_example_is_arithmetic(question):
    """The caption promises no API call, so it must never reach a model."""
    intent = intent_classifier.classify_by_rule(question, has_index=True)
    assert intent is not None, f"{question!r} would have been sent to the model"
    assert intent.name == intent_classifier.CALCULATION
    assert router.route(intent, has_index=True).name == router.CALCULATE


@pytest.mark.parametrize("question", examples.CALCULATION)
def test_every_calculation_example_computes(question):
    from backend import calculator

    _, value = calculator.calculate(question)
    assert isinstance(value, (int, float))


@pytest.mark.parametrize("question", examples.GENERAL)
def test_no_general_example_is_mistaken_for_arithmetic(question):
    """A general question caught by the calculator would answer as a number."""
    from backend import calculator

    assert not calculator.looks_arithmetic(question)


@pytest.mark.parametrize("question", examples.GENERAL)
def test_general_examples_answer_without_documents(question):
    """With nothing indexed these must still route somewhere that answers."""
    intent = intent_classifier.classify(question, has_index=False, provider=None)
    assert router.route(intent, has_index=False).name == router.DIRECT


@pytest.mark.parametrize("template", examples.DOCUMENT)
def test_document_examples_name_the_file(template):
    assert "{name}" in template, "the question does not say which document"
    filled = template.format(name=DOCUMENT_NAME)
    assert DOCUMENT_NAME in filled


@pytest.mark.parametrize("template", examples.DOCUMENT)
def test_document_examples_assume_nothing_about_the_contents(template):
    """A question about leave policy is noise to someone who uploaded a thesis."""
    filled = template.format(name=DOCUMENT_NAME).lower()
    for assumed in ("leave", "refund", "salary", "invoice", "my experience"):
        assert assumed not in filled, f"assumes the document is about {assumed!r}"


# --- Picking ---------------------------------------------------------------

def test_pick_returns_one_per_route():
    chosen = examples.pick(DOCUMENT_NAME)
    assert len(chosen) == 3
    notes = [note for _, note in chosen]
    assert notes == [
        examples.NOTE_DOCUMENT, examples.NOTE_GENERAL, examples.NOTE_CALCULATION
    ]


def test_without_a_document_the_first_slot_explains_instead_of_asking():
    first, note = examples.pick(None)[0]
    assert note is None, "an unanswerable question was offered as a question"
    assert first == examples.NO_DOCUMENTS


def test_the_document_name_is_filled_in():
    question, _ = examples.pick(DOCUMENT_NAME)[0]
    assert DOCUMENT_NAME in question
    assert "{name}" not in question


def test_the_selection_actually_varies():
    """Fixed examples become furniture; that is the point of drawing them."""
    seen = {tuple(q for q, _ in examples.pick(DOCUMENT_NAME)) for _ in range(40)}
    assert len(seen) > 1, "the same three questions came back every time"


def test_a_seeded_draw_is_repeatable():
    """So a rerun can be given the same three rather than reshuffling."""
    first = examples.pick(DOCUMENT_NAME, rng=random.Random(11))
    second = examples.pick(DOCUMENT_NAME, rng=random.Random(11))
    assert first == second


def test_no_question_is_empty_or_unterminated():
    for pool in (examples.GENERAL, examples.CALCULATION):
        for question in pool:
            assert question.strip()
            assert not question.endswith(" ")
