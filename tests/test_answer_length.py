"""Answer length is a prompt setting, not a token cap.

The models allow 65536 output tokens and the app caps at 8192, so length has
never been limited by what is permitted - it is limited by what is asked for.
These check that the asking actually changes, and that the grounding rules
survive every setting.
"""

import pytest

from backend import rag
from backend.providers.base import Provider
from utils import prompts


class Recorder(Provider):
    """Captures the system prompt it was handed."""

    name = "recorder"

    def __init__(self):
        self.system = None

    def stream(self, system, prompt):
        self.system = system
        yield "answer"

    def check_ready(self):
        return None


@pytest.mark.parametrize("length", list(prompts.LENGTHS))
def test_every_length_produces_a_usable_prompt(length):
    for build in (prompts.rag_system, prompts.direct_system):
        text = build(length)
        assert "{length}" not in text, "the slot was never filled"
        assert text.strip()


def test_the_three_lengths_differ():
    asked = {name: prompts.rag_system(name) for name in prompts.LENGTHS}
    assert len(set(asked.values())) == len(asked), "two settings ask for the same thing"


def test_thorough_asks_for_more_than_brief():
    assert len(prompts.rag_system("Thorough")) > len(prompts.rag_system("Brief"))


@pytest.mark.parametrize("length", list(prompts.LENGTHS))
def test_the_grounding_rules_survive_every_length(length):
    """Length must never be bought by loosening what may be claimed."""
    text = prompts.rag_system(length)
    assert "Beyond your documents" in text
    assert "Never attach [n] to a claim its excerpt does not support" in text
    assert "the excerpts win" in text


def test_thorough_forbids_padding_on_the_cited_path():
    """Longer has to come from the excerpts, not from restating them."""
    assert "never from restating" in prompts.rag_system("Thorough")


def test_an_unknown_length_falls_back_rather_than_raising():
    assert prompts.rag_system("Enormous") == prompts.rag_system(prompts.DEFAULT_LENGTH)


def test_length_reaches_the_provider():
    for length in prompts.LENGTHS:
        recorder = Recorder()
        list(rag.stream_answer("q", [], provider=recorder, length=length))
        assert recorder.system == prompts.rag_system(length)

        recorder = Recorder()
        list(rag.stream_direct_answer("q", [], provider=recorder, length=length))
        assert recorder.system == prompts.direct_system(length)


def test_the_default_is_unchanged_from_before():
    """Standard must still be the wording the evaluation was measured against."""
    assert "Be concise and direct" in prompts.rag_system("Standard")
