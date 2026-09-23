"""Conversation memory: what the model sees, and what it must not cite.

Rewriting fixes the search; it does nothing for the answer. "Where does that
figure come from?" can retrieve exactly the right excerpts and still read as
a non-sequitur, because the model writing the reply never saw what it was
being asked about.

Passing the conversation in creates the opposite hazard: a model given
sources and a transcript in one prompt will cite the transcript. So these
check both that history arrives and that it is kept apart from the evidence.
"""

import pytest

from backend import rag
from backend.providers.base import Provider
from utils import config, prompts

HISTORY = [
    {"question": "How many days of annual leave do employees get?",
     "answer": "Full-time employees accrue 25 days a year [1]."},
    {"question": "And how many can be carried over?",
     "answer": "Up to five days [1]."},
]


class Recorder(Provider):
    name = "recorder"

    def __init__(self):
        self.system = None
        self.prompt = None

    def stream(self, system, prompt):
        self.system, self.prompt = system, prompt
        yield "answer"

    def check_ready(self):
        return None


def test_the_retrieval_path_receives_the_conversation():
    recorder = Recorder()
    list(rag.stream_answer("Where does that figure come from?", [],
                           provider=recorder, history=HISTORY))
    assert "25 days a year" in recorder.prompt, "the answer had no idea what was asked"


def test_the_conversation_is_kept_out_of_the_excerpts():
    """Run together, a model cites the transcript as though it were a source."""
    built = rag.build_prompt("Where does that come from?", [], HISTORY)
    conversation = built.index("Conversation so far")
    excerpts = built.index("Here are the source excerpts")
    assert conversation < excerpts
    assert "---" in built[conversation:excerpts], "the two blocks are not separated"


def test_the_prompt_forbids_citing_the_conversation():
    system = prompts.rag_system()
    assert "not evidence" in system.lower()
    assert "never cite it" in system.lower()


def test_no_conversation_means_no_empty_heading():
    """An empty 'Conversation so far:' invites the model to invent one."""
    built = rag.build_prompt("What is the leave allowance?", [])
    assert "Conversation so far" not in built


def test_only_the_recent_turns_are_carried():
    """Memory is a window; an unbounded transcript crowds out the excerpts."""
    long_history = [
        {"question": f"question {n}", "answer": f"answer {n}"} for n in range(20)
    ]
    built = rag.build_prompt("and now?", [], long_history)
    assert "question 19" in built
    assert "question 0" not in built, "the whole transcript was pasted in"
    assert built.count("User:") == config.HISTORY_TURNS


def test_a_long_answer_is_trimmed_in_the_recap():
    """The gist resolves a pronoun; the full text would crowd out the sources."""
    verbose = [{"question": "explain", "answer": "word " * 500}]
    built = rag.build_prompt("and why?", [], verbose)
    assert len(built) < 3000, "an old answer was pasted in whole"


def test_the_direct_path_still_gets_history_too():
    recorder = Recorder()
    list(rag.stream_direct_answer("and why is that?", HISTORY, provider=recorder))
    assert "25 days a year" in recorder.prompt


def test_history_does_not_leak_into_the_system_prompt():
    """The contract is fixed text; the conversation belongs in the user turn."""
    recorder = Recorder()
    list(rag.stream_answer("q", [], provider=recorder, history=HISTORY))
    assert "25 days a year" not in recorder.system
