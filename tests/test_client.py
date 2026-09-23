"""What the page hands the client, the client must hand the pipeline.

The page lets you choose a provider and type a key into the sidebar. Neither
is in config.py, and on a deployed copy there is no .env either - so if the
client builds its provider from config alone, the sidebar reports Ready from
one provider while a question is answered by another, or by none.

That is not hypothetical: it shipped. The sidebar said "Using the key you
entered" and every question came back "No Gemini API key found", because the
key stopped at the status line.
"""

from types import SimpleNamespace

import pytest

from api import collections
from api.client import HttpClient, LocalClient


@pytest.fixture(autouse=True)
def isolated_root(tmp_path, monkeypatch):
    monkeypatch.setattr(collections, "ROOT", tmp_path / "collections")


@pytest.fixture
def recorded(monkeypatch):
    """Record what the client asked for, and answer without a model."""
    seen = {}

    def fake_get_provider(name, api_key=None):
        seen["name"] = name
        seen["api_key"] = api_key
        return SimpleNamespace(name=name)

    def fake_process(question, store, history, provider, on_stage, length):
        seen["provider"] = provider
        return SimpleNamespace(
            answer="Canberra.", message=None, sources=[],
            trace=SimpleNamespace(
                intent=SimpleNamespace(name="general", method="heuristic", reason=""),
                route=SimpleNamespace(name="direct", reason=""),
                rewrite=None, api_calls=1, timings={}, notes=[],
            ),
        )

    import backend.llm
    import backend.query_processor

    monkeypatch.setattr(backend.llm, "get_provider", fake_get_provider)
    monkeypatch.setattr(backend.query_processor, "process", fake_process)
    return seen


def ask(**kwargs):
    return list(LocalClient().ask(
        "default", "What is the capital of Australia?",
        history=[], length="Standard", **kwargs,
    ))


def test_the_typed_key_reaches_the_provider(recorded):
    ask(provider_name="gemini", api_key="typed-into-the-sidebar")
    assert recorded["api_key"] == "typed-into-the-sidebar"


def test_the_chosen_provider_is_the_one_that_answers(recorded):
    """Choosing ollama in the sidebar must not answer with config.PROVIDER."""
    ask(provider_name="ollama", api_key=None)
    assert recorded["name"] == "ollama"


def test_no_choice_falls_back_to_config(recorded):
    from utils import config

    ask()
    assert recorded["name"] == config.PROVIDER
    assert recorded["api_key"] is None


def test_the_answer_still_comes_back(recorded):
    events = ask(provider_name="gemini", api_key="k")
    tokens = [e["text"] for e in events if e["type"] == "token"]
    assert tokens == ["Canberra."]


def test_a_remote_service_is_never_sent_the_key(monkeypatch):
    """A key typed into this page belongs to whoever typed it."""
    sent = {}

    class FakeResponse:
        def raise_for_status(self): pass
        def iter_lines(self): return iter(())
        def __enter__(self): return self
        def __exit__(self, *exc): return False

    class FakeHttp:
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def stream(self, method, url, data=None):
            sent.update(data or {})
            return FakeResponse()

    client = HttpClient("https://example.invalid")
    monkeypatch.setattr(client, "_client", lambda: FakeHttp())

    list(client.ask("default", "q", history=[], length="Standard",
                    provider_name="gemini", api_key="secret-key"))

    assert "secret-key" not in str(sent), "the key was forwarded to a third party"
