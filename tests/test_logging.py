"""Logs must carry the detail and none of the secrets.

Two things must never reach a log file. API keys, because logs get pasted
into issues and sent to people helping with a problem. And the contents of
anyone's documents, because a log file is not where someone's private files
belong - the whole point of this project is that those files stay theirs.

These run the real paths with a planted secret and assert it never appears.
"""

import logging

import pytest

from backend import intent_classifier, query_rewriter
from backend.providers.base import Provider, ProviderError
from ingestion.pipeline import build_index
from utils import logs

KEY = "AQ.super-secret-key-value-9f2b"
PRIVATE = "The vault access code is BLUEJAY-7741 and the keyholder is Okonkwo."


@pytest.fixture
def captured():
    """Everything written to the azriel logger during a test."""
    logs.setup()
    logger = logging.getLogger("azriel")
    records = []

    class Collect(logging.Handler):
        def emit(self, record):
            records.append(self.format(record))

    handler = Collect()
    handler.setFormatter(logging.Formatter("%(message)s"))
    previous = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


class Exploding(Provider):
    """Fails with the key in the message, the way a real client would."""

    name = "exploding"

    def __init__(self, message):
        self.message = message

    def stream(self, system, prompt):
        raise ProviderError(self.message)
        yield  # pragma: no cover

    def check_ready(self):
        return None


def test_ingest_logs_names_not_contents(tmp_path, captured):
    document = tmp_path / "private_note.txt"
    document.write_text(PRIVATE, encoding="utf-8")
    build_index([document])

    written = "\n".join(captured)
    assert "private_note.txt" in written, "the file name is worth logging"
    assert "BLUEJAY-7741" not in written, "document contents reached the log"
    assert "Okonkwo" not in written


def test_routing_logs_the_decision_not_the_question(captured):
    from backend.query_processor import process

    process("What is the vault access code BLUEJAY-7741?", store=None, provider=None)
    written = "\n".join(captured)
    assert "intent=" in written, "the decision is worth logging"
    assert "BLUEJAY-7741" not in written, "the question reached the log"


def test_a_failed_rewrite_is_logged_without_the_question(captured):
    history = [{"question": "Explain Transformers.", "answer": "They use attention."}]
    result = query_rewriter.rewrite(
        "what about its " + PRIVATE, history, provider=Exploding("upstream is down")
    )
    written = "\n".join(captured)
    assert not result.changed
    assert "rewrite" in written.lower(), "a swallowed failure left no trace"
    assert "BLUEJAY-7741" not in written


def test_a_classifier_key_in_an_error_does_not_reach_the_log(captured):
    """Clients put the key in the URL, so it turns up in their exceptions."""
    provider = Exploding(f"401 from https://api.example.com/v1?key={KEY}")
    intent = intent_classifier.classify(
        "what is the refund window?", has_index=True, provider=provider
    )

    assert intent.method == "fallback", "the question should still be routed"
    assert KEY not in "\n".join(captured), "an API key reached the log"


def test_setup_is_idempotent():
    """Streamlit re-runs the script on every interaction."""
    logs.setup()
    before = len(logging.getLogger("azriel").handlers)
    for _ in range(5):
        logs.setup()
    assert len(logging.getLogger("azriel").handlers) == before


def test_loggers_are_namespaced():
    assert logs.get("ingestion.pipeline").name == "azriel.pipeline"
    assert logs.get(__name__).name.startswith("azriel.")


# --- Redaction -------------------------------------------------------------

@pytest.mark.parametrize(
    "message, must_go",
    [
        ("GET https://api.example.com/v1?key=AIzaSyFAKE123456789 failed", "AIzaSyFAKE123456789"),
        ("401 from ?api_key=sk-test-abcdefghij", "sk-test-abcdefghij"),
        ("denied ?access_token=ya29.a0PRIVATE&alt=json", "ya29.a0PRIVATE"),
    ],
)
def test_a_key_in_a_url_is_removed(message, must_go):
    """Covers keys this process has never seen, so cannot match by value."""
    cleaned = logs.redact(message)
    assert must_go not in cleaned
    assert "<redacted>" in cleaned


def test_redaction_keeps_the_message_readable():
    """Removing the key should not take the sentence with it."""
    cleaned = logs.redact("GET https://api.example.com/v1?key=AIzaSyFAKE123 failed")
    assert cleaned == "GET https://api.example.com/v1?key=<redacted> failed"


def test_redaction_introduces_no_control_characters():
    """A backreference written wrongly substitutes a raw byte instead."""
    cleaned = logs.redact("?key=AIzaSyFAKE123456789")
    assert all(ch == "\n" or ch >= " " for ch in cleaned), repr(cleaned)


def test_a_known_key_is_removed_wherever_it_appears(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", KEY)
    assert KEY not in logs.redact(f"the key {KEY} was rejected")
    assert KEY not in logs.redact(f"Authorization: Bearer {KEY}")


def test_ordinary_messages_are_left_alone():
    plain = "ingest: indexed ['handbook.txt'], 15 chunks added"
    assert logs.redact(plain) == plain


def test_a_short_env_value_is_not_treated_as_a_key(monkeypatch):
    """Redacting a two-character value would blank half the log."""
    monkeypatch.setenv("GOOGLE_API_KEY", "ab")
    assert logs.redact("a cab drove past") == "a cab drove past"
