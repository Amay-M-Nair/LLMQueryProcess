"""Preprocessing must tidy without editing the question."""

import pytest

from utils.preprocessing import preprocess


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("  what is the policy?  ", "what is the policy?"),
        ("what\n\nis   the\tpolicy?", "what is the policy?"),
        ("what is the policy?", "what is the policy?"),
        ("", ""),
        ("    ", ""),
    ],
)
def test_collapses_whitespace(raw, expected):
    assert preprocess(raw).cleaned == expected


def test_normalises_unicode_lookalikes():
    """Curly quotes and non-breaking spaces arrive with every PDF paste."""
    assert preprocess("what is this?").cleaned == "what is this?"
    assert preprocess("２５% of 800").cleaned == "25% of 800"


def test_keeps_the_original_verbatim():
    raw = "  What is NOT covered?  "
    query = preprocess(raw)
    assert query.original == raw
    assert query.cleaned == "What is NOT covered?"


def test_does_not_lowercase_or_strip_punctuation():
    """Casing and negation carry meaning the rest of the pipeline reads."""
    query = preprocess("What is NOT covered?")
    assert "NOT" in query.cleaned
    assert query.cleaned.endswith("?")


def test_flags_empty():
    assert preprocess("   ").is_empty
    assert not preprocess("hello").is_empty


@pytest.mark.parametrize(
    "raw, short",
    [
        ("why?", True),
        ("what about its limitations?", True),
        ("", False),  # empty is its own case, not a short one
        ("what is the refund policy for digital goods", False),
    ],
)
def test_flags_short_queries(raw, short):
    assert preprocess(raw).is_short is short


def test_handles_none():
    assert preprocess(None).cleaned == ""
