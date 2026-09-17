"""Tidy a question without changing what it asks.

The temptation here is to do what search engines used to do - lowercase,
strip punctuation, drop stopwords. That would be a mistake. Everything
downstream is either an embedding model or an LLM, and both read the question
as language: "What is NOT covered?" and "what is covered" are opposites that
aggressive cleaning would collapse into each other.

So this only removes noise that carries no meaning: stray whitespace from a
copy-paste, and the invisible unicode variants of ordinary characters.
"""

import re
import unicodedata
from dataclasses import dataclass

# A question this short is usually a fragment ("limitations?", "why") that
# only makes sense against what was said before.
SHORT_QUERY_WORDS = 4


@dataclass
class Query:
    """A question in both the forms the pipeline needs.

    `original` is what the user typed and what the answer is generated from -
    their exact wording carries emphasis and intent that a cleaned copy loses.
    `cleaned` is what retrieval and classification see.
    """

    original: str
    cleaned: str

    @property
    def is_empty(self) -> bool:
        return not self.cleaned

    @property
    def word_count(self) -> int:
        return len(self.cleaned.split())

    @property
    def is_short(self) -> bool:
        """Short enough that it probably leans on the conversation so far."""
        return 0 < self.word_count <= SHORT_QUERY_WORDS


def preprocess(text: str) -> Query:
    """Normalise whitespace and unicode form, and nothing else."""
    original = text if text is not None else ""

    # NFKC folds the lookalikes a paste from a PDF or a chat app drags in -
    # curly quotes, non-breaking spaces, full-width digits - onto the plain
    # characters the rest of the system expects.
    cleaned = unicodedata.normalize("NFKC", original)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return Query(original=original, cleaned=cleaned)
