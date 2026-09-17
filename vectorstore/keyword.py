"""Keyword (BM25) scoring, to sit alongside vector search.

Vector search matches *meaning*, which fails on lookups where the wording is
what matters: "how do I contact this person" is semantically far from a wall
of ML jargon, even when the chunk literally contains an email address.
BM25 catches exactly those cases, so the two are used together.

BM25 scores a chunk by how many query words it contains, weighting rare words
more heavily than common ones and discounting very long chunks.
"""

import math
import re
from collections import Counter, defaultdict

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9@.+-]+")

K1 = 1.5   # term-frequency saturation
B = 0.75   # length normalisation

# Words carrying no retrieval signal. Normally idf handles these by itself -
# a word in every chunk scores near zero - but that only holds for ordinary
# prose. In a corpus of code, logs, tables or fragments, "the" can be rare
# enough to look discriminating, and then it drags unrelated chunks up the
# ranking. Dropping them outright costs nothing: BM25 matches bags of words,
# never phrases, so no query loses meaning it could have used.
STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has have
he her him his how i if in into is it its me my no nor not of on or our out
she should so some such than that the their them then there these they this
those to too was we were what when where which who whom why will with would
you your
""".split())


def tokenize(text: str) -> list[str]:
    return [w for w in TOKEN_RE.findall(text.lower()) if w not in STOPWORDS]


class BM25:
    """Scores documents against a query.

    The postings map (word -> the chunks containing it) means scoring costs
    time proportional to the number of chunks that actually contain a query
    word, not to the size of the whole index. A query word nobody uses costs
    nothing at all.
    """

    def __init__(self, documents: list[str]):
        tokenized = [tokenize(d) for d in documents]
        self.count = len(tokenized)
        self.lengths = np.array([len(d) for d in tokenized], dtype="float32")
        self.avg_length = float(self.lengths.mean()) if self.count else 0.0

        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, doc in enumerate(tokenized):
            for word, freq in Counter(doc).items():
                self.postings[word].append((index, freq))

        self.idf = {
            word: math.log(1 + (self.count - len(posting) + 0.5) / (len(posting) + 0.5))
            for word, posting in self.postings.items()
        }

    def scores(self, query: str) -> np.ndarray:
        """BM25 score for every document, in index order."""
        result = np.zeros(self.count, dtype="float32")
        if not self.count:
            return result

        for word in tokenize(query):
            idf = self.idf.get(word)
            if idf is None:
                continue
            for index, freq in self.postings[word]:
                norm = 1 - B + B * (self.lengths[index] / (self.avg_length or 1))
                result[index] += idf * (freq * (K1 + 1)) / (freq + K1 * norm)
        return result
