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
from collections import Counter

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9@.+-]+")

K1 = 1.5   # term-frequency saturation
B = 0.75   # length normalisation


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, documents: list[str]):
        self.docs = [tokenize(d) for d in documents]
        self.lengths = np.array([len(d) for d in self.docs], dtype="float32")
        self.avg_length = float(self.lengths.mean()) if len(self.docs) else 0.0
        self.counts = [Counter(d) for d in self.docs]

        # In how many chunks does each word appear?
        seen = Counter()
        for doc in self.docs:
            seen.update(set(doc))
        total = len(self.docs)
        self.idf = {
            word: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
            for word, freq in seen.items()
        }

    def scores(self, query: str) -> np.ndarray:
        result = np.zeros(len(self.docs), dtype="float32")
        if not len(self.docs):
            return result

        for word in tokenize(query):
            idf = self.idf.get(word)
            if idf is None:
                continue
            for index, counts in enumerate(self.counts):
                freq = counts.get(word, 0)
                if not freq:
                    continue
                norm = 1 - B + B * (self.lengths[index] / (self.avg_length or 1))
                result[index] += idf * (freq * (K1 + 1)) / (freq + K1 * norm)
        return result
