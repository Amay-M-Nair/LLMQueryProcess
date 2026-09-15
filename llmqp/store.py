"""A minimal vector store: one numpy array of vectors, one list of chunks.

Plenty fast for tens of thousands of chunks. If this project ever outgrows
that, this is the single file you would swap for FAISS or Chroma.
"""

import json
from pathlib import Path

import numpy as np

from . import config
from .chunker import Chunk
from .keyword import BM25

VECTORS_FILE = "vectors.npy"
CHUNKS_FILE = "chunks.json"


class VectorStore:
    def __init__(self, vectors: np.ndarray, chunks: list[Chunk]):
        if len(vectors) != len(chunks):
            raise ValueError(
                f"{len(vectors)} vectors but {len(chunks)} chunks - they must match"
            )
        self.vectors = vectors
        self.chunks = chunks
        self._bm25: BM25 | None = None

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def sources(self) -> list[str]:
        """Distinct file names in the index, in insertion order."""
        seen: dict[str, None] = {}
        for chunk in self.chunks:
            seen.setdefault(chunk.source, None)
        return list(seen)

    @property
    def total_words(self) -> int:
        return sum(len(chunk.text.split()) for chunk in self.chunks)

    @property
    def bm25(self) -> BM25:
        """Built on first use and cached - cheap, but not free."""
        if self._bm25 is None:
            self._bm25 = BM25([chunk.text for chunk in self.chunks])
        return self._bm25

    def search(
        self, query_vector: np.ndarray, k: int, query: str | None = None
    ) -> list[tuple[Chunk, float]]:
        """Rank chunks by meaning, and by keyword overlap when a query is given.

        The returned score is always the cosine similarity, so it stays
        comparable and readable in the UI - the keyword signal affects the
        ordering, not the number shown.
        """
        if len(self) == 0:
            return []

        # Both sides are unit vectors, so this dot product IS cosine similarity.
        cosine = self.vectors @ query_vector.ravel()
        ranking = cosine

        if query:
            lexical = self.bm25.scores(query)
            if lexical.max() > 0:
                # Put both on a 0-1 scale before blending, since BM25 scores
                # are unbounded while cosine is not.
                lexical = lexical / lexical.max()
                spread = cosine.max() - cosine.min()
                dense = (cosine - cosine.min()) / spread if spread else cosine * 0
                ranking = (1 - config.KEYWORD_WEIGHT) * dense + config.KEYWORD_WEIGHT * lexical

        k = min(k, len(ranking))
        # argpartition finds the top k without sorting all of them, then we
        # sort just those k.
        top = np.argpartition(-ranking, k - 1)[:k]
        top = top[np.argsort(-ranking[top])]

        return [(self.chunks[i], float(cosine[i])) for i in top]

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / VECTORS_FILE, self.vectors)
        (directory / CHUNKS_FILE).write_text(
            json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> "VectorStore | None":
        """Load a saved index, or None if there isn't one yet."""
        directory = Path(directory)
        vectors_path = directory / VECTORS_FILE
        chunks_path = directory / CHUNKS_FILE
        if not vectors_path.exists() or not chunks_path.exists():
            return None

        vectors = np.load(vectors_path)
        raw = json.loads(chunks_path.read_text(encoding="utf-8"))
        return cls(vectors, [Chunk(**item) for item in raw])
