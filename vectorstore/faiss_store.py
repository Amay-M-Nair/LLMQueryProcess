"""The vector index: FAISS for the vectors, MetadataStore for everything else.

Vectors arrive unit-normalised from the embedder, which is what makes
`IndexFlatIP` (inner product) return exactly the cosine similarity - no
approximation, no training step, and the numbers shown in the UI mean what
they say.

`IndexFlat` also stores the vectors themselves, so it is the single source of
truth: removing a document rebuilds the index from the rows that survive
rather than keeping a shadow copy of everything in numpy.
"""

from pathlib import Path

import numpy as np

from ingestion.chunker import Chunk
from utils import config
from vectorstore.metadata_store import DocumentRecord, MetadataStore

INDEX_FILE = "index.faiss"


def _new_index():
    import faiss

    return faiss.IndexFlatIP(config.EMBED_DIM)


class VectorStore:
    def __init__(self, index=None, metadata: MetadataStore | None = None):
        self.index = index if index is not None else _new_index()
        self.metadata = metadata if metadata is not None else MetadataStore()
        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                f"{self.index.ntotal} vectors but {len(self.metadata)} chunks - "
                "the index and its metadata are out of step. Clear the index "
                "and re-add your documents."
            )

    def __len__(self) -> int:
        return self.index.ntotal

    # --- Read-through to the metadata -------------------------------------

    @property
    def chunks(self) -> list[Chunk]:
        return self.metadata.chunks

    @property
    def sources(self) -> list[str]:
        return self.metadata.sources

    @property
    def documents(self) -> dict[str, DocumentRecord]:
        return self.metadata.documents

    @property
    def total_words(self) -> int:
        return self.metadata.total_words

    def is_current(self, name: str, content_hash: str) -> bool:
        return self.metadata.is_current(name, content_hash)

    def has_name(self, name: str) -> bool:
        return self.metadata.has_name(name)

    # --- Writing -----------------------------------------------------------

    def add(
        self, vectors: np.ndarray, chunks: list[Chunk], document: DocumentRecord
    ) -> None:
        if len(vectors) != len(chunks):
            raise ValueError(f"{len(vectors)} vectors but {len(chunks)} chunks")
        self.index.add(np.ascontiguousarray(vectors, dtype="float32"))
        self.metadata.add(chunks, document)

    def remove_document(self, name: str) -> int:
        """Drop a document and its vectors. Returns how many chunks went.

        `IndexFlat` supports `remove_ids`, but it renumbers the rows left
        behind, which is exactly the kind of implicit reindexing that gets the
        metadata out of step. Rebuilding from the surviving vectors is a few
        milliseconds at this scale and cannot drift.
        """
        doomed = set(self.metadata.positions_for(name))
        if not doomed:
            return 0

        keep = [i for i in range(len(self.metadata)) if i not in doomed]
        survivors = (
            self.index.reconstruct_n(0, self.index.ntotal)[keep]
            if keep
            else np.zeros((0, config.EMBED_DIM), dtype="float32")
        )

        self.index = _new_index()
        if len(survivors):
            self.index.add(np.ascontiguousarray(survivors, dtype="float32"))
        self.metadata.keep(keep)
        return len(doomed)

    # --- Searching ---------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        query: str | None = None,
        min_similarity: float | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Rank chunks by meaning, and by keyword overlap when a query is given.

        The returned score is always the cosine similarity, so it stays
        comparable and readable in the UI - the keyword signal affects the
        ordering, not the number shown. Ordering by one number and reporting
        another does mean the scores you see need not descend.

        `min_similarity` drops chunks nothing like the question. It is applied
        here rather than by the caller because this is the only place that can
        see both signals: a chunk can be a decisive keyword match and still
        score poorly on cosine, and dropping it on the cosine alone would
        throw away the exact case BM25 is here to catch.
        """
        total = len(self)
        if total == 0:
            return []

        k = min(k, total)

        # Blending BM25 needs a candidate set wide enough that a chunk which
        # wins on keywords is in it at all. Below the limit we simply take
        # every chunk, which makes the blend exact; above it, a pool.
        if query and total <= config.HYBRID_EXACT_LIMIT:
            pool = total
        elif query:
            pool = min(total, max(k * config.HYBRID_POOL_MULTIPLIER, config.HYBRID_POOL_MIN))
        else:
            pool = k

        probe = np.ascontiguousarray(query_vector, dtype="float32").reshape(1, -1)
        cosine, ids = self.index.search(probe, pool)
        cosine, ids = cosine[0], ids[0]

        # FAISS pads the tail with -1 when it finds fewer than `pool` results.
        found = ids >= 0
        cosine, ids = cosine[found], ids[found]
        if not len(ids):
            return []

        lexical = None
        if query:
            raw = self.metadata.bm25.scores(query)[ids]
            if raw.max() > 0:
                # Put both on a 0-1 scale before blending, since BM25 scores
                # are unbounded while cosine is not.
                lexical = raw / raw.max()
                spread = cosine.max() - cosine.min()
                dense = (cosine - cosine.min()) / spread if spread else cosine * 0
                ranking = (
                    (1 - config.KEYWORD_WEIGHT) * dense
                    + config.KEYWORD_WEIGHT * lexical
                )
                order = np.argsort(-ranking)
                cosine, ids, lexical = cosine[order], ids[order], lexical[order]

        if min_similarity is not None:
            related = cosine >= min_similarity
            if lexical is not None:
                # Rescue the strong keyword matches. BM25 already discounts
                # words that appear everywhere, so a chunk only gets near the
                # top of this scale by containing the distinctive words of the
                # question - which is reason enough to show it, whatever the
                # vectors think.
                related |= lexical >= config.KEYWORD_RESCUE
            cosine, ids = cosine[related], ids[related]

        return [
            (self.metadata.chunks[int(i)], float(score))
            for i, score in zip(ids[:k], cosine[:k])
        ]

    # --- Persistence -------------------------------------------------------

    def save(self, directory: Path) -> None:
        import faiss

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / INDEX_FILE))
        self.metadata.save(directory)

    @classmethod
    def load(cls, directory: Path) -> "VectorStore | None":
        """Load a saved index, or None if there isn't one yet."""
        import faiss

        directory = Path(directory)
        index_path = directory / INDEX_FILE
        metadata = MetadataStore.load(directory)
        if not index_path.exists() or metadata is None:
            return None

        return cls(faiss.read_index(str(index_path)), metadata)
