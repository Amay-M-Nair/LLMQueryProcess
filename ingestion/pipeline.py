"""Wire the loading side together: files in, saved index out."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ingestion import document_loader as loaders
from ingestion import embedder
from ingestion.chunker import Chunk, chunk_pages
from utils import config
from vectorstore.faiss_store import VectorStore


@dataclass
class IngestReport:
    """What happened during an ingest, so the UI can report it honestly."""

    indexed: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (name, reason)
    chunks_added: int = 0
    total_chunks: int = 0


def build_index(
    paths: list[Path],
    existing: VectorStore | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[VectorStore, IngestReport]:
    """Read each file, chunk it, embed the chunks, add them to the store.

    Files whose name is already in the index are skipped, so re-running this
    after adding one new document is cheap.
    """
    report = IngestReport()
    already_indexed = set(existing.sources) if existing else set()
    new_chunks: list[Chunk] = []

    for path in paths:
        path = Path(path)
        name = path.name

        if name in already_indexed:
            report.skipped.append((name, "already indexed"))
            continue

        if on_progress:
            on_progress(f"Reading {name}")

        try:
            pages = loaders.load_pages(path)
        except Exception as exc:  # a corrupt or unreadable file shouldn't kill the run
            report.skipped.append((name, f"could not read: {exc}"))
            continue

        if loaders.looks_scanned(path, pages):
            report.skipped.append(
                (name, "no extractable text - looks like a scanned PDF, needs OCR")
            )
            continue

        if not pages:
            report.skipped.append((name, "file is empty"))
            continue

        chunks = chunk_pages(pages, source=name)
        if not chunks:
            report.skipped.append((name, "produced no text chunks"))
            continue

        new_chunks.extend(chunks)
        report.indexed.append(name)

    if new_chunks:
        capacity = embedder.max_input_words()
        if config.CHUNK_WORDS > capacity:
            report.skipped.append((
                "config",
                f"CHUNK_WORDS is {config.CHUNK_WORDS} but the embedding model "
                f"only reads the first ~{capacity} words of a chunk - the rest "
                "is invisible to search. Lower CHUNK_WORDS in utils/config.py.",
            ))
        if on_progress:
            on_progress(f"Embedding {len(new_chunks)} chunks")
        vectors = embedder.embed([c.text for c in new_chunks])

        if existing and len(existing) > 0:
            vectors = np.vstack([existing.vectors, vectors])
            chunks = existing.chunks + new_chunks
        else:
            chunks = new_chunks

        store = VectorStore(vectors, chunks)
    else:
        store = existing or VectorStore(
            np.zeros((0, config.EMBED_DIM), dtype="float32"), []
        )

    report.chunks_added = len(new_chunks)
    report.total_chunks = len(store)
    return store, report


def retrieve(
    store: VectorStore,
    question: str,
    k: int | None = None,
    min_similarity: float | None = None,
) -> list[tuple[Chunk, float]]:
    """Pull the chunks worth showing the model.

    When the whole index is small enough to fit in a prompt, retrieval is
    skipped entirely and everything is returned. Retrieval exists to compress
    a corpus that cannot fit in the context window; running it over a two-page
    document only creates a chance to drop the answer. This is what makes
    questions *about* a document ("who wrote this?") work - a question whose
    words appear nowhere in the text cannot be retrieved by any method.
    """
    k = config.TOP_K if k is None else k
    min_similarity = (
        config.MIN_SIMILARITY if min_similarity is None else min_similarity
    )

    if len(store) == 0:
        return []

    query_vector = embedder.embed([question])[0]

    if store.total_words <= config.FULL_CONTEXT_WORDS:
        # Small enough to send whole - rank for readable citation order only.
        return store.search(query_vector, len(store), query=question)

    hits = store.search(query_vector, k, query=question)
    return [(chunk, score) for chunk, score in hits if score >= min_similarity]
