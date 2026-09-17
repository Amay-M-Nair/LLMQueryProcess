"""Wire the loading side together: files in, saved index out."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ingestion import document_loader as loaders
from ingestion import embedder
from ingestion.chunker import Chunk, chunk_pages
from utils import config
from vectorstore.faiss_store import VectorStore
from vectorstore.metadata_store import DocumentRecord, hash_file


@dataclass
class IngestReport:
    """What happened during an ingest, so the UI can report it honestly."""

    indexed: list[str] = field(default_factory=list)
    reindexed: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (name, reason)
    chunks_added: int = 0
    total_chunks: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.indexed or self.reindexed)


def build_index(
    paths: list[Path],
    existing: VectorStore | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[VectorStore, IngestReport]:
    """Read each file, chunk it, embed the chunks, add them to the store.

    A file is identified by its contents, not its name. Re-adding the same
    bytes is skipped; re-adding a file whose contents changed replaces the
    chunks already held under that name, so an edited document does not leave
    a stale copy of itself in the index to be retrieved later.
    """
    report = IngestReport()
    store = existing if existing is not None else VectorStore()

    capacity = None  # looked up lazily; it loads the embedding model

    for path in paths:
        path = Path(path)
        name = path.name

        try:
            content_hash = hash_file(path)
        except OSError as exc:
            report.skipped.append((name, f"could not read: {exc}"))
            continue

        if store.is_current(name, content_hash):
            report.skipped.append((name, "already indexed"))
            continue

        replacing = store.has_name(name)

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

        if capacity is None:
            capacity = embedder.max_input_words()
            if config.CHUNK_WORDS > capacity:
                report.skipped.append((
                    "config",
                    f"CHUNK_WORDS is {config.CHUNK_WORDS} but the embedding model "
                    f"only reads the first ~{capacity} words of a chunk - the rest "
                    "is invisible to search. Lower CHUNK_WORDS in utils/config.py.",
                ))

        if on_progress:
            on_progress(f"Embedding {len(chunks)} chunks from {name}")

        # Embed before touching the store. If this raises, the index is
        # exactly as it was - a half-embedded document must never be left
        # marked as indexed.
        try:
            vectors = embedder.embed([c.text for c in chunks])
        except Exception as exc:
            report.skipped.append((name, f"could not embed: {exc}"))
            continue

        if replacing:
            store.remove_document(name)

        store.add(
            vectors,
            chunks,
            DocumentRecord(
                name=name,
                content_hash=content_hash,
                pages=len(pages),
                chunk_count=len(chunks),
            ),
        )

        report.chunks_added += len(chunks)
        (report.reindexed if replacing else report.indexed).append(name)

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

    return store.search(query_vector, k, query=question, min_similarity=min_similarity)
