"""Everything about a chunk that is not its vector.

FAISS only knows vectors and integer ids. This is the other half of doc §6.2:
the map from vector id to the chunk text and its source, plus a register of
which documents are in the index and what they hashed to when they went in.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ingestion.chunker import Chunk
from vectorstore.keyword import BM25

METADATA_FILE = "metadata.json"
FORMAT_VERSION = 2  # bumped when chunk_id and document hashing were added


def hash_file(path: Path) -> str:
    """A content fingerprint, so an edited file is not mistaken for the old one."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class DocumentRecord:
    """One indexed file. `content_hash` is what makes re-indexing correct."""

    name: str
    content_hash: str
    pages: int
    chunk_count: int
    indexed_at: str = ""

    def __post_init__(self):
        if not self.indexed_at:
            self.indexed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class MetadataStore:
    chunks: list[Chunk] = field(default_factory=list)
    documents: dict[str, DocumentRecord] = field(default_factory=dict)
    _bm25: BM25 | None = field(default=None, init=False, repr=False)

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def sources(self) -> list[str]:
        """Distinct file names in the index, in insertion order."""
        return list(self.documents)

    @property
    def total_words(self) -> int:
        return sum(len(chunk.text.split()) for chunk in self.chunks)

    @property
    def bm25(self) -> BM25:
        """Built on first use and cached - cheap, but not free."""
        if self._bm25 is None:
            self._bm25 = BM25([chunk.text for chunk in self.chunks])
        return self._bm25

    def invalidate(self) -> None:
        """Drop the cached BM25 index; the chunk list underneath it changed."""
        self._bm25 = None

    # --- Documents ---------------------------------------------------------

    def is_current(self, name: str, content_hash: str) -> bool:
        """True when this exact file content is already indexed under this name."""
        record = self.documents.get(name)
        return record is not None and record.content_hash == content_hash

    def has_name(self, name: str) -> bool:
        return name in self.documents

    def positions_for(self, name: str) -> list[int]:
        """Which rows belong to a document - the ids FAISS must forget."""
        return [i for i, chunk in enumerate(self.chunks) if chunk.source == name]

    def add(self, chunks: list[Chunk], document: DocumentRecord) -> None:
        self.chunks.extend(chunks)
        self.documents[document.name] = document
        self.invalidate()

    def keep(self, positions: list[int]) -> None:
        """Retain only these rows, in this order. Mirrors a FAISS rebuild."""
        self.chunks = [self.chunks[i] for i in positions]
        remaining = {chunk.source for chunk in self.chunks}
        self.documents = {
            name: record
            for name, record in self.documents.items()
            if name in remaining
        }
        self.invalidate()

    # --- Persistence -------------------------------------------------------

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": FORMAT_VERSION,
            "documents": [asdict(record) for record in self.documents.values()],
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }
        (directory / METADATA_FILE).write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> "MetadataStore | None":
        path = Path(directory) / METADATA_FILE
        if not path.exists():
            return None

        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != FORMAT_VERSION:
            # An index written by an older layout. Refusing to guess is safer
            # than silently searching half-populated metadata.
            raise ValueError(
                f"The index in {directory} was written in format "
                f"v{payload.get('version')}, but this version of the app reads "
                f"v{FORMAT_VERSION}. Clear the index and re-add your documents."
            )

        return cls(
            chunks=[Chunk(**item) for item in payload["chunks"]],
            documents={
                item["name"]: DocumentRecord(**item) for item in payload["documents"]
            },
        )
