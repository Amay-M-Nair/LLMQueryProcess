"""One index per collection.

This is what makes the API safe to expose. A single shared index is correct
for one person on one machine and disqualifying the moment the service has a
URL: every caller's uploads land in the same place, so one caller's question
retrieves and cites another's documents.

A collection is just a name the caller chooses and keeps sending. There is no
login, because an API does not need one to keep callers apart - it needs a
namespace, and the caller is the only one who knows which namespace is
theirs. Whoever deploys this decides whether those names are secret, guessed,
or handed out by something in front.

The name is checked hard, because it becomes a directory. A caller who sends
"../../etc" must get a rejection and not a path.
"""

import re
import shutil
from pathlib import Path

from ingestion.pipeline import build_index
from utils import config, logs
from vectorstore.faiss_store import VectorStore

log = logs.get(__name__)

ROOT = config.DATA_DIR / "collections"

# Letters, digits, dash, underscore. Nothing that can traverse or collide on a
# case-insensitive filesystem once lowercased.
VALID_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")


class BadCollectionName(ValueError):
    """The name cannot be used as a directory. Shown to the caller."""


def normalise(name: str) -> str:
    """Validate a caller-supplied name and return its canonical form."""
    if not name or not VALID_NAME.match(name):
        raise BadCollectionName(
            "A collection name must be 1-64 characters of letters, digits, "
            "dashes or underscores, and start with a letter or digit."
        )
    return name.lower()


def index_dir(name: str) -> Path:
    directory = ROOT / normalise(name)
    # Belt and braces: the regex already forbids traversal, but a path that
    # escaped the root would be a directory disclosure, so it is checked.
    if ROOT.resolve() not in directory.resolve().parents:
        raise BadCollectionName("That collection name is not usable.")
    return directory


def upload_dir(name: str) -> Path:
    return index_dir(name) / "uploads"


def load(name: str) -> VectorStore | None:
    """This collection's index, or None if it has none yet."""
    return VectorStore.load(index_dir(name))


def save_uploads(name: str, files: list[tuple[str, bytes]]) -> list[Path]:
    """Write (filename, bytes) pairs into this collection's own directory."""
    directory = upload_dir(name)
    directory.mkdir(parents=True, exist_ok=True)

    saved = []
    for filename, data in files:
        # Take the basename again rather than trust a name from a request.
        safe = Path(filename).name
        if not safe or safe.startswith("."):
            raise BadCollectionName(f"{filename!r} is not a usable file name.")
        destination = directory / safe
        destination.write_bytes(data)
        saved.append(destination)
    return saved


def ingest(name: str, paths: list[Path], on_progress=None):
    existing = load(name)
    store, report = build_index(paths, existing=existing, on_progress=on_progress)
    store.save(index_dir(name))
    return store, report


def clear(name: str) -> bool:
    """Remove the index and the documents it was built from. True if it existed.

    The uploads go too. Someone clearing their documents means the documents,
    not the search over them.
    """
    directory = index_dir(name)
    existed = directory.exists()
    shutil.rmtree(directory, ignore_errors=True)
    if existed:
        log.info("cleared collection %s", normalise(name))
    return existed


def describe(name: str) -> dict:
    """What is in this collection, for a status endpoint."""
    store = load(name)
    if store is None:
        return {"collection": normalise(name), "documents": [], "chunks": 0}
    return {
        "collection": normalise(name),
        "documents": [
            {"name": record.name, "pages": record.pages, "chunks": record.chunk_count}
            for record in store.documents.values()
        ],
        "chunks": len(store),
    }
