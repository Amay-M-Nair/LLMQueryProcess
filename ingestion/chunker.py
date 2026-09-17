"""Split page text into overlapping chunks small enough to embed.

Chunks never span a page boundary, which keeps the page citation honest at
the cost of a few short chunks at the end of each page.
"""

from dataclasses import asdict, dataclass

from utils import config


@dataclass
class Chunk:
    text: str
    source: str   # file name the chunk came from
    page: int     # page number within that file

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def label(self) -> str:
        return f"{self.source} p.{self.page}"


def chunk_pages(
    pages: list[tuple[int, str]],
    source: str,
    chunk_words: int | None = None,
    overlap_words: int | None = None,
) -> list[Chunk]:
    """Slide a fixed-size window over each page's words.

    The sizes default to None and are looked up in config when the function
    runs - binding config values as default arguments would freeze them at
    import time, so later changes would silently have no effect.
    """
    chunk_words = config.CHUNK_WORDS if chunk_words is None else chunk_words
    overlap_words = (
        config.CHUNK_OVERLAP_WORDS if overlap_words is None else overlap_words
    )
    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")

    step = chunk_words - overlap_words
    chunks: list[Chunk] = []

    for page_number, text in pages:
        words = text.split()
        if not words:
            continue
        for start in range(0, len(words), step):
            window = words[start : start + chunk_words]
            # Skip a tiny trailing remainder already covered by the overlap.
            if start > 0 and len(window) < overlap_words:
                break
            chunks.append(
                Chunk(text=" ".join(window), source=source, page=page_number)
            )

    return chunks
