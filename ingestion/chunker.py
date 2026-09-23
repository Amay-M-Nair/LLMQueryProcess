"""Split page text into overlapping chunks small enough to embed.

LangChain's RecursiveCharacterTextSplitter does the splitting. It is given a
word-counting length function rather than the default character count,
because the limit that matters here is what the embedding model can read -
256 tokens, about 176 words - and characters are a poor proxy for that.

What it buys over a fixed window is where the cuts land. A window closes at
exactly 160 words whatever is there, so chunks ended "...before 10:00" and
"...A". The splitter prefers a paragraph break, then a sentence, then a word,
and only cuts mid-word if it must.

Splitting happens per page, so a chunk never spans a page boundary. That
costs a few short chunks at the end of each page and keeps the page citation
honest, which is the whole point of tracking pages at all.
"""

from dataclasses import asdict, dataclass

from utils import config


@dataclass
class Chunk:
    """One retrievable passage, and everything needed to cite it.

    `chunk_id` follows the shape `employee_handbook_page4_chunk12` - unique
    across the whole index, and readable enough to grep for in a debug panel.
    The word range locates the chunk inside its page, which is what makes
    overlapping neighbours distinguishable when two chunks look alike.
    """

    text: str
    source: str        # file name the chunk came from
    page: int          # page number within that file
    chunk_id: str = ""
    start_word: int = 0
    end_word: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def label(self) -> str:
        return f"{self.source} p.{self.page}"


def _slug(source: str) -> str:
    """A file name reduced to something safe to embed in an identifier."""
    stem = source.rsplit(".", 1)[0]
    return "".join(c if c.isalnum() else "_" for c in stem).strip("_").lower()


def _splitter(chunk_words: int, overlap_words: int):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_words,
        chunk_overlap=overlap_words,
        # The binding constraint is the embedding model's token limit, so the
        # unit is words. With the default character count, a chunk of long
        # technical words would quietly exceed what the model reads.
        length_function=lambda text: len(text.split()),
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        keep_separator=True,
    )


def chunk_pages(
    pages: list[tuple[int, str]],
    source: str,
    chunk_words: int | None = None,
    overlap_words: int | None = None,
) -> list[Chunk]:
    """Split each page into overlapping chunks.

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

    splitter = _splitter(chunk_words, overlap_words)
    slug = _slug(source)
    chunks: list[Chunk] = []

    for page_number, text in pages:
        if not text.split():
            continue

        cursor = 0
        for piece in splitter.split_text(text):
            piece = piece.strip()
            if not piece:
                continue

            # Locate the piece in the page so the word range means something.
            # Searching from the cursor keeps overlapping pieces in order
            # rather than all matching the first occurrence.
            position = text.find(piece, cursor)
            if position == -1:
                position = text.find(piece)
            if position != -1:
                cursor = position + 1
                start_word = len(text[:position].split())
            else:
                start_word = 0

            chunks.append(
                Chunk(
                    text=piece,
                    source=source,
                    page=page_number,
                    chunk_id=f"{slug}_page{page_number}_chunk{len(chunks)}",
                    start_word=start_word,
                    end_word=start_word + len(piece.split()),
                )
            )

    return chunks
