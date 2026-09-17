"""Turn files on disk into pages of plain text.

A "page" is (page_number, text). PDFs have real pages; text files are
treated as a single page 1. Keeping the page number lets us cite it later.
"""

from pathlib import Path

from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


def load_pages(path: Path) -> list[tuple[int, str]]:
    """Read one file into a list of (page_number, text)."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type {suffix!r}. Supported: "
            f"{', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = []
        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append((number, text))
        return pages

    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return [(1, text)] if text else []


def looks_scanned(path: Path, pages: list[tuple[int, str]]) -> bool:
    """True if a PDF yielded almost no text, which usually means it is images.

    Such a file needs OCR before this pipeline can do anything with it.
    """
    if Path(path).suffix.lower() != ".pdf":
        return False
    if not pages:
        return True
    total_chars = sum(len(text) for _, text in pages)
    return total_chars / max(len(pages), 1) < 100
