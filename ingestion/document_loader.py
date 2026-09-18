"""Turn files on disk into pages of plain text.

A "page" is (page_number, text). PDFs have real pages; text files are treated
as a single page 1. Keeping the page number lets us cite it later.

Two of the formats here do not give up their text easily:

A scanned PDF has no text layer at all - it is pictures of words - so it is
rasterised and read back with OCR, page by page. That is slow enough that it
only happens to pages that yielded nothing on their own.

A .docx has no pages. Pagination is decided by whatever renders the file, so
a page number would be invented. Only breaks the author actually inserted are
honoured; a document without them is one page, which is the truth.
"""

from pathlib import Path
from typing import Callable

from pypdf import PdfReader

from utils import config, logs

log = logs.get(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".docx"}

# Below this many characters a PDF page has effectively no text on it, whether
# because it is a scan or because it is a full-page figure.
BLANK_PAGE_CHARS = 40


def load_pages(
    path: Path,
    allow_ocr: bool = True,
    on_progress: Callable[[str], None] | None = None,
) -> list[tuple[int, str]]:
    """Read one file into a list of (page_number, text)."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type {suffix!r}. Supported: "
            f"{', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    if suffix == ".pdf":
        return _load_pdf(path, allow_ocr, on_progress)
    if suffix == ".docx":
        return _load_docx(path)

    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return [(1, text)] if text else []


# --- PDF -------------------------------------------------------------------

def _load_pdf(path, allow_ocr, on_progress) -> list[tuple[int, str]]:
    """Extract what text there is, then OCR only the pages that had none.

    Doing it per page rather than per document matters for the common real
    case: a report that is mostly typed with a scanned appendix, or a letter
    with a photographed signature page.
    """
    reader = PdfReader(str(path))
    extracted: list[tuple[int, str]] = []
    empty: list[int] = []

    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if len(text) >= BLANK_PAGE_CHARS:
            extracted.append((number, text))
        else:
            empty.append(number)

    if empty and allow_ocr and config.OCR_ENABLED and ocr_available():
        extracted.extend(_ocr_pdf_pages(path, empty, on_progress))

    extracted.sort(key=lambda item: item[0])
    return [(number, text) for number, text in extracted if text.strip()]


def ocr_available() -> bool:
    """Whether the optional OCR dependencies are installed.

    They are optional on purpose: they add about 47 MB and are useless to
    anyone whose documents are already text.
    """
    try:
        import pymupdf  # noqa: F401
        import rapidocr_onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


_reader = None


def _get_reader():
    """Built once; constructing it loads the detection and recognition models."""
    global _reader
    if _reader is None:
        from rapidocr_onnxruntime import RapidOCR

        _reader = RapidOCR()
    return _reader


def _ocr_pdf_pages(path, numbers, on_progress) -> list[tuple[int, str]]:
    """Rasterise the given pages and read them back. Slow: seconds per page."""
    import pymupdf

    reader = _get_reader()
    out = []

    with pymupdf.open(str(path)) as document:
        for position, number in enumerate(numbers, start=1):
            if on_progress:
                on_progress(
                    f"Reading page {number} with OCR ({position} of {len(numbers)})"
                )
            try:
                pixmap = document[number - 1].get_pixmap(dpi=config.OCR_DPI)
                result, _ = reader(pixmap.tobytes("png"))
            except Exception:
                # One unreadable page should not lose the rest of the document,
                # but it should not vanish either - a document that comes back
                # short is otherwise inexplicable.
                log.warning("OCR failed on %s page %d", path.name, number,
                            exc_info=True)
                continue

            lines = [line[1] for line in result] if result else []
            if lines:
                out.append((number, "\n".join(lines)))

    return out


# --- DOCX ------------------------------------------------------------------

def _load_docx(path) -> list[tuple[int, str]]:
    """Paragraphs and tables, split only on page breaks the author inserted.

    Tables are included because policy documents put their most citable facts
    in them - rates, thresholds, deadlines - and a reader that skipped tables
    would miss exactly the things people ask about.
    """
    import docx

    document = docx.Document(str(path))
    pages: list[list[str]] = [[]]

    for block in _docx_blocks(document):
        text, breaks_page = block
        if breaks_page and pages[-1]:
            pages.append([])
        if text:
            pages[-1].append(text)

    return [
        (number, "\n".join(lines).strip())
        for number, lines in enumerate(pages, start=1)
        if "".join(lines).strip()
    ]


def _docx_blocks(document):
    """Yield (text, starts_a_new_page) for each paragraph and table, in order.

    python-docx exposes paragraphs and tables as separate collections, which
    loses the order they appear in. The body's own XML children keep it, so a
    table between two paragraphs stays between them.
    """
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    body = document.element.body
    for child in body.iterchildren():
        tag = child.tag.split("}")[-1]

        if tag == "p":
            paragraph = Paragraph(child, document)
            # w:br with type="page" is an explicit page break.
            breaks_page = any(
                br.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}type")
                == "page"
                for br in child.iter(
                    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}br"
                )
            )
            yield paragraph.text.strip(), breaks_page

        elif tag == "tbl":
            table = Table(child, document)
            rows = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    rows.append(" | ".join(cells))
            yield "\n".join(rows), False


# --- Diagnosis -------------------------------------------------------------

def looks_scanned(path: Path, pages: list[tuple[int, str]]) -> bool:
    """True if a PDF yielded almost no text even after OCR was given a turn.

    Reaching here now means either the OCR dependencies are missing or the
    scan was too poor to read, so the message shown to the user depends on
    which - see ocr_available().
    """
    if Path(path).suffix.lower() != ".pdf":
        return False
    if not pages:
        return True
    total_chars = sum(len(text) for _, text in pages)
    return total_chars / max(len(pages), 1) < 100
