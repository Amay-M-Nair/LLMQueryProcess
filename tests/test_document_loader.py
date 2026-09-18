"""Reading the formats that do not give up their text easily.

The OCR tests are skipped when the optional dependencies are absent, because
they are genuinely optional - a project whose documents are already text
should not be forced to install 47 MB to run its own test suite.
"""

import pytest

from ingestion import document_loader as loader

docx = pytest.importorskip("docx")
pymupdf = pytest.importorskip("pymupdf")

needs_ocr = pytest.mark.skipif(
    not loader.ocr_available(), reason="OCR dependencies are not installed"
)

SECRET = "BLUEJAY-7741"


# --- .docx -----------------------------------------------------------------

@pytest.fixture
def written(tmp_path):
    """A document with a table, a page break, and text after the break."""
    document = docx.Document()
    document.add_paragraph("Northwind Logistics - Travel Policy")
    document.add_paragraph("Economy class is standard for flights under six hours.")

    table = document.add_table(rows=3, cols=2)
    table.cell(0, 0).text = "Destination"
    table.cell(0, 1).text = "Nightly cap"
    table.cell(1, 0).text = "London"
    table.cell(1, 1).text = "180 pounds"
    table.cell(2, 0).text = "Elsewhere in the UK"
    table.cell(2, 1).text = "140 pounds"

    document.add_page_break()
    document.add_paragraph("Appendix A. The reference code is " + SECRET + ".")

    path = tmp_path / "policy.docx"
    document.save(str(path))
    return path


def test_reads_paragraphs(written):
    pages = loader.load_pages(written)
    assert "Economy class is standard" in pages[0][1]


def test_reads_tables_too(written):
    """Policy documents put their most citable facts in tables."""
    text = " ".join(t for _, t in loader.load_pages(written))
    assert "180 pounds" in text, "the table was skipped"
    assert "London" in text


def test_keeps_tables_in_document_order(written):
    """A table between two paragraphs must not be appended at the end."""
    first_page = loader.load_pages(written)[0][1]
    assert first_page.index("Economy class") < first_page.index("Nightly cap")


def test_an_author_page_break_starts_a_new_page(written):
    pages = loader.load_pages(written)
    assert len(pages) == 2
    assert SECRET in pages[1][1]
    assert SECRET not in pages[0][1]


def test_a_document_without_breaks_is_one_page(tmp_path):
    """Pagination is the renderer's business, so it is not invented here."""
    document = docx.Document()
    for _ in range(40):
        document.add_paragraph("A paragraph that would wrap across pages when printed.")
    path = tmp_path / "long.docx"
    document.save(str(path))

    pages = loader.load_pages(path)
    assert len(pages) == 1, "page numbers were invented for a document that has none"


def test_empty_docx_yields_nothing(tmp_path):
    path = tmp_path / "empty.docx"
    docx.Document().save(str(path))
    assert loader.load_pages(path) == []


def test_docx_is_offered_as_a_supported_type():
    assert ".docx" in loader.SUPPORTED_SUFFIXES


# --- Scanned PDFs ----------------------------------------------------------

def make_scanned_pdf(path, lines, extra_text_page=None):
    """A PDF whose pages are pictures of words, which is what a scan is."""
    typed = pymupdf.open()
    page = typed.new_page()
    y = 90
    for line in lines:
        page.insert_text((60, y), line, fontsize=15, fontname="helv")
        y += 32

    scanned = pymupdf.open()
    for source in typed:
        pixmap = source.get_pixmap(dpi=200)
        target = scanned.new_page(width=source.rect.width, height=source.rect.height)
        target.insert_image(target.rect, pixmap=pixmap)

    if extra_text_page:
        page = scanned.new_page()
        page.insert_text((60, 90), extra_text_page, fontsize=15, fontname="helv")

    scanned.save(str(path))
    return path


def test_a_scan_really_has_no_text_layer(tmp_path):
    """Guards the fixture itself: if this fails the OCR tests prove nothing."""
    path = make_scanned_pdf(tmp_path / "scan.pdf", ["The code is " + SECRET])
    pages = loader.load_pages(path, allow_ocr=False)
    assert sum(len(t) for _, t in pages) == 0


@needs_ocr
def test_ocr_recovers_the_text(tmp_path):
    path = make_scanned_pdf(
        tmp_path / "scan.pdf",
        ["Northwind Logistics", "The vault code is " + SECRET + "."],
    )
    text = " ".join(t for _, t in loader.load_pages(path))
    assert SECRET in text.replace(" ", ""), f"OCR did not recover the code: {text!r}"


@needs_ocr
def test_only_the_pages_without_text_are_ocred(tmp_path):
    """A part-scanned document is the common real case, and the expensive one."""
    path = make_scanned_pdf(
        tmp_path / "mixed.pdf",
        ["Scanned page one with the code " + SECRET],
        extra_text_page="This page is ordinary typed text, long enough to count as text.",
    )
    seen = []
    pages = loader.load_pages(path, on_progress=seen.append)

    assert len(pages) == 2
    ocr_messages = [m for m in seen if "OCR" in m]
    assert len(ocr_messages) == 1, f"OCR ran on the wrong number of pages: {seen}"
    assert "page 1" in ocr_messages[0]


@needs_ocr
def test_pages_come_back_in_order(tmp_path):
    """OCR'd pages are collected separately and must be merged back in order."""
    path = make_scanned_pdf(
        tmp_path / "mixed.pdf",
        ["Scanned first page"],
        extra_text_page="Typed second page, with enough characters to count as text.",
    )
    numbers = [n for n, _ in loader.load_pages(path)]
    assert numbers == sorted(numbers)


def test_ocr_can_be_declined(tmp_path):
    path = make_scanned_pdf(tmp_path / "scan.pdf", ["The code is " + SECRET])
    assert loader.load_pages(path, allow_ocr=False) == []


def test_a_scan_is_still_reported_as_scanned_when_ocr_is_off(tmp_path):
    path = make_scanned_pdf(tmp_path / "scan.pdf", ["The code is " + SECRET])
    pages = loader.load_pages(path, allow_ocr=False)
    assert loader.looks_scanned(path, pages)


def test_unsupported_types_still_refuse(tmp_path):
    path = tmp_path / "sheet.xlsx"
    path.write_bytes(b"not really a spreadsheet")
    with pytest.raises(ValueError, match="Unsupported file type"):
        loader.load_pages(path)
