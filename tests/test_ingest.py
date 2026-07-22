"""Ingest / page-triage / collage tests."""
from __future__ import annotations

import base64

import pytest

from app.pipeline.ingest import (
    ParsedDoc,
    UnsupportedFileError,
    _build_collage,
    parse_document,
)


def test_text_file_parsed():
    doc = parse_document("notes.txt", b"hello world")
    assert isinstance(doc, ParsedDoc)
    assert doc.text == "hello world"
    assert not doc.has_image_pages


def test_unsupported_extension_raises():
    with pytest.raises(UnsupportedFileError):
        parse_document("archive.zip", b"PK\x03\x04")


def test_pdf_page_triage_routes_blank_page_to_collage():
    import fitz

    doc = fitz.open()
    p1 = doc.new_page()
    # Many short lines that fit on the page -> comfortably above the 50-token threshold.
    p1.insert_text((72, 100), "\n".join(["resume skills experience"] * 30), fontsize=10)
    doc.new_page()  # blank -> should be treated as an image page
    data = doc.tobytes()

    parsed = parse_document("cv.pdf", data)
    assert parsed.image_page_indices == [1]
    assert parsed.has_image_pages
    assert "resume" in parsed.text


def test_build_collage_returns_png_base64():
    from PIL import Image
    import io

    def png(color):
        buf = io.BytesIO()
        Image.new("RGB", (300, 300), color).save(buf, format="PNG")
        return buf.getvalue()

    # New signature: (true page index, PNG bytes) per tile.
    b64 = _build_collage([(0, png("red")), (2, png("blue")), (4, png("green"))])
    raw = base64.b64decode(b64)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic

    # 3 tiles -> 2x2 grid of 520px cells.
    from PIL import Image
    import io as _io

    with Image.open(_io.BytesIO(raw)) as im:
        assert (im.width, im.height) == (2 * 520, 2 * 520)
