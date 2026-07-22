"""Ingest stage: turn a raw file into a normalized representation.

Emits ``ParsedDoc = {text, page_texts, image_page_indices, collage_b64}``. The
core idea (page triage): PDFs are parsed per page and any page whose extractable
text is below ``IMAGE_PAGE_MIN_TOKENS`` is assumed to be a scan/image. Those pages
are rasterized and stitched into a single collage that a vision model reads once,
cheaply, to inform classification. Native image uploads are treated as one image
page. The (optional) vision call itself is ``describe_images`` so the orchestrator
can attribute its cost to the ingest stage event.
"""
from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass, field

from app.config import get_settings
from app.llm.client import VisionOutcome, get_client
from app.llm.prompts import VISION_SYSTEM, VISION_USER

try:  # tiktoken is the accurate path; fall back to a word heuristic if unavailable
    import tiktoken

    _ENC = tiktoken.get_encoding("o200k_base")

    def _count_tokens(text: str) -> int:
        return len(_ENC.encode(text or ""))
except Exception:  # noqa: BLE001

    def _count_tokens(text: str) -> int:
        return len((text or "").split())


class UnsupportedFileError(ValueError):
    pass


@dataclass
class ParsedDoc:
    text: str
    page_texts: list[str] = field(default_factory=list)
    image_page_indices: list[int] = field(default_factory=list)
    collage_b64: str | None = None

    @property
    def has_image_pages(self) -> bool:
        return self.collage_b64 is not None


def parse_document(filename: str, data: bytes) -> ParsedDoc:
    """Pure (no network) parse + page triage + collage assembly."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "pdf":
        return _parse_pdf(data)
    if ext in {"png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff"}:
        return _parse_image(data)
    if ext == "docx":
        return _parse_docx(data)
    if ext in {"txt", "md", "csv", ""}:
        return _parse_text(data)
    raise UnsupportedFileError(f"unsupported file type: .{ext}")


def _parse_text(data: bytes) -> ParsedDoc:
    text = data.decode("utf-8", errors="replace").strip()
    return ParsedDoc(text=text, page_texts=[text])


def _parse_docx(data: bytes) -> ParsedDoc:
    import docx  # python-docx

    document = docx.Document(io.BytesIO(data))
    text = "\n".join(p.text for p in document.paragraphs).strip()
    return ParsedDoc(text=text, page_texts=[text])


def _parse_image(data: bytes) -> ParsedDoc:
    # A standalone image is a single image page (Page 1), labelled like any other.
    collage = _build_collage([(0, data)])
    return ParsedDoc(text="", page_texts=[""], image_page_indices=[0], collage_b64=collage)


def _parse_pdf(data: bytes) -> ParsedDoc:
    import fitz  # PyMuPDF

    threshold = get_settings().image_page_min_tokens
    page_texts: list[str] = []
    image_pages: list[int] = []
    rendered: list[tuple[int, bytes]] = []  # (0-based page index, PNG bytes)

    with fitz.open(stream=data, filetype="pdf") as doc:
        for i, page in enumerate(doc):
            txt = (page.get_text() or "").strip()
            page_texts.append(txt)
            if _count_tokens(txt) < threshold:
                image_pages.append(i)
                # Rasterize at ~1.5x for legible-but-small collage tiles.
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                rendered.append((i, pix.tobytes("png")))

    collage_b64 = _build_collage(rendered) if rendered else None
    full_text = _assemble_text(page_texts, set(image_pages))
    return ParsedDoc(
        text=full_text,
        page_texts=page_texts,
        image_page_indices=image_pages,
        collage_b64=collage_b64,
    )


def _assemble_text(page_texts: list[str], image_pages: set[int]) -> str:
    """Join page texts with 1-based page markers so text and scanned (image) pages
    stay interleaved in order — a scanned page points to its collage tile ('Page N'),
    which the vision transcription uses the same label for."""
    parts: list[str] = []
    for i, txt in enumerate(page_texts):
        label = f"Page {i + 1}"
        if i in image_pages:
            parts.append(f"[{label} — scanned image; see vision transcription for '{label}']")
        elif txt:
            parts.append(f"[{label}]\n{txt}")
    return "\n\n".join(parts).strip()


def _build_collage(pages: list[tuple[int, bytes]], max_pages: int = 9, cell: int = 520) -> str:
    """Stitch rendered pages into one labelled grid PNG (base64).

    Each tile is a fixed square cell with a border and a 'Page N' header using the
    page's TRUE 1-based index in the document — so the vision model can name which
    page a field came from, which matters when text and image pages interleave.
    """
    from PIL import Image, ImageDraw, ImageFont

    pages = pages[:max_pages]
    n = len(pages)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    label_h, pad = 32, 8

    canvas = Image.new("RGB", (cols * cell, rows * cell), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:  # older Pillow without size arg
        font = ImageFont.load_default()

    for idx, (page_no, raw) in enumerate(pages):
        cx, cy = (idx % cols) * cell, (idx // cols) * cell
        # Cell border + label bar with the true page number.
        draw.rectangle([cx, cy, cx + cell - 1, cy + cell - 1], outline="black", width=2)
        draw.rectangle([cx, cy, cx + cell - 1, cy + label_h], fill="black")
        draw.text((cx + 8, cy + 5), f"Page {page_no + 1}", fill="white", font=font)
        # Page image, fit under the label and centred in the cell.
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img.thumbnail((cell - 2 * pad, cell - label_h - 2 * pad))
        ox = cx + (cell - img.width) // 2
        oy = cy + label_h + (cell - label_h - img.height) // 2
        canvas.paste(img, (ox, oy))

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


async def describe_images(collage_b64: str) -> VisionOutcome:
    """Single vision call over the collage → notes used by classify."""
    return await get_client().vision_describe(VISION_SYSTEM, VISION_USER, collage_b64)
