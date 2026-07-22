"""End-to-end orchestrator tests with the fake LLM client.

Covers the happy path, per-document failure isolation, hallucination nulling via
the verify grounding pass, and low-confidence document flagging.
"""
from __future__ import annotations

import pytest

from app.models.api import DocStatus
from app.models.schemas import DocType
from app.pipeline.orchestrator import process_batch
from app.pipeline.verify import FieldJudgement

pytestmark = pytest.mark.asyncio


async def test_happy_path_resume(fake_client):
    fake_client(
        letter_rules={"curriculum": "A"},
        extract_data={
            "full_name": "Ada Lovelace",
            "skills": ["mathematics"],
            "experience": [{"company": "Analytical Engine Co", "title": "Engineer"}],
        },
    )
    report = await process_batch([("cv.txt", b"Curriculum Vitae of Ada Lovelace")])
    doc = report.documents[0]
    assert doc.type is DocType.RESUME
    assert doc.status is DocStatus.DONE
    assert doc.data["full_name"] == "Ada Lovelace"
    assert doc.type_confidence > 0.6
    assert report.summary.succeeded == 1


async def test_failure_isolation(fake_client):
    fake_client(letter_rules={"curriculum": "A"}, extract_data={"full_name": "Ada"})
    report = await process_batch(
        [
            ("cv.txt", b"Curriculum Vitae"),
            ("weird.zip", b"PK\x03\x04"),  # unsupported -> fails, must not break batch
        ]
    )
    statuses = {d.filename: d.status for d in report.documents}
    assert statuses["cv.txt"] is DocStatus.DONE
    assert statuses["weird.zip"] is DocStatus.FAILED
    assert report.summary.succeeded == 1
    assert report.summary.failed == 1
    assert report.summary.failures[0].stage == "ingest"


async def test_hallucinated_field_nulled_and_flagged(fake_client):
    fake_client(
        letter_rules={"curriculum": "A"},
        extract_data={"full_name": "Ada Lovelace", "email": "made-up@nowhere.com"},
        judgements=[FieldJudgement(path="email", supported=False, confidence=0.1)],
    )
    report = await process_batch([("cv.txt", b"Curriculum Vitae of Ada")])
    doc = report.documents[0]
    assert doc.data["email"] is None  # unsupported value removed
    assert any(f.field == "email" and "unsupported" in f.reason for f in doc.flagged_fields)


def _png_bytes() -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (200, 200), "white").save(buf, format="PNG")
    return buf.getvalue()


async def test_vision_failure_degrades_not_fails(fake_client):
    # A standalone image forces the vision path; vision errors must degrade the doc.
    fake_client(vision_error=True, default_letter="F", extract_data={})
    report = await process_batch([("scan.png", _png_bytes())])
    doc = report.documents[0]
    assert doc.status is DocStatus.DONE  # not FAILED
    assert any("vision transcription failed" in f.reason for f in doc.flagged_fields)


async def test_verify_failure_preserves_extraction(fake_client):
    fake_client(
        letter_rules={"curriculum": "A"},
        extract_data={"full_name": "Ada Lovelace", "skills": ["math"]},
        verify_error=True,
    )
    report = await process_batch([("cv.txt", b"Curriculum Vitae of Ada")])
    doc = report.documents[0]
    assert doc.status is DocStatus.DONE
    assert doc.data["full_name"] == "Ada Lovelace"  # extraction NOT discarded
    assert any("verification skipped" in f.reason for f in doc.flagged_fields)


async def test_other_type_always_flagged_for_review(fake_client):
    fake_client(default_letter="F", extract_data={"document_summary": "misc"})
    report = await process_batch([("misc.txt", b"random content")])
    doc = report.documents[0]
    assert doc.type is DocType.OTHER
    assert doc.signals.fill_rate is None  # not the misleading 1.0
    assert any("generic fallback" in f.reason for f in doc.flagged_fields)


async def test_zero_fill_first_class_type_flagged(fake_client):
    # A bare photo misread as an ID: classified id_document but nothing extracted.
    fake_client(default_letter="D", extract_data={})  # D = id_document
    report = await process_batch([("photo.txt", b"an ambiguous image")])
    doc = report.documents[0]
    assert doc.type is DocType.ID_DOCUMENT
    assert doc.signals.fill_rate == 0.0
    assert any("no fields extracted" in f.reason for f in doc.flagged_fields)


async def test_missing_logprobs_do_not_tank_confidence(fake_client):
    fake_client(
        empty_distribution=True,
        letter_rules={"curriculum": "A"},
        extract_data={"full_name": "Ada", "skills": ["x"], "experience": [{"company": "Y"}]},
    )
    report = await process_batch([("cv.txt", b"Curriculum Vitae")])
    doc = report.documents[0]
    assert doc.status is DocStatus.DONE
    # Would have collapsed to ~0.4 (and been flagged) before the fix.
    assert doc.type_confidence > 0.6
    assert doc.signals.logprob_top1 is None


async def test_low_confidence_document_flagged(fake_client):
    # Ambiguous classification: top1 barely above top2 -> tiny margin -> low composite.
    fake_client(
        letter_rules={"stuff": "B"},
        top1=0.34,
        top2=0.33,
        extract_data={},
    )
    report = await process_batch([("mystery.txt", b"some stuff here")])
    doc = report.documents[0]
    assert doc.type_confidence < 0.6
    assert any(f.level == "document" for f in doc.flagged_fields)
    assert report.summary.flag_rate > 0
