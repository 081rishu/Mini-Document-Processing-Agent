"""Batch orchestrator: the per-document state machine + batch concurrency.

Each document walks INGEST → CLASSIFY → ROUTE → EXTRACT → VERIFY → DONE. Every
stage is wrapped by the observability collector (timing, cost, signals) and the
whole walk is inside a try/except so one document's failure is isolated — it is
marked FAILED with the offending stage and the batch still completes.

Classification confidence is a composite of three independent signals (logprob
margin, cross-modal agreement, schema fill rate); documents below the configured
threshold are flagged for human review with the specific signals that were weak.
"""
from __future__ import annotations

import asyncio
import uuid

from app.config import get_settings
from app.models.api import (
    ConfidenceSignals,
    DocStatus,
    DocumentResult,
    FlaggedItem,
)
from app.models.schemas import DocType, spec_for
from app.obs.events import BatchCollector
from app.pipeline import classify as classify_stage
from app.pipeline import extract as extract_stage
from app.pipeline import ingest as ingest_stage
from app.pipeline import verify as verify_stage
from app.pipeline.summarize import summarize
from app.models.api import BatchReport


def _save_collage(debug_dir: str, batch_id: str, filename: str, collage_b64: str) -> None:
    """Write a scanned doc's vision collage to disk for inspection (debug only)."""
    import base64
    import os
    import re

    os.makedirs(debug_dir, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
    path = os.path.join(debug_dir, f"{batch_id[:8]}_{safe}.png")
    with open(path, "wb") as fh:
        fh.write(base64.b64decode(collage_b64))


_NEUTRAL_PRIOR = 0.5  # used for the logprob component when logprobs are unavailable


def composite_confidence(
    top1: float | None,
    margin: float | None,
    fill_rate: float | None,
    modal_agree: bool | None,
) -> tuple[float, list[str]]:
    """Blend the available signals into one 0-1 score and name the weak ones.

    - logprob signal (top-1 probability + top1/top2 margin) is the calibrated core,
    - schema fill rate catches wrong-schema routing,
    - cross-modal disagreement (text vs. vision) applies a strong penalty.

    When logprobs are unavailable (top1/margin are None) the core component falls back
    to a neutral prior rather than 0 — a missing measurement must not be read as "no
    confidence", which would silently flag every document.
    """
    logprobs_available = top1 is not None and margin is not None
    logprob_signal = (0.7 * top1 + 0.3 * margin) if logprobs_available else _NEUTRAL_PRIOR

    parts = [(logprob_signal, 0.6)]
    if fill_rate is not None:
        parts.append((fill_rate, 0.4))
    score = sum(v * w for v, w in parts) / sum(w for _, w in parts)

    if modal_agree is False:
        score *= 0.6
    elif modal_agree is True:
        score = min(1.0, score * 1.05)

    weak: list[str] = []
    if not logprobs_available:
        weak.append("logprob_unavailable")
    elif margin < 0.15:
        weak.append("logprob_margin")
    if modal_agree is False:
        weak.append("modal_agree")
    if fill_rate is not None and fill_rate < 0.5:
        weak.append("fill_rate")
    return round(score, 3), weak


async def _process_one(
    filename: str, data: bytes, collector: BatchCollector, sem: asyncio.Semaphore
) -> DocumentResult:
    settings = get_settings()
    result = DocumentResult(filename=filename)
    doc_id = uuid.uuid4().hex[:8]
    signals = ConfidenceSignals()
    current_stage = "ingest"

    async with sem:
        try:
            # --- INGEST -------------------------------------------------------
            current_stage = "ingest"
            with collector.stage(doc_id, filename, "ingest") as ev:
                parsed = ingest_stage.parse_document(filename, data)
                vlm_notes: str | None = None
                if parsed.has_image_pages:
                    if settings.collage_debug_dir:
                        _save_collage(settings.collage_debug_dir, collector.batch_id, filename, parsed.collage_b64)
                    ev.model = settings.vision_model
                    # Vision is an enhancement — a failure degrades the doc (proceed
                    # without a transcription), it does not fail the whole document.
                    try:
                        vision = await ingest_stage.describe_images(parsed.collage_b64)
                        vlm_notes = vision.text
                        ev.tokens_in += vision.usage.tokens_in
                        ev.tokens_out += vision.usage.tokens_out
                        ev.cost_usd += vision.usage.cost_usd
                    except Exception as exc:  # noqa: BLE001 - degrade, don't fail
                        ev.signals["vision_failed"] = f"{type(exc).__name__}: {exc}"
                        result.flagged_fields.append(FlaggedItem(
                            doc=filename, level="document",
                            reason="vision transcription failed — scanned pages not read",
                        ))
                result.status = DocStatus.INGESTED

            # --- CLASSIFY -----------------------------------------------------
            current_stage = "classify"
            with collector.stage(doc_id, filename, "classify", settings.classify_model) as ev:
                cls = await classify_stage.classify(parsed.text, vlm_notes)
                ev.tokens_in, ev.tokens_out = cls.usage.tokens_in, cls.usage.tokens_out
                ev.cost_usd = cls.usage.cost_usd
                signals.logprob_top1 = cls.logprob_top1
                signals.logprob_margin = cls.logprob_margin
                # Cross-modal agreement: compare text-only read vs. vision-informed read.
                if parsed.has_image_pages and vlm_notes:
                    text_only = await classify_stage.classify(parsed.text, None)
                    signals.modal_agree = text_only.doc_type == cls.doc_type
                    ev.tokens_in += text_only.usage.tokens_in
                    ev.tokens_out += text_only.usage.tokens_out
                    ev.cost_usd += text_only.usage.cost_usd
                result.type = cls.doc_type
                result.status = DocStatus.CLASSIFIED
                ev.signals = {
                    "logprob_margin": cls.logprob_margin,
                    "logprob_top1": cls.logprob_top1,
                    "modal_agree": signals.modal_agree,
                }

            # --- ROUTE + EXTRACT ---------------------------------------------
            current_stage = "extract"
            spec = spec_for(cls.doc_type)
            with collector.stage(doc_id, filename, "extract", settings.extract_model) as ev:
                ext = await extract_stage.extract(parsed.text, vlm_notes, spec)
                ev.tokens_in, ev.tokens_out = ext.usage.tokens_in, ext.usage.tokens_out
                ev.cost_usd = ext.usage.cost_usd
                signals.fill_rate = ext.fill_rate
                ev.signals = {"fill_rate": ext.fill_rate}
                result.data = ext.data
                result.status = DocStatus.EXTRACTED

            # --- VERIFY -------------------------------------------------------
            current_stage = "verify"
            with collector.stage(doc_id, filename, "verify", settings.verify_model) as ev:
                # Verify is an enhancement — a failure must not discard a good
                # extraction. Degrade: keep the extracted data, mark it unverified.
                try:
                    ver = await verify_stage.verify(
                        cls.doc_type, ext.data, parsed.text, filename, vlm_notes
                    )
                    ev.tokens_in, ev.tokens_out = ver.usage.tokens_in, ver.usage.tokens_out
                    ev.cost_usd = ver.usage.cost_usd
                    result.data = ver.data
                    result.flagged_fields.extend(ver.flagged_fields)
                except Exception as exc:  # noqa: BLE001 - degrade, don't fail
                    ev.signals["verify_failed"] = f"{type(exc).__name__}: {exc}"
                    result.flagged_fields.append(FlaggedItem(
                        doc=filename, level="document",
                        reason="verification skipped (error) — fields not grounded",
                    ))
                result.status = DocStatus.VERIFIED

            # --- CONFIDENCE + FLAGGING ---------------------------------------
            # Pass signals through as-is (None means "unavailable", not 0).
            composite, weak = composite_confidence(
                signals.logprob_top1,
                signals.logprob_margin,
                signals.fill_rate,
                signals.modal_agree,
            )
            signals.composite = composite
            result.signals = signals
            result.type_confidence = composite

            if composite < settings.classify_confidence_threshold:
                result.flagged_fields.insert(
                    0,
                    FlaggedItem(
                        doc=filename,
                        level="document",
                        reason="low classification confidence",
                        confidence=composite,
                        failed_signals=weak,
                    ),
                )
            # The fallback type has no dedicated schema — always surface it for review.
            if cls.doc_type is DocType.OTHER:
                result.flagged_fields.append(
                    FlaggedItem(
                        doc=filename,
                        level="document",
                        reason="unrecognised type — extracted with the generic fallback",
                    )
                )
            result.status = DocStatus.DONE

        except ingest_stage.UnsupportedFileError as exc:
            result.status = DocStatus.FAILED
            result.failed_stage = "ingest"
            result.error = str(exc)
        except Exception as exc:  # noqa: BLE001 - per-document isolation boundary
            result.status = DocStatus.FAILED
            result.failed_stage = current_stage
            result.error = f"{type(exc).__name__}: {exc}"

    return result


async def process_batch(files: list[tuple[str, bytes]]) -> BatchReport:
    batch_id = uuid.uuid4().hex
    collector = BatchCollector(batch_id)
    sem = asyncio.Semaphore(get_settings().max_concurrency)

    # return_exceptions=True so an unexpected raise in one task can never sink the
    # whole batch. _process_one is already defensive; this is belt-and-suspenders.
    results = await asyncio.gather(
        *(_process_one(name, data, collector, sem) for name, data in files),
        return_exceptions=True,
    )

    docs: list[DocumentResult] = []
    for (name, _), r in zip(files, results):
        if isinstance(r, DocumentResult):
            docs.append(r)
        else:  # an exception escaped isolation — record it as a failed doc
            docs.append(DocumentResult(
                filename=name, status=DocStatus.FAILED, failed_stage="orchestrator",
                error=f"{type(r).__name__}: {r}",
            ))

    summary = summarize(docs, collector)
    return BatchReport(batch_id=batch_id, summary=summary, documents=docs)
