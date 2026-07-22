"""Classify stage: constrained single-token choice with a logprob-based confidence.

Returns the chosen ``DocType`` plus the two logprob-derived signals (top-1
probability and the top1-top2 margin) that feed the composite confidence in the
orchestrator. Text is truncated to a representative window — classification needs
the gist, not the whole document.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.llm.client import Usage, get_client
from app.llm.prompts import CLASSIFY_SYSTEM, classify_user
from app.models.schemas import LETTER_TO_TYPE, DocType

_MAX_CHARS = 8000  # representative window fed to the classifier


@dataclass
class ClassifyResult:
    doc_type: DocType
    distribution: dict[str, float]  # letter -> prob
    # None when the model returned no usable logprobs — "signal unavailable", which is
    # NOT the same as "zero confidence". The confidence composite treats it as unknown.
    logprob_top1: float | None
    logprob_margin: float | None
    usage: Usage


def _truncate(text: str) -> str:
    if len(text) <= _MAX_CHARS:
        return text
    head = text[: _MAX_CHARS - 500]
    tail = text[-500:]
    return f"{head}\n...[truncated]...\n{tail}"


async def classify(text: str, vlm_notes: str | None) -> ClassifyResult:
    candidates = list(LETTER_TO_TYPE.keys())
    outcome = await get_client().classify(
        CLASSIFY_SYSTEM,
        classify_user(_truncate(text), vlm_notes),
        candidate_letters=candidates,
    )

    doc_type = LETTER_TO_TYPE.get(outcome.letter, DocType.OTHER)

    dist = outcome.distribution
    if dist:
        ordered = sorted(dist.values(), reverse=True)
        top1 = ordered[0]
        top2 = ordered[1] if len(ordered) > 1 else 0.0
        margin = top1 - top2
    else:
        # No logprobs came back — mark the signal unavailable rather than 0.0.
        top1 = margin = None

    return ClassifyResult(
        doc_type=doc_type,
        distribution=dist,
        logprob_top1=top1,
        logprob_margin=margin,
        usage=outcome.usage,
    )
