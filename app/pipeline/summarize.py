"""Summarize stage: aggregate per-document results into the batch report.

Pure function over the finished document records plus the batch's observability
collector — produces counts, the by-type breakdown, the flag rate, and the pooled
list of everything a human should review (low-confidence documents + flagged fields).
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from app.models.api import (
    BatchSummary,
    DocStatus,
    DocumentResult,
    Failure,
    FlaggedItem,
    ReliabilityMetrics,
)
from app.obs.events import BatchCollector


def _count_leaf_values(value: Any) -> int:
    """Count non-null, non-empty scalar leaves in an extracted-data structure."""
    if value is None:
        return 0
    if isinstance(value, dict):
        return sum(_count_leaf_values(v) for v in value.values())
    if isinstance(value, list):
        return sum(_count_leaf_values(v) for v in value)
    if isinstance(value, str) and not value.strip():
        return 0
    return 1


def _reliability(docs: list[DocumentResult], succeeded: int) -> ReliabilityMetrics:
    """Label-free reliability signals derived from the verify output.

    hallucination_rate is over fields the verifier actually judged: the values it kept
    plus the ones it nulled as unsupported. deterministic_violation_rate is doc-level.
    """
    ok = [d for d in docs if d.status is not DocStatus.FAILED]
    docs_verified = sum(1 for d in ok if d.grounded)

    hallucinated = 0
    deterministic = 0
    docs_with_violation = 0
    for d in ok:
        cats = [f.category for f in d.flagged_fields]
        h = sum(1 for c in cats if c == "hallucination")
        v = sum(1 for c in cats if c == "deterministic")
        hallucinated += h
        deterministic += v
        if v:
            docs_with_violation += 1

    kept_fields = sum(_count_leaf_values(d.data) for d in ok if d.grounded)
    judged = kept_fields + hallucinated  # verifier saw kept + nulled-unsupported

    return ReliabilityMetrics(
        docs_verified=docs_verified,
        fields_extracted=kept_fields,
        hallucinated_fields=hallucinated,
        hallucination_rate=round(hallucinated / judged, 3) if judged else 0.0,
        deterministic_violations=deterministic,
        deterministic_violation_rate=(
            round(docs_with_violation / succeeded, 3) if succeeded else 0.0
        ),
    )


def summarize(docs: list[DocumentResult], collector: BatchCollector) -> BatchSummary:
    total = len(docs)
    failed = sum(1 for d in docs if d.status is DocStatus.FAILED)
    succeeded = total - failed

    by_type: Counter[str] = Counter(
        d.type.value for d in docs if d.type is not None and d.status is not DocStatus.FAILED
    )

    failures = [
        Failure(doc=d.filename, stage=d.failed_stage or "unknown", error=d.error or "unknown")
        for d in docs
        if d.status is DocStatus.FAILED
    ]

    flagged: list[FlaggedItem] = []
    for d in docs:
        if d.status is DocStatus.FAILED:
            continue
        flagged.extend(d.flagged_fields)

    flagged_docs = sum(1 for d in docs if d.status is not DocStatus.FAILED and d.flagged_fields)
    flag_rate = round(flagged_docs / total, 3) if total else 0.0

    return BatchSummary(
        total=total,
        succeeded=succeeded,
        failed=failed,
        by_type=dict(by_type),
        flag_rate=flag_rate,
        reliability=_reliability(docs, succeeded),
        flagged_for_review=flagged,
        failures=failures,
        total_cost_usd=collector.total_cost_usd,
        duration_ms=collector.duration_ms,
    )
