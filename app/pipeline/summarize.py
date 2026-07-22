"""Summarize stage: aggregate per-document results into the batch report.

Pure function over the finished document records plus the batch's observability
collector — produces counts, the by-type breakdown, the flag rate, and the pooled
list of everything a human should review (low-confidence documents + flagged fields).
"""
from __future__ import annotations

from collections import Counter

from app.models.api import (
    BatchSummary,
    DocStatus,
    DocumentResult,
    Failure,
    FlaggedItem,
)
from app.obs.events import BatchCollector


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
        flagged_for_review=flagged,
        failures=failures,
        total_cost_usd=collector.total_cost_usd,
        duration_ms=collector.duration_ms,
    )
