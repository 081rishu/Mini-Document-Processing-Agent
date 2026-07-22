"""Lightweight structured observability.

Every pipeline stage emits one ``StageEvent`` under a per-document correlation id.
Events are (a) written as one-line JSON to the logger — greppable / ships to any
log backend — and (b) collected per batch so the summary report can surface
throughput, cost, flag rate and per-type confidence. The *same* confidence signals
that gate human-review flagging are the ones logged here: one source of truth.

In production this emitter would forward to Langfuse / Arize Phoenix / OpenTelemetry;
the interface is deliberately that thin.
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger("docagent.obs")


@dataclass
class StageEvent:
    batch_id: str
    doc_id: str
    filename: str
    stage: str
    status: str = "ok"
    model: str | None = None
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    confidence: float | None = None
    signals: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def emit(self) -> None:
        logger.info(json.dumps(asdict(self), default=str))


class BatchCollector:
    """Accumulates events for one batch and derives the aggregate metrics."""

    def __init__(self, batch_id: str) -> None:
        self.batch_id = batch_id
        self.events: list[StageEvent] = []
        self._t0 = time.perf_counter()

    def record(self, event: StageEvent) -> None:
        event.emit()
        self.events.append(event)

    @contextmanager
    def stage(self, doc_id: str, filename: str, stage: str, model: str | None = None):
        """Times a stage and emits an event; caller mutates the yielded event.

        On exception the event is marked failed and re-raised so the orchestrator's
        per-document isolation can catch it.
        """
        ev = StageEvent(
            batch_id=self.batch_id, doc_id=doc_id, filename=filename,
            stage=stage, model=model,
        )
        start = time.perf_counter()
        try:
            yield ev
        except Exception as exc:  # noqa: BLE001 - deliberately broad; isolation boundary
            ev.status = "error"
            ev.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            ev.latency_ms = int((time.perf_counter() - start) * 1000)
            self.record(ev)

    @property
    def total_cost_usd(self) -> float:
        return round(sum(e.cost_usd for e in self.events), 6)

    @property
    def duration_ms(self) -> int:
        return int((time.perf_counter() - self._t0) * 1000)
