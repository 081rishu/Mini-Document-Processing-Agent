"""API-facing and cross-stage data models.

These types are both the state-machine record that flows through the pipeline and
the JSON the service returns, so the internal representation and the contract stay
in one place.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.models.schemas import DocType


class DocStatus(str, Enum):
    """State-machine states for a single document."""
    PENDING = "PENDING"
    INGESTED = "INGESTED"
    CLASSIFIED = "CLASSIFIED"
    EXTRACTED = "EXTRACTED"
    VERIFIED = "VERIFIED"
    DONE = "DONE"
    FAILED = "FAILED"


class ConfidenceSignals(BaseModel):
    """The three independent signals behind classification confidence."""
    logprob_margin: float | None = Field(
        None, description="p(top1) - p(top2) over the class letter distribution"
    )
    logprob_top1: float | None = Field(None, description="probability of the chosen class")
    modal_agree: bool | None = Field(
        None, description="did the text signal and the vision (VLM) read agree on type?"
    )
    fill_rate: float | None = Field(
        None, description="fraction of the predicted type's required fields populated"
    )
    composite: float | None = Field(None, description="combined 0-1 confidence")


class FlaggedItem(BaseModel):
    doc: str
    level: str  # "document" | "field"
    field: str | None = None
    reason: str
    confidence: float | None = None
    failed_signals: list[str] = Field(default_factory=list)


class Failure(BaseModel):
    doc: str
    stage: str
    error: str


class DocumentResult(BaseModel):
    filename: str
    type: DocType | None = None
    type_confidence: float | None = None
    status: DocStatus = DocStatus.PENDING
    data: dict[str, Any] | None = None
    flagged_fields: list[FlaggedItem] = Field(default_factory=list)
    signals: ConfidenceSignals | None = None
    error: str | None = None
    failed_stage: str | None = None


class BatchSummary(BaseModel):
    total: int
    succeeded: int
    failed: int
    by_type: dict[str, int] = Field(default_factory=dict)
    flag_rate: float = 0.0
    flagged_for_review: list[FlaggedItem] = Field(default_factory=list)
    failures: list[Failure] = Field(default_factory=list)
    total_cost_usd: float = 0.0
    duration_ms: int = 0


class BatchReport(BaseModel):
    batch_id: str
    summary: BatchSummary
    documents: list[DocumentResult] = Field(default_factory=list)
