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
    # Machine-readable flag class, e.g. "hallucination", "deterministic",
    # "low_confidence", "fallback_type", "no_extraction", "vision_failed",
    # "verify_failed", "low_grounding". Drives the reliability metrics.
    category: str | None = None
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
    grounded: bool = False  # whether the LLM grounding/verify pass ran on this doc
    error: str | None = None
    failed_stage: str | None = None


class ReliabilityMetrics(BaseModel):
    """How much to trust this batch's output — measurable without ground-truth labels.

    Hallucination is what the verifier caught (LLM-judged, a proxy); deterministic
    violations are rule-based (arithmetic/date/format — closer to ground truth).
    """
    docs_verified: int = 0                    # docs that ran the LLM grounding pass
    fields_extracted: int = 0                 # non-null leaf values kept after verify
    hallucinated_fields: int = 0              # values nulled as unsupported by the verifier
    hallucination_rate: float = 0.0           # hallucinated / (kept + hallucinated)
    deterministic_violations: int = 0         # rule violations (arithmetic/date/format)
    deterministic_violation_rate: float = 0.0  # docs with >=1 violation / succeeded


class BatchSummary(BaseModel):
    total: int
    succeeded: int
    failed: int
    by_type: dict[str, int] = Field(default_factory=dict)
    flag_rate: float = 0.0
    reliability: ReliabilityMetrics = Field(default_factory=ReliabilityMetrics)
    flagged_for_review: list[FlaggedItem] = Field(default_factory=list)
    failures: list[Failure] = Field(default_factory=list)
    total_cost_usd: float = 0.0
    duration_ms: int = 0


class BatchReport(BaseModel):
    batch_id: str
    summary: BatchSummary
    documents: list[DocumentResult] = Field(default_factory=list)
