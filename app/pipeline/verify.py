"""Verify stage: the core hallucination-mitigation step. Two independent layers.

1. Deterministic validation — cheap, type-aware business rules (invoice arithmetic,
   date ordering/parseability, email/phone format). These never hard-fail a document;
   they raise field-level flags.
2. LLM grounding — a second pass asks a stronger model whether each extracted value
   is actually supported by the source text. Unsupported values are nulled (a
   hallucination removed); supported-but-low-confidence values are kept and flagged.

Returns the (possibly nulled) data plus the field-level flags and the LLM usage.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from pydantic import BaseModel

from app.config import get_settings
from app.llm.client import Usage, get_client
from app.llm.prompts import VERIFY_SYSTEM, verify_user
from app.models.api import FlaggedItem
from app.models.schemas import DocType

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE_RE = re.compile(r"^[+()\d][\d\s().\-]{6,}$")


# --- LLM grounding structured output ----------------------------------------
class FieldJudgement(BaseModel):
    path: str
    supported: bool
    confidence: float


class VerificationReport(BaseModel):
    judgements: list[FieldJudgement]


@dataclass
class VerifyResult:
    data: dict[str, Any]
    flagged_fields: list[FlaggedItem] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


def _parse_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def _deterministic_checks(doc_type: DocType, data: dict[str, Any], filename: str) -> list[FlaggedItem]:
    flags: list[FlaggedItem] = []

    def flag(path: str, reason: str) -> None:
        flags.append(FlaggedItem(doc=filename, level="field", field=path, reason=reason))

    if doc_type is DocType.INVOICE:
        items = [li for li in (data.get("line_items") or []) if isinstance(li, dict)]

        def line_net(li: dict) -> float:
            # Prefer the explicit net/taxable value; else amount minus any discount.
            if li.get("net_amount") is not None:
                return li["net_amount"]
            return (li.get("amount") or 0) - (li.get("discount") or 0)

        line_sum = sum(line_net(li) for li in items)
        charges = [c for c in (data.get("additional_charges") or []) if isinstance(c, dict)]
        charges_sum = sum((c.get("amount") or 0) for c in charges)
        subtotal, tax, total = data.get("subtotal"), data.get("tax"), data.get("total")

        # Robust invariant: total = subtotal + tax + order-level charges. This holds
        # regardless of whether a fee is modelled as a line item or a charge.
        if subtotal is not None and total is not None:
            expected = subtotal + (tax or 0) + charges_sum
            if abs(expected - total) > 0.02 * max(abs(total), 1):
                flag("total", f"subtotal + tax + charges ({expected:.2f}) != total ({total})")
        # Line-to-subtotal reconciliation only when there are no order-level charges,
        # since otherwise the subtotal's composition is genuinely ambiguous.
        if items and not charges and subtotal is not None and abs(line_sum - subtotal) > 0.02 * max(abs(subtotal), 1):
            flag(
                "subtotal",
                f"net line items sum to {line_sum:.2f} but subtotal is {subtotal}",
            )
        issue, due = _parse_date(data.get("issue_date")), _parse_date(data.get("due_date"))
        if data.get("issue_date") and issue is None:
            flag("issue_date", "unparseable date")
        if data.get("due_date") and due is None:
            flag("due_date", "unparseable date")
        if issue and due and due < issue:
            flag("due_date", "due date precedes issue date")

    elif doc_type is DocType.RESUME:
        email = data.get("email")
        if email and not _EMAIL_RE.match(str(email)):
            flag("email", "malformed email")
        phone = data.get("phone")
        if phone and not _PHONE_RE.match(str(phone)):
            flag("phone", "implausible phone number")

    elif doc_type is DocType.AGREEMENT:
        if data.get("effective_date") and _parse_date(data.get("effective_date")) is None:
            flag("effective_date", "unparseable date")

    elif doc_type is DocType.ID_DOCUMENT:
        issue = _parse_date(data.get("issue_date"))
        expiry = _parse_date(data.get("expiry_date"))
        if data.get("date_of_birth") and _parse_date(data.get("date_of_birth")) is None:
            flag("date_of_birth", "unparseable date")
        if issue and expiry and expiry < issue:
            flag("expiry_date", "expiry precedes issue date")

    return flags


def _set_by_path(data: dict[str, Any], path: str, value: Any) -> None:
    """Set a dotted path (supporting list indices) to ``value``; best-effort."""
    parts = path.split(".")
    cur: Any = data
    try:
        for p in parts[:-1]:
            cur = cur[int(p)] if p.isdigit() else cur[p]
        last = parts[-1]
        if isinstance(cur, list) and last.isdigit():
            cur[int(last)] = value
        elif isinstance(cur, dict):
            cur[last] = value
    except (KeyError, IndexError, ValueError, TypeError):
        pass


async def verify(
    doc_type: DocType,
    data: dict[str, Any],
    text: str,
    filename: str,
    vlm_notes: str | None = None,
) -> VerifyResult:
    import json

    flags = _deterministic_checks(doc_type, data, filename)

    # The grounding source is the extracted text plus any vision transcription, so
    # scanned/image documents (which have no text) can still be verified.
    source = "\n\n".join(s for s in (text, vlm_notes) if s and s.strip())

    # LLM grounding pass. Skipped for the generic fallback (no schema to ground).
    usage = Usage()
    if doc_type is not DocType.OTHER and source.strip():
        outcome = await get_client().parse(
            model=get_settings().verify_model,
            system=VERIFY_SYSTEM,
            user=verify_user(source, json.dumps(data, default=str, indent=2)),
            response_format=VerificationReport,
        )
        usage = outcome.usage
        report: VerificationReport | None = outcome.parsed  # type: ignore[assignment]
        threshold = get_settings().field_confidence_threshold
        for j in (report.judgements if report else []):
            if not j.supported:
                _set_by_path(data, j.path, None)
                flags.append(FlaggedItem(
                    doc=filename, level="field", field=j.path,
                    reason="unsupported by source (nulled)", confidence=j.confidence,
                ))
            elif j.confidence < threshold:
                flags.append(FlaggedItem(
                    doc=filename, level="field", field=j.path,
                    reason="low grounding confidence", confidence=j.confidence,
                ))

    return VerifyResult(data=data, flagged_fields=flags, usage=usage)
