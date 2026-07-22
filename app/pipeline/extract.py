"""Extract stage: schema-constrained extraction via OpenAI structured outputs.

The routed Pydantic model is passed as ``response_format`` so the model can only
return schema-conforming JSON — no free-form parsing or repair. The prompt instructs
"null, never a guess" for absent fields. Also computes the required-field fill rate,
one of the three classification-confidence signals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import get_settings
from app.llm.client import Usage, get_client
from app.llm.prompts import EXTRACT_SYSTEM, extract_user
from app.models.schemas import DocType, SchemaSpec


@dataclass
class ExtractResult:
    data: dict[str, Any]
    # None for the OTHER fallback (no required fields, so fill rate is not meaningful
    # and must not inflate confidence).
    fill_rate: float | None
    usage: Usage


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, list, dict)) and len(value) == 0:
        return True
    return False


def compute_fill_rate(data: dict[str, Any], required: list[str]) -> float:
    """Fraction of the type's required fields that came back populated.

    A near-zero fill rate is evidence the document was routed to the wrong schema
    (e.g. a resume sent to the invoice extractor), which the orchestrator folds
    back into classification confidence.
    """
    if not required:
        return 1.0
    filled = sum(1 for f in required if not _is_empty(data.get(f)))
    return round(filled / len(required), 3)


async def extract(text: str, vlm_notes: str | None, spec: SchemaSpec) -> ExtractResult:
    model = get_settings().extract_model
    client = get_client()
    user = extract_user(text, vlm_notes, spec.extraction_hint)

    outcome = await client.parse(
        model=model, system=EXTRACT_SYSTEM, user=user, response_format=spec.model
    )
    # Repair-retry: if the model returned nothing parseable (e.g. a refusal or an
    # empty structured response), nudge it once more before giving up.
    total_usage = outcome.usage
    if outcome.parsed is None:
        retry = await client.parse(
            model=model,
            system=EXTRACT_SYSTEM,
            user=user + "\n\nReturn a valid object conforming exactly to the schema.",
            response_format=spec.model,
        )
        outcome = retry
        total_usage = _sum_usage(total_usage, retry.usage)

    parsed = outcome.parsed
    data = parsed.model_dump() if parsed is not None else {}
    fill_rate = (
        None
        if spec.doc_type is DocType.OTHER
        else compute_fill_rate(data, spec.required_fields)
    )
    return ExtractResult(data=data, fill_rate=fill_rate, usage=total_usage)


def _sum_usage(a: Usage, b: Usage) -> Usage:
    return Usage(
        tokens_in=a.tokens_in + b.tokens_in,
        tokens_out=a.tokens_out + b.tokens_out,
        cost_usd=round(a.cost_usd + b.cost_usd, 6),
    )
