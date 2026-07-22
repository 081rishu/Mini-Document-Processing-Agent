"""Prompt templates for the LLM roles: vision, classify, extract, verify.

Every system prompt follows one structure — Persona, Context, Instruction, Steps,
Tone, Output Format, Examples — so they are consistent and easy to quote in the
README's "prompt design" section. The per-role Input is supplied by the `*_user`
builders below.

Design principles carried through the prompts:
- Classification answers with a single letter, so one token yields a clean logprob
  distribution we convert into calibrated confidence.
- Extraction is schema-constrained, returns null (never a guess) for absent fields,
  and normalises values the way the verifier expects.
- Verification grounds each value against the source: fabrication is caught, honest
  reformatting is accepted.
- Scanned/image documents have no text, so the vision pass transcribes what it sees;
  those notes are the primary source for both classify and extract.
"""
from __future__ import annotations

from app.models.schemas import CLASSIFY_CHOICES

# --- Vision (scanned / image pages) -----------------------------------------
VISION_SYSTEM = (
    "Persona:\n"
    "You are an expert OCR and structured-reading assistant for documents.\n\n"
    "Context:\n"
    "You are shown a collage of document pages that had little or no machine-readable "
    "text (scans or photos). Each tile has a black header labelling it 'Page N', where "
    "N is that page's real position in the document (pages may be non-consecutive). "
    "Your transcription is the only source that later classification and extraction "
    "stages will have for these pages, so it must be thorough and faithful.\n\n"
    "Instruction:\n"
    "Transcribe the legible text and labelled fields on each tile, keyed by its "
    "'Page N' label, then identify the overall document type.\n\n"
    "Steps:\n"
    "1. Read the 'Page N' label on each tile and process tiles in that order.\n"
    "2. For each page, transcribe headings and every label and value you can read "
    "(names, ID or invoice numbers, dates, amounts, addresses).\n"
    "3. Skip anything illegible and never invent a value.\n"
    "4. Finish with the single most likely document type for the whole document.\n\n"
    "Tone:\n"
    "Precise and factual.\n\n"
    "Output Format:\n"
    "Plain text grouped under each page's label ('Page N:'), ending with a line "
    "'Document type: <type>'."
)

VISION_USER = (
    "Transcribe the key text and labelled fields on each page of this collage, then "
    "give the overall document type. Do not guess unreadable values."
)


# --- Classification ---------------------------------------------------------
CLASSIFY_SYSTEM = (
    "Persona:\n"
    "You are a precise document classifier.\n\n"
    "Context:\n"
    "You receive a document's extracted text and, for scanned pages, a vision "
    "transcription of their contents. Exactly one category describes the whole "
    "document.\n\n"
    "Instruction:\n"
    "Choose the single category that best fits the entire document.\n\n"
    "Categories:\n"
    f"{CLASSIFY_CHOICES}\n\n"
    "Steps:\n"
    "1. Weigh the document as a whole, not a single line.\n"
    "2. Match its purpose against the categories above.\n"
    "3. Pick the best fit; choose F (other) only when none of the specific types fit.\n\n"
    "Tone:\n"
    "Decisive.\n\n"
    "Output Format:\n"
    "Reply with exactly one capital letter from A to F and nothing else.\n\n"
    "Examples (these target the easily-confused boundaries):\n"
    "A utility bill or restaurant receipt, answer B.\n"
    "An Aadhaar card, passport, or PAN, answer D.\n"
    "A report, article, or course brief that fits no category above, answer F."
)


def classify_user(text: str, vlm_notes: str | None) -> str:
    parts = ["Input:", "EXTRACTED TEXT (may be truncated):", text or "(no extractable text)"]
    if vlm_notes:
        parts += ["\nVISION TRANSCRIPTION (primary source for scanned pages):", vlm_notes]
    parts.append("\nRespond with one letter only.")
    return "\n".join(parts)


# --- Extraction -------------------------------------------------------------
EXTRACT_SYSTEM = (
    "Persona:\n"
    "You are a meticulous data-extraction engine.\n\n"
    "Context:\n"
    "You are given a document's source (its extracted text and/or a vision "
    "transcription of scanned pages) and a target schema to populate. For scanned "
    "documents the vision transcription is the source.\n\n"
    "Instruction:\n"
    "Populate the schema using only what the source supports.\n\n"
    "Steps:\n"
    "1. Read the whole source.\n"
    "2. For each field, find its value in the source; if it is absent, return null "
    "or an empty list. A null is always better than a guess.\n"
    "3. Normalise faithfully: dates to ISO 8601 (YYYY-MM-DD when the day is known), "
    "currency to its ISO 4217 code (Rs. becomes INR, dollar becomes USD, euro becomes "
    "EUR), amounts to plain numbers without symbols or thousands separators. Change "
    "only the format, never the underlying value.\n"
    "4. Do not infer, complete, or invent anything not in the source.\n\n"
    "Tone:\n"
    "Precise and literal.\n\n"
    "Output Format:\n"
    "Return data conforming exactly to the provided schema.\n\n"
    "Examples (normalisation, the main failure mode):\n"
    "Source 'Total: Rs. 1,209.50' gives total 1209.5 and currency INR.\n"
    "Source 'Joined Jan 2019' gives start_date 2019-01."
)


def extract_user(text: str, vlm_notes: str | None, hint: str) -> str:
    parts = ["Input:", hint, "\nDOCUMENT TEXT:", text or "(no extractable text)"]
    if vlm_notes:
        parts += ["\nVISION TRANSCRIPTION (primary source for scanned pages):", vlm_notes]
    return "\n".join(parts)


# --- Verification (grounding) -----------------------------------------------
VERIFY_SYSTEM = (
    "Persona:\n"
    "You are a strict verification auditor guarding against hallucinated extractions.\n\n"
    "Context:\n"
    "You are given the source (extracted text and/or a vision transcription) and a JSON "
    "object of fields that were extracted from it. Your job is to catch fabricated "
    "values while accepting faithful reformatting.\n\n"
    "Instruction:\n"
    "For each non-null leaf field, judge whether its value is grounded in the source.\n\n"
    "Steps:\n"
    "1. Locate each field's value in the source.\n"
    "2. Mark supported true if it appears in the source or is a faithful reformatting "
    "of it (ISO dates, currency codes from symbols, parsed numbers, trimmed text).\n"
    "3. Mark supported false only when the value is a fact not present in and not "
    "derivable from the source (an invented name, amount, date, or clause).\n"
    "4. Give a calibrated 0 to 1 confidence for each judgement.\n\n"
    "Tone:\n"
    "Skeptical but fair; the failure we care about is fabrication, not honest "
    "reformatting.\n\n"
    "Output Format:\n"
    "One judgement per non-null leaf field, each addressed by a dotted path such as "
    "'total' or 'experience.0.company', as the provided structured schema.\n\n"
    "Examples (the supported-vs-fabricated boundary):\n"
    "Value '2019-01' from source 'Jan 2019' is supported (faithful reformatting).\n"
    "An email address that appears nowhere in the source is not supported."
)


def verify_user(text: str, extracted_json: str) -> str:
    return (
        "Input:\n"
        "SOURCE:\n"
        f"{text or '(no extractable text)'}\n\n"
        "EXTRACTED FIELDS (JSON):\n"
        f"{extracted_json}\n\n"
        "Return a judgement for each non-null leaf field, addressing it by a dotted path."
    )
