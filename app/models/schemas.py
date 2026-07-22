"""Per-document-type extraction schemas + the routing registry.

Each ``DocType`` maps to a Pydantic model (the extraction target that OpenAI
structured-outputs is constrained to) plus a short, type-specific extraction hint
used in the prompt. Adding a new document type is a one-entry change to REGISTRY.

Design notes:
- Every field is Optional. The extractor is instructed to emit ``null`` when a
  value is not present in the source rather than guessing — null is a first-class,
  honest answer and is how we avoid hallucinated fields.
- ``REQUIRED_FIELDS`` per type is *not* a validation gate (we never hard-fail a doc
  for a missing field); it defines the denominator for the "schema fill rate"
  confidence signal in the classifier — a resume misclassified as an invoice will
  leave the invoice's required fields empty.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class DocType(str, Enum):
    RESUME = "resume"
    INVOICE = "invoice"          # invoices, utility bills, receipts, POs
    AGREEMENT = "agreement"      # contracts, NDAs, terms, offer letters
    ID_DOCUMENT = "id_document"  # passport, license, national ID, Aadhaar, PAN
    FORM = "form"                # filled application / registration / KYC forms
    OTHER = "other"              # structured fallback for the long tail


# --- Resume -----------------------------------------------------------------
class ExperienceItem(BaseModel):
    company: str | None = None
    title: str | None = None
    start_date: str | None = Field(None, description="ISO 8601 if possible, else as written")
    end_date: str | None = Field(None, description="ISO 8601, or 'present'")


class EducationItem(BaseModel):
    institution: str | None = None
    degree: str | None = None
    year: str | None = None


class Resume(BaseModel):
    full_name: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = None
    skills: list[str] = Field(default_factory=list)
    experience: list[ExperienceItem] = Field(default_factory=list)
    education: list[EducationItem] = Field(default_factory=list)
    total_years_experience: float | None = None


# --- Invoice / utility bill -------------------------------------------------
class LineItem(BaseModel):
    description: str | None = None
    quantity: float | None = None
    unit_price: float | None = None
    amount: float | None = Field(None, description="Gross line amount before discount")
    discount: float | None = Field(None, description="Discount applied to this line, if any")
    net_amount: float | None = Field(
        None, description="Net/taxable amount after discount (amount - discount)"
    )


class Charge(BaseModel):
    """A generic, non-line-item monetary adjustment (fee, discount, tip, round-off).

    Deliberately open-ended so the schema captures charges from any vendor without
    enumerating vendor-specific fields. A negative amount denotes a discount/credit.
    """
    description: str | None = None
    amount: float | None = None


class Invoice(BaseModel):
    vendor_name: str | None = None
    invoice_number: str | None = None
    issue_date: str | None = Field(None, description="ISO 8601 if possible")
    due_date: str | None = Field(None, description="ISO 8601 if possible")
    line_items: list[LineItem] = Field(default_factory=list)
    additional_charges: list[Charge] = Field(
        default_factory=list,
        description="Order-level fees or discounts not tied to a single line "
        "(delivery, service, packaging, tip, convenience fee, round-off, etc.). "
        "Do NOT put tax here — tax has its own field.",
    )
    subtotal: float | None = None
    tax: float | None = None
    total: float | None = None
    currency: str | None = Field(
        None, description="ISO 4217 code; normalise symbols (₹/Rs.->INR, $->USD, €->EUR)"
    )


# --- Agreement / contract ---------------------------------------------------
class Agreement(BaseModel):
    title: str | None = None
    parties: list[str] = Field(default_factory=list)
    effective_date: str | None = Field(None, description="ISO 8601 if possible")
    term: str | None = Field(None, description="Duration or end condition of the agreement")
    governing_law: str | None = None
    key_obligations: list[str] = Field(default_factory=list)
    termination: str | None = Field(None, description="Termination clause summary")
    signatories: list[str] = Field(default_factory=list)


# --- ID document ------------------------------------------------------------
class IdDocument(BaseModel):
    id_type: str | None = Field(
        None, description="Kind of ID, e.g. Aadhaar, Passport, Driver's License, PAN"
    )
    full_name: str | None = None
    id_number: str | None = None
    date_of_birth: str | None = Field(None, description="ISO 8601 if possible")
    gender: str | None = None
    address: str | None = None
    issue_date: str | None = Field(None, description="ISO 8601 if possible")
    expiry_date: str | None = Field(None, description="ISO 8601 if possible")
    issuing_authority: str | None = None
    nationality: str | None = None


# --- Form -------------------------------------------------------------------
class FormField(BaseModel):
    label: str | None = None
    value: str | None = None


class Form(BaseModel):
    form_title: str | None = Field(
        None, description="Title/name of the form, e.g. 'Form 6 - Voter Registration'"
    )
    form_number: str | None = None
    issuing_body: str | None = Field(None, description="Authority or organization")
    applicant_name: str | None = None
    fields: list[FormField] = Field(
        default_factory=list, description="The filled-in label/value pairs on the form"
    )
    submission_date: str | None = Field(None, description="ISO 8601 if possible")


# --- Fallback ---------------------------------------------------------------
class KeyValue(BaseModel):
    key: str | None = None
    value: str | None = None


class GenericDocument(BaseModel):
    """Structured fallback for documents outside the known types.

    Uses a list of key/value pairs (not an open-ended dict) because OpenAI strict
    structured-outputs forbids free-form objects. Still returns real, useful data —
    a best-guess type, a summary, and salient entities/dates/amounts.
    """
    detected_type: str | None = Field(
        None, description="Your best-guess of the document type, e.g. 'medical report'"
    )
    document_summary: str | None = None
    entities: list[str] = Field(
        default_factory=list, description="Notable people, organisations, or places"
    )
    dates: list[str] = Field(default_factory=list)
    amounts: list[str] = Field(default_factory=list)
    key_values: list[KeyValue] = Field(default_factory=list)


class SchemaSpec(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    doc_type: DocType
    model: type[BaseModel]
    extraction_hint: str
    required_fields: list[str]
    gloss: str = ""  # short description shown to the classifier


REGISTRY: dict[DocType, SchemaSpec] = {
    DocType.RESUME: SchemaSpec(
        doc_type=DocType.RESUME,
        model=Resume,
        extraction_hint=(
            "Extract the candidate's contact details, skills, work experience "
            "(most recent first), and education."
        ),
        required_fields=["full_name", "experience", "skills"],
        gloss="CV / candidate profile",
    ),
    DocType.INVOICE: SchemaSpec(
        doc_type=DocType.INVOICE,
        model=Invoice,
        extraction_hint=(
            "Extract vendor, invoice/bill number, and dates. For each product/service "
            "line capture amount, and (when shown) its discount and net/taxable amount. "
            "Put order-level fees or discounts not tied to a single line into "
            "additional_charges, but never tax. Capture subtotal, tax, total, and "
            "currency as an ISO 4217 code."
        ),
        required_fields=["vendor_name", "total"],
        gloss="invoice, utility bill, receipt, purchase order",
    ),
    DocType.AGREEMENT: SchemaSpec(
        doc_type=DocType.AGREEMENT,
        model=Agreement,
        extraction_hint=(
            "Extract the agreement title, all parties, effective date, term, "
            "governing law, key obligations, and termination conditions."
        ),
        required_fields=["parties", "effective_date"],
        gloss="contract, NDA, terms, offer letter",
    ),
    DocType.ID_DOCUMENT: SchemaSpec(
        doc_type=DocType.ID_DOCUMENT,
        model=IdDocument,
        extraction_hint=(
            "Extract the identity document's type, the holder's name and ID number, "
            "date of birth, address, issue/expiry dates, and issuing authority."
        ),
        required_fields=["full_name", "id_number"],
        gloss="passport, driver's license, national ID, Aadhaar, PAN",
    ),
    DocType.FORM: SchemaSpec(
        doc_type=DocType.FORM,
        model=Form,
        extraction_hint=(
            "Extract the form's title/number, the issuing body, the applicant's name, "
            "and every filled-in field as a label/value pair."
        ),
        required_fields=["form_title", "fields"],
        gloss="filled application / registration / KYC form",
    ),
    DocType.OTHER: SchemaSpec(
        doc_type=DocType.OTHER,
        model=GenericDocument,
        extraction_hint=(
            "This document did not match a known type. Give your best-guess type, a "
            "short summary, and any salient entities, dates, amounts, and key/value pairs."
        ),
        required_fields=[],
        gloss="anything not matching the above",
    ),
}


def spec_for(doc_type: DocType) -> SchemaSpec:
    return REGISTRY.get(doc_type, REGISTRY[DocType.OTHER])


# Single-token letter codes for constrained, logprob-measurable classification.
# One token == one decision (word labels tokenize to several tokens and would
# only expose the first token's probability). See classify.py. Order is fixed so
# letters stay stable as types are added.
_LETTER_ORDER: list[DocType] = [
    DocType.RESUME,      # A
    DocType.INVOICE,     # B
    DocType.AGREEMENT,   # C
    DocType.ID_DOCUMENT, # D
    DocType.FORM,        # E
    DocType.OTHER,       # F
]
LETTER_TO_TYPE: dict[str, DocType] = {
    chr(ord("A") + i): dt for i, dt in enumerate(_LETTER_ORDER)
}
TYPE_TO_LETTER: dict[DocType, str] = {v: k for k, v in LETTER_TO_TYPE.items()}

# Letter = value (gloss) — richer choices improve classification accuracy.
CLASSIFY_CHOICES = "\n".join(
    f"{letter} = {dtype.value} ({REGISTRY[dtype].gloss})"
    for letter, dtype in LETTER_TO_TYPE.items()
)
