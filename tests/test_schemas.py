"""Registry / routing tests."""
from __future__ import annotations

from app.models.schemas import (
    LETTER_TO_TYPE,
    REGISTRY,
    TYPE_TO_LETTER,
    DocType,
    Invoice,
    Resume,
    spec_for,
)


def test_every_doctype_has_a_spec():
    for dt in DocType:
        assert dt in REGISTRY
        assert REGISTRY[dt].model is not None


def test_routing_returns_expected_model():
    assert spec_for(DocType.RESUME).model is Resume
    assert spec_for(DocType.INVOICE).model is Invoice


def test_letter_maps_are_consistent_and_single_char():
    assert set(LETTER_TO_TYPE.values()) == set(DocType)
    for letter in LETTER_TO_TYPE:
        assert len(letter) == 1
    for dt, letter in TYPE_TO_LETTER.items():
        assert LETTER_TO_TYPE[letter] is dt


def _has_open_object(schema: dict) -> bool:
    """True if any subschema is an open-ended object (dict), which OpenAI strict
    structured-outputs rejects — the bug that broke the old fallback."""
    if isinstance(schema, dict):
        ap = schema.get("additionalProperties")
        if isinstance(ap, dict):  # additionalProperties: {schema} == free-form dict
            return True
        return any(_has_open_object(v) for v in schema.values())
    if isinstance(schema, list):
        return any(_has_open_object(v) for v in schema)
    return False


def test_all_extraction_schemas_are_strict_compatible():
    # Every routed model must be usable as an OpenAI structured-output response_format.
    for spec in REGISTRY.values():
        assert not _has_open_object(spec.model.model_json_schema()), (
            f"{spec.model.__name__} contains an open-ended object (dict) field"
        )


def test_other_fallback():
    # spec_for never raises; OTHER is the structured fallback
    assert spec_for(DocType.OTHER).doc_type is DocType.OTHER


def test_new_types_registered():
    for dt in (DocType.ID_DOCUMENT, DocType.FORM):
        assert dt in REGISTRY
        assert REGISTRY[dt].required_fields  # each first-class type has required fields
