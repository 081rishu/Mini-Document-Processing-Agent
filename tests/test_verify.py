"""Deterministic-verification and path-nulling tests (no LLM)."""
from __future__ import annotations

from app.models.schemas import DocType
from app.pipeline.verify import _deterministic_checks, _set_by_path


def test_invoice_arithmetic_mismatch_flagged():
    data = {
        "line_items": [{"amount": 50}, {"amount": 40}],
        "subtotal": 100,  # should be 90
        "tax": 10,
        "total": 110,
    }
    flags = _deterministic_checks(DocType.INVOICE, data, "inv.pdf")
    fields = {f.field for f in flags}
    assert "subtotal" in fields


def test_invoice_totals_consistent_not_flagged():
    data = {
        "line_items": [{"amount": 50}, {"amount": 40}],
        "subtotal": 90,
        "tax": 10,
        "total": 100,
    }
    flags = _deterministic_checks(DocType.INVOICE, data, "inv.pdf")
    assert flags == []


def test_discounted_line_items_reconcile_to_subtotal():
    # Swiggy-style invoice: gross amounts differ from subtotal because of discounts.
    data = {
        "line_items": [
            {"amount": 249.0, "discount": 128.70, "net_amount": 120.30},
            {"amount": 245.0, "discount": 110.0, "net_amount": 135.0},
            {"amount": 15.0, "discount": 0.0, "net_amount": 15.0},
        ],
        "subtotal": 270.30,
        "tax": 13.52,
        "total": 283.82,
    }
    flags = _deterministic_checks(DocType.INVOICE, data, "swiggy.pdf")
    assert flags == []  # net values reconcile; must not false-positive


def test_additional_charges_reconcile_to_total():
    # Order-level delivery fee sits between subtotal and total; must reconcile.
    data = {
        "line_items": [{"amount": 100.0, "net_amount": 100.0}],
        "additional_charges": [{"description": "Delivery fee", "amount": 20.0}],
        "subtotal": 100.0,
        "tax": 0.0,
        "total": 120.0,
    }
    assert _deterministic_checks(DocType.INVOICE, data, "inv.pdf") == []


def test_negative_charge_as_discount_reconciles():
    data = {
        "line_items": [{"amount": 100.0, "net_amount": 100.0}],
        "additional_charges": [{"description": "Coupon", "amount": -10.0}],
        "subtotal": 100.0,
        "tax": 0.0,
        "total": 90.0,
    }
    assert _deterministic_checks(DocType.INVOICE, data, "inv.pdf") == []


def test_gross_amounts_without_net_still_flag_real_mismatch():
    data = {
        "line_items": [{"amount": 100.0}, {"amount": 50.0}],
        "subtotal": 200.0,  # genuinely inconsistent
        "tax": 0.0,
        "total": 200.0,
    }
    flags = _deterministic_checks(DocType.INVOICE, data, "inv.pdf")
    assert any(f.field == "subtotal" for f in flags)


def test_due_before_issue_flagged():
    data = {"issue_date": "2024-05-10", "due_date": "2024-05-01"}
    flags = _deterministic_checks(DocType.INVOICE, data, "inv.pdf")
    assert any(f.field == "due_date" for f in flags)


def test_malformed_email_flagged():
    flags = _deterministic_checks(DocType.RESUME, {"email": "not-an-email"}, "cv.pdf")
    assert any(f.field == "email" for f in flags)


def test_set_by_path_nulls_nested_list_field():
    data = {"experience": [{"company": "Acme"}, {"company": "Globex"}]}
    _set_by_path(data, "experience.1.company", None)
    assert data["experience"][1]["company"] is None
    assert data["experience"][0]["company"] == "Acme"
