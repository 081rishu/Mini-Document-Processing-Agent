"""Unit tests for the composite confidence math and fill-rate signal."""
from __future__ import annotations

from app.pipeline.extract import compute_fill_rate
from app.pipeline.orchestrator import composite_confidence


def test_high_confidence_when_all_signals_strong():
    score, weak = composite_confidence(top1=0.95, margin=0.9, fill_rate=1.0, modal_agree=True)
    assert score > 0.9
    assert weak == []


def test_low_margin_is_flagged_and_lowers_score():
    strong, _ = composite_confidence(0.9, 0.8, 1.0, None)
    weakish, weak = composite_confidence(0.45, 0.05, 1.0, None)
    assert weakish < strong
    assert "logprob_margin" in weak


def test_cross_modal_disagreement_penalises():
    agree, _ = composite_confidence(0.9, 0.8, 1.0, True)
    disagree, weak = composite_confidence(0.9, 0.8, 1.0, False)
    assert disagree < agree
    assert "modal_agree" in weak


def test_low_fill_rate_flags_wrong_schema():
    _, weak = composite_confidence(0.9, 0.8, 0.2, None)
    assert "fill_rate" in weak


def test_missing_signals_do_not_crash():
    score, _ = composite_confidence(0.9, 0.8, None, None)
    assert 0.0 <= score <= 1.0


def test_unavailable_logprobs_use_neutral_prior_not_zero():
    # The key regression: missing logprobs must NOT read as zero confidence.
    score, weak = composite_confidence(None, None, 1.0, None)
    assert score > 0.6  # would have been ~0.4 (flagged) under the old 0-coercion
    assert "logprob_unavailable" in weak


def test_unavailable_logprobs_with_no_other_signals_is_neutral():
    score, weak = composite_confidence(None, None, None, None)
    assert score == 0.5  # pure neutral prior
    assert "logprob_unavailable" in weak


def test_compute_fill_rate():
    assert compute_fill_rate({"a": "x", "b": [1]}, ["a", "b"]) == 1.0
    assert compute_fill_rate({"a": "x", "b": []}, ["a", "b"]) == 0.5
    assert compute_fill_rate({}, ["a", "b"]) == 0.0
    assert compute_fill_rate({}, []) == 1.0  # no required fields -> trivially full
