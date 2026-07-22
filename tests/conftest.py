"""Shared test fixtures — a fake LLM client so the pipeline runs offline.

The fake is injected by setting the module-level singleton in ``app.llm.client``;
every stage calls ``get_client()`` which returns that singleton, so no network or
API key is needed under test.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

import app.llm.client as llm_client
from app.llm.client import ParseOutcome, Usage, VisionOutcome
from app.pipeline.verify import VerificationReport


@dataclass
class FakeClient:
    """Configurable stand-in for LLMClient.

    - ``letter_for(text)`` decides the classification letter,
    - ``extract_data`` is validated into whatever extraction schema is requested,
    - ``judgements`` is returned from the verify (grounding) call,
    - ``vision_text`` is returned from vision calls.
    """
    letter_rules: dict[str, str] = field(default_factory=dict)
    default_letter: str = "F"  # F = other (structured fallback)
    top1: float = 0.95
    top2: float = 0.02
    extract_data: dict = field(default_factory=dict)
    judgements: list = field(default_factory=list)
    vision_text: str = "A scanned document."
    # Failure-injection toggles for the degraded-path tests.
    empty_distribution: bool = False  # simulate missing logprobs
    vision_error: bool = False
    verify_error: bool = False

    def letter_for(self, text: str) -> str:
        for kw, letter in self.letter_rules.items():
            if kw.lower() in (text or "").lower():
                return letter
        return self.default_letter

    async def classify(self, system, user, candidate_letters):
        letter = self.letter_for(user)
        if self.empty_distribution:
            return _ClsOutcome(letter=letter, distribution={}, usage=Usage(10, 1, 0.0))
        dist = {c: self.top2 for c in candidate_letters}
        dist[letter] = self.top1
        z = sum(dist.values())
        dist = {c: v / z for c, v in dist.items()}
        return _ClsOutcome(letter=letter, distribution=dist, usage=Usage(10, 1, 0.0))

    async def parse(self, model, system, user, response_format):
        if response_format is VerificationReport or (
            isinstance(response_format, type)
            and issubclass(response_format, VerificationReport)
        ):
            if self.verify_error:
                raise RuntimeError("verify boom")
            return ParseOutcome(
                parsed=VerificationReport(judgements=self.judgements),
                usage=Usage(50, 20, 0.0),
            )
        parsed = response_format.model_validate(self.extract_data)
        return ParseOutcome(parsed=parsed, usage=Usage(100, 40, 0.0))

    async def vision_describe(self, system, user, image_b64):
        if self.vision_error:
            raise RuntimeError("vision boom")
        return VisionOutcome(text=self.vision_text, usage=Usage(200, 30, 0.0))


@dataclass
class _ClsOutcome:
    letter: str
    distribution: dict
    usage: Usage


@pytest.fixture
def fake_client():
    def _install(**kwargs) -> FakeClient:
        client = FakeClient(**kwargs)
        llm_client._client = client
        return client

    yield _install
    llm_client._client = None
