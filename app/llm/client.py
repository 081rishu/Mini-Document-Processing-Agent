"""Thin async wrapper over the OpenAI SDK.

Responsibilities kept here so the pipeline stages stay declarative:
- retries with exponential backoff on transient/rate-limit errors (tenacity),
- structured-outputs parsing (``responses``-style ``parse``) for extraction/verify,
- a constrained single-token classification call that returns the *logprob
  distribution* over our class letters (the backbone of measured confidence),
- vision calls for the image-page collage,
- token accounting -> USD so every call feeds the observability cost metric.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TypeVar

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings

T = TypeVar("T", bound=BaseModel)

# Approximate USD per 1M tokens (input, output). Used only for the cost metric;
# not billing-accurate, and centralised so it is trivial to update.
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}

_RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


def _cost(model: str, usage) -> float:
    if usage is None:
        return 0.0
    inp, out = _PRICING.get(model, (0.0, 0.0))
    return round(
        (usage.prompt_tokens * inp + usage.completion_tokens * out) / 1_000_000, 6
    )


@dataclass
class Usage:
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


@dataclass
class ClassifyOutcome:
    letter: str  # chosen class letter (top-1)
    distribution: dict[str, float]  # letter -> probability, over our candidates
    usage: Usage


@dataclass
class ParseOutcome:
    parsed: BaseModel | None
    usage: Usage


@dataclass
class VisionOutcome:
    text: str
    usage: Usage


class LLMClient:
    def __init__(self) -> None:
        s = get_settings()
        self._settings = s
        self._client = AsyncOpenAI(
            api_key=s.openai_api_key, timeout=s.llm_timeout_seconds
        )

    def _retry(self):
        return retry(
            retry=retry_if_exception_type(_RETRYABLE),
            wait=wait_exponential(multiplier=1, min=1, max=20),
            stop=stop_after_attempt(self._settings.llm_max_retries),
            reraise=True,
        )

    # -- Classification: constrained single-token choice with logprobs ---------
    async def classify(
        self, system: str, user: str, candidate_letters: list[str]
    ) -> ClassifyOutcome:
        @self._retry()
        async def _call():
            return await self._client.chat.completions.create(
                model=self._settings.classify_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                max_tokens=1,
                logprobs=True,
                top_logprobs=10,
            )

        resp = await _call()
        choice = resp.choices[0]
        letter = (choice.message.content or "").strip()[:1].upper()

        distribution: dict[str, float] = {}
        try:
            top = choice.logprobs.content[0].top_logprobs
            # Sum (don't overwrite) mass across token variants that normalise to the
            # same letter — e.g. "B", " B", "b" all count toward class B. A naive dict
            # comprehension would keep only the last variant and deflate confidence.
            raw: dict[str, float] = {}
            for t in top:
                key = t.token.strip().upper()
                raw[key] = raw.get(key, 0.0) + math.exp(t.logprob)
            # Keep only our candidate letters, then renormalise into a clean
            # distribution over the classes (drops the vocab tail).
            picked = {c: raw.get(c, 0.0) for c in candidate_letters}
            z = sum(picked.values())
            if z > 0:
                distribution = {c: p / z for c, p in picked.items()}
        except (AttributeError, IndexError, TypeError):
            distribution = {}

        if letter not in candidate_letters and distribution:
            letter = max(distribution, key=distribution.get)

        return ClassifyOutcome(
            letter=letter,
            distribution=distribution,
            usage=Usage(
                tokens_in=resp.usage.prompt_tokens,
                tokens_out=resp.usage.completion_tokens,
                cost_usd=_cost(self._settings.classify_model, resp.usage),
            ),
        )

    # -- Structured extraction / verification ---------------------------------
    async def parse(
        self, model: str, system: str, user: str, response_format: type[T]
    ) -> ParseOutcome:
        @self._retry()
        async def _call():
            return await self._client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                response_format=response_format,
            )

        resp = await _call()
        return ParseOutcome(
            parsed=resp.choices[0].message.parsed,
            usage=Usage(
                tokens_in=resp.usage.prompt_tokens,
                tokens_out=resp.usage.completion_tokens,
                cost_usd=_cost(model, resp.usage),
            ),
        )

    # -- Vision: describe an image-page collage --------------------------------
    async def vision_describe(self, system: str, user: str, image_b64: str) -> VisionOutcome:
        @self._retry()
        async def _call():
            return await self._client.chat.completions.create(
                model=self._settings.vision_model,
                messages=[
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}",
                                    "detail": "low",
                                },
                            },
                        ],
                    },
                ],
                temperature=0,
                max_tokens=900,
            )

        resp = await _call()
        return VisionOutcome(
            text=resp.choices[0].message.content or "",
            usage=Usage(
                tokens_in=resp.usage.prompt_tokens,
                tokens_out=resp.usage.completion_tokens,
                cost_usd=_cost(self._settings.vision_model, resp.usage),
            ),
        )


_client: LLMClient | None = None


def get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
