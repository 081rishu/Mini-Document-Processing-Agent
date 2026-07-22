"""Runtime configuration, loaded from environment (12-factor).

Every knob that we expect to tune on the live deploy — model choices, the
image-page token threshold, confidence thresholds, concurrency and size caps —
lives here so it can be changed via Render env vars without a redeploy.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Later files win, so a git-ignored .env.local overrides .env for real secrets.
    model_config = SettingsConfigDict(env_file=(".env", ".env.local"), extra="ignore")

    # Required
    openai_api_key: str = ""

    # Models (cost tiering: cheap for high-volume classify/extract, strong for verify/vision)
    classify_model: str = "gpt-4o-mini"
    extract_model: str = "gpt-4o-mini"
    verify_model: str = "gpt-4o"
    vision_model: str = "gpt-4o"

    # Ingest / page triage
    image_page_min_tokens: int = 50

    # Confidence gating
    classify_confidence_threshold: float = 0.6
    field_confidence_threshold: float = 0.5

    # Concurrency & limits
    max_concurrency: int = 5
    max_batch_files: int = 25
    max_file_mb: int = 15

    # LLM robustness
    llm_max_retries: int = 3
    llm_timeout_seconds: int = 60

    # Logging: level for our own logs; third-party HTTP client noise is always quieted.
    # Set to WARNING on a small/free instance to cut log volume to failures + requests.
    log_level: str = "INFO"

    # Debug: when set, the vision collage for each scanned doc is written here as PNG
    # (off by default — the service is otherwise stateless and stores nothing).
    collage_debug_dir: str = ""

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
