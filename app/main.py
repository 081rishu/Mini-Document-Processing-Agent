"""FastAPI service entrypoint.

Synchronous single-shot processing (the demo batches are small): POST files ->
get the full report back. Internally the batch still runs async-concurrent across
LLM calls. A minimal upload UI is served at ``/`` and Swagger at ``/docs``.
"""
from __future__ import annotations

import logging
import sys

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.config import get_settings
from app.models.api import BatchReport
from app.pipeline.orchestrator import process_batch


def _configure_logging() -> None:
    """Route our logs to stdout (so hosts don't tag them as errors) and silence the
    per-request HTTP client chatter that otherwise floods a small instance's logs."""
    level = getattr(logging, get_settings().log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        stream=sys.stdout,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    # These log one INFO line per OpenAI call — pure noise in production logs.
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


_configure_logging()

app = FastAPI(
    title="Mini Document-Processing Agent",
    description=(
        "Upload a batch of mixed documents (resumes, invoices/utility bills, "
        "agreements). Each is classified, routed to a type-specific schema, "
        "extracted with structured outputs, verified against its source, and "
        "flagged for human review when confidence is low."
    ),
    version="1.0.0",
)

_STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@app.get("/health")
async def health() -> dict[str, str | bool]:
    return {"status": "ok", "openai_key_configured": bool(get_settings().openai_api_key)}


@app.post("/process", response_model=BatchReport)
async def process(files: list[UploadFile] = File(...)) -> BatchReport:
    settings = get_settings()
    if not settings.openai_api_key:
        raise HTTPException(500, "OPENAI_API_KEY is not configured on the server.")
    if not files:
        raise HTTPException(400, "No files uploaded.")
    if len(files) > settings.max_batch_files:
        raise HTTPException(
            413, f"Too many files: {len(files)} > limit {settings.max_batch_files}."
        )

    payloads: list[tuple[str, bytes]] = []
    for f in files:
        data = await f.read()
        if len(data) > settings.max_file_bytes:
            raise HTTPException(
                413, f"{f.filename} exceeds the {settings.max_file_mb} MB limit."
            )
        payloads.append((f.filename or "unnamed", data))

    report = await process_batch(payloads)
    return report


@app.exception_handler(Exception)
async def unhandled(_request, exc: Exception) -> JSONResponse:  # noqa: ANN001
    logging.exception("unhandled error")
    return JSONResponse(status_code=500, content={"detail": f"internal error: {exc}"})
