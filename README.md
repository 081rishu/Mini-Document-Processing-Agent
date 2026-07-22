# Mini Document-Processing Agent

An agentic pipeline that takes a batch of mixed documents — resumes, invoices /
utility bills, agreements — and autonomously **classifies → routes → extracts
structured JSON → verifies**, then returns a batch report that flags failures and
low-confidence fields for human review.

It runs as a hosted FastAPI service with a minimal upload UI, a JSON API, and
auto-generated Swagger docs.

- **Live URL:** `https://<your-service>.onrender.com`  ·  UI at `/`  ·  Swagger at `/docs`
- **Quick test:**
  ```bash
  curl -s -X POST https://<your-service>.onrender.com/process \
    -F "files=@samples/inputs/resume_ada.txt" \
    -F "files=@samples/inputs/invoice_acme.txt" \
    -F "files=@samples/inputs/agreement_nda.txt" | jq .
  ```

---

## Architecture

A **per-document state machine** run over the batch with async I/O concurrency and
per-document isolation. One document's failure never aborts the batch.

```
        ┌──────────── batch orchestrator (asyncio.gather + Semaphore) ───────────┐
upload →│ per doc:  INGEST → CLASSIFY → ROUTE → EXTRACT → VERIFY → DONE           │→ SUMMARIZE → report
        │              └──── any stage raises → FAILED (isolated) ────────────────┘│
        └────────────────────────────────────────────────────────────────────────┘
```

`PENDING → INGESTED → CLASSIFIED → EXTRACTED → VERIFIED → DONE`, or `FAILED` from any stage.

| Stage | File | What it does |
|-------|------|--------------|
| **Ingest** | [`app/pipeline/ingest.py`](app/pipeline/ingest.py) | Parse PDF/DOCX/TXT/image → text. **Page triage:** any PDF page with < `IMAGE_PAGE_MIN_TOKENS` extractable text is treated as a scan; those pages are rasterized, stitched into one **collage**, and read by a vision model in a single cheap call. |
| **Classify** | [`app/pipeline/classify.py`](app/pipeline/classify.py) | Fuse text + vision notes → document type, emitted as a constrained single-letter choice with **logprobs** so confidence is *measured*, not self-reported. |
| **Route** | [`app/models/schemas.py`](app/models/schemas.py) | Registry maps type → Pydantic schema + extraction hint. Pure lookup. |
| **Extract** | [`app/pipeline/extract.py`](app/pipeline/extract.py) | OpenAI **structured outputs** constrained to the routed schema; "null, never guess". |
| **Verify** | [`app/pipeline/verify.py`](app/pipeline/verify.py) | Deterministic checks (invoice arithmetic, date ordering, email/phone) **+** an LLM grounding pass that nulls unsupported values. |
| **Summarize** | [`app/pipeline/summarize.py`](app/pipeline/summarize.py) | Aggregate counts, by-type, flag rate, cost, and the human-review queue. |

### Document taxonomy

Five first-class types, each with its own extraction schema, plus a structured
fallback for the long tail:

`resume` · `invoice` (invoices/utility bills/receipts) · `agreement` (contracts/NDAs) ·
`id_document` (passport/license/Aadhaar/PAN) · `form` (filled application/KYC forms) ·
`other`.

The taxonomy is chosen by **structural distinctness** (split a type only when its
documents need a genuinely different schema — an invoice and a utility bill share a
shape, an invoice and an ID card do not), not by enumerating topics. Adding a type is
a **one-entry change** to the registry in [`schemas.py`](app/models/schemas.py).

`other` is a *structured* fallback, not a dead end: it still returns a best-guess
type, a summary, and salient entities/dates/amounts/key-values — so an unseen document
(a medical report, say) yields useful data instead of nothing. (Its schema uses a list
of key/value pairs rather than an open-ended dict, because OpenAI strict
structured-outputs forbids free-form objects.)

The **orchestrator** ([`app/pipeline/orchestrator.py`](app/pipeline/orchestrator.py))
drives the state machine, computes the composite confidence, and fans the batch out
concurrently under a semaphore.

## Confidence: measured, not self-reported

Asking the model for `"confidence": 0.8` is miscalibrated and indefensible. Instead
classification confidence is a **composite of three independent signals**:

1. **Logprob margin (primary).** Classification is a constrained *single-token letter*
   choice (`A/B/C/D`), so `top_logprobs` gives a real probability distribution over the
   classes. We convert `p = exp(logprob)`, renormalize over the candidate letters, and
   use `p_top1` and the **margin `p_top1 − p_top2`** — a small margin (e.g. 0.44 vs 0.39)
   is exactly the ambiguous case we flag. Single-token letters guarantee *one token = one
   decision* (word labels tokenize to several tokens and would only expose the first).
2. **Cross-modal agreement.** When a document had image pages, the text-only read and the
   vision-informed read are compared; disagreement is a strong flag.
3. **Schema fill rate.** The fraction of the predicted type's required fields that came
   back populated. A near-empty schema is evidence of *wrong-schema routing* (a resume
   can't fill `invoice_number`/`total`).

`composite = f(logprob_signal, fill_rate)` with a strong penalty on cross-modal
disagreement (see `composite_confidence` in the orchestrator). Documents below
`CLASSIFY_CONFIDENCE_THRESHOLD` are flagged with **which signals were weak**.

## Hallucination mitigation

- **Structured outputs** constrain the shape — no free-form JSON parsing or repair.
- **Independent verifier pass** grounds every extracted value against the source text;
  unsupported values are **nulled** and flagged rather than trusted.
- **Deterministic cross-checks** — invoice line items must sum to the subtotal,
  `subtotal + tax == total`, dates must parse and be ordered, emails/phones must be
  well-formed.
- **"Null, never guess"** instruction + temperature 0 throughout.
- Confidence gating means low-support fields surface for a human instead of shipping silently.

## Failure handling

- Per-document `try/except` isolation — the batch always returns partial results, with
  each failure recorded as `{doc, stage, error}`.
- **Retries with exponential backoff** (`tenacity`) on transient OpenAI / rate-limit errors.
- `asyncio.Semaphore` caps concurrency (cost, rate limits, memory).
- Batch-size and per-file-size caps keep a run inside the request window.

## Observability

Every stage emits one structured JSON event under a per-document correlation id —
stage, model, latency, tokens, **cost**, and the confidence signals:

```json
{"doc_id":"a1b2","batch_id":"...","stage":"classify","model":"gpt-4o-mini",
 "latency_ms":812,"tokens_in":1300,"tokens_out":1,"cost_usd":0.0003,
 "signals":{"logprob_margin":0.05,"modal_agree":false},"status":"ok"}
```

The batch report surfaces the aggregates (throughput, flag rate, cost). The *same*
signals that gate flagging are the ones logged — one source of truth. See
[`app/obs/events.py`](app/obs/events.py).

## Prompt design

Three prompt families, all in [`app/llm/prompts.py`](app/llm/prompts.py):

- **Classify** — describes the letter-coded choices and asks for exactly one letter, so
  the answer is a single token with a clean logprob distribution.
- **Extract** — schema-driven, with the hard rule "use only what's in the source; return
  null, never a guess."
- **Verify** — a skeptical grounding checker that judges each extracted field
  `supported: bool` + `confidence` against the source text.

---

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then set OPENAI_API_KEY
uvicorn app.main:app --reload # → http://localhost:8000
```

Open `http://localhost:8000` for the upload UI, or:

```bash
curl -s -X POST http://localhost:8000/process \
  -F "files=@samples/inputs/resume_ada.txt" \
  -F "files=@samples/inputs/invoice_acme.txt" | jq .
```

### Tests

```bash
pip install pytest pytest-asyncio
pytest            # 23 tests, fully offline (LLM client is faked)
```

The suite covers the confidence math, schema routing, deterministic verification,
page triage, and full-orchestrator runs including failure isolation and
hallucination nulling.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Minimal upload UI |
| `GET` | `/health` | Liveness (+ whether the API key is configured) |
| `POST` | `/process` | Multipart file upload → full batch report |
| `GET` | `/docs` | Swagger UI |

Sample input files are in [`samples/inputs/`](samples/inputs/); an illustrative report
(matching the `/process` schema) is in
[`samples/outputs/sample_report.json`](samples/outputs/sample_report.json).

## Deploy (Render, free tier)

1. Push to GitHub. In Render: **New → Blueprint**, point at the repo ([`render.yaml`](render.yaml)).
2. Set `OPENAI_API_KEY` in the dashboard (it's `sync: false`, kept out of git).
3. Deploy. Health check is `/health`. All thresholds/models are env vars — tune on the
   live deploy without a redeploy.

Container is a single [`Dockerfile`](Dockerfile) (Python 3.12 slim; PyMuPDF/Pillow use
manylinux wheels, so no system packages needed).

---

## What I'd change for production

- **Async job model + durable queue.** The demo processes synchronously; at scale, `POST`
  returns a `job_id` backed by Celery/RQ + Redis, with object storage for uploads —
  for durability and horizontal scale, not for async-ness (I/O is already concurrent).
- **Evals.** A golden set with field-level precision/recall/F1, plus an LLM-as-judge for
  fuzzy fields and a regression suite gating deploys. Right now correctness is argued,
  not measured.
- **Confidence, turned up.** Add self-consistency (classify N× at temp>0, majority vote,
  agreement %) and calibrate the logprob→confidence mapping against the golden set.
- **Observability backend.** Ship the structured events to Langfuse / Arize Phoenix /
  OpenTelemetry for tracing, cost dashboards, and drift alerts.
- **Cost/scale.** Prompt caching, the Batch API for non-interactive runs, and model
  routing (escalate to a stronger model only on low confidence).
- **Robustness & governance.** Auth + rate limiting + per-tenant quotas; PII detection,
  redaction, and retention policy; a prompt/version registry; and a human-in-the-loop
  review UI backed directly by the `flagged_for_review` output.
- **Ingest depth.** Per-page (not collage) vision for full extraction from scans, plus
  real OCR fallback for low-quality images.

## Project layout

```
app/
  main.py            config.py
  models/    schemas.py  api.py
  pipeline/  orchestrator.py  ingest.py  classify.py  extract.py  verify.py  summarize.py
  llm/       client.py  prompts.py
  obs/       events.py
  static/    index.html
tests/       test_confidence.py  test_schemas.py  test_verify.py  test_ingest.py  test_pipeline.py
samples/     inputs/  outputs/
Dockerfile   render.yaml   requirements.txt   .env.example
```

> AI coding tools were used to accelerate the build; every design decision — the
> logprob-based confidence, the collage-based page triage, the two-layer verifier — is
> deliberate and defensible.
