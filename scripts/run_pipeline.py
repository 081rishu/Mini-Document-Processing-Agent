#!/usr/bin/env python
"""Run the full pipeline locally and (by default) dump the vision collages.

Unlike the hosted service this is a convenience harness for testing/inspection: it
runs the exact same `process_batch`, but also saves each scanned document's collage
PNG so you can see what the vision model was given.

Usage:
    python scripts/run_pipeline.py <file-or-folder> [more...] [options]

Options:
    --collage-dir DIR   Where to save collages (default: debug/collages)
    --no-collage        Don't save collages
    --out FILE          Also write the full JSON report to FILE

Examples:
    python scripts/run_pipeline.py samples/inputs/pdfs
    python scripts/run_pipeline.py samples/inputs/resume_ada.txt some/scan.pdf --out report.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Allow running as `python scripts/run_pipeline.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_SKIP = {".gitkeep", "README.md"}


def _ignore(name: str) -> bool:
    # Skip housekeeping files and WSL/NTFS alternate-data-stream stubs
    # (e.g. "file.pdf:Zone.Identifier") that show up as separate entries.
    return (
        name in _SKIP
        or name.startswith(".")
        or ":Zone.Identifier" in name
    )


def _collect(paths: list[str]) -> list[tuple[str, bytes]]:
    files: list[tuple[str, bytes]] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for f in sorted(path.iterdir()):
                if f.is_file() and not _ignore(f.name):
                    files.append((f.name, f.read_bytes()))
        elif path.is_file():
            files.append((path.name, path.read_bytes()))
        else:
            print(f"warning: skipping missing path {p}", file=sys.stderr)
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the document pipeline and dump collages.")
    ap.add_argument("paths", nargs="+", help="files and/or folders to process")
    ap.add_argument("--collage-dir", default="debug/collages")
    ap.add_argument("--no-collage", action="store_true")
    ap.add_argument("--out", help="write the full JSON report to this file")
    args = ap.parse_args()

    # Must be set BEFORE importing app (settings are cached at first import).
    if not args.no_collage:
        os.environ["COLLAGE_DEBUG_DIR"] = args.collage_dir

    from app.pipeline.orchestrator import process_batch  # noqa: E402

    files = _collect(args.paths)
    if not files:
        print("No files to process.", file=sys.stderr)
        return 1

    print(f"Processing {len(files)} file(s)...", file=sys.stderr)
    report = asyncio.run(process_batch(files))

    s = report.summary
    print(f"\nbatch {report.batch_id}")
    print(f"  processed {s.succeeded}/{s.total}  failed {s.failed}  "
          f"flag_rate {s.flag_rate}  cost ${s.total_cost_usd}  {s.duration_ms}ms")
    print(f"  by_type: {s.by_type}")
    for d in report.documents:
        line = f"  - {d.filename}: {d.type.value if d.type else '-'} " \
               f"[{d.status.value}] conf={d.type_confidence}"
        if d.error:
            line += f"  ERROR@{d.failed_stage}: {d.error}"
        print(line)
        for fl in d.flagged_fields:
            print(f"      flag: {fl.level} {fl.field or ''} - {fl.reason}")

    if not args.no_collage:
        saved = list(Path(args.collage_dir).glob("*.png")) if Path(args.collage_dir).exists() else []
        print(f"\ncollages saved: {len(saved)} -> {args.collage_dir}/", file=sys.stderr)

    if args.out:
        Path(args.out).write_text(json.dumps(report.model_dump(), indent=2, default=str))
        print(f"report written -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
