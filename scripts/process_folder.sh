#!/usr/bin/env bash
# Post every file in a folder to the /process endpoint as one batch.
# Usage: scripts/process_folder.sh <folder> [base_url]
set -euo pipefail

FOLDER="${1:?usage: process_folder.sh <folder> [base_url]}"
BASE_URL="${2:-http://localhost:8000}"

args=()
count=0
shopt -s nullglob
for f in "$FOLDER"/*; do
  [ -f "$f" ] || continue
  case "$(basename "$f")" in .gitkeep|README.md) continue;; esac
  args+=(-F "files=@${f}")
  count=$((count + 1))
done

if [ "$count" -eq 0 ]; then
  echo "No files found in $FOLDER" >&2
  exit 1
fi

echo "Posting ${count} file(s) to ${BASE_URL}/process ..." >&2
curl -s -X POST "${BASE_URL}/process" "${args[@]}" | (jq . 2>/dev/null || cat)
