#!/usr/bin/env bash
set -euo pipefail

INPUT_MODEL="${1:-expression_model.h5}"
OUTPUT_DIR="${2:-frontend/public/web_model}"

if [[ ! -f "$INPUT_MODEL" ]]; then
  echo "Input model not found: $INPUT_MODEL" >&2
  exit 1
fi

if ! command -v tensorflowjs_converter >/dev/null 2>&1; then
  echo "tensorflowjs_converter not found. Install with: pip install tensorflowjs" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
tensorflowjs_converter --input_format=keras "$INPUT_MODEL" "$OUTPUT_DIR"

echo "Exported TF.js model to $OUTPUT_DIR"
