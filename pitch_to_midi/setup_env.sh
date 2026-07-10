#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV="${VENV:-.venv}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "$VENV" ]; then
  echo "Creating virtual environment: $VENV"
  "$PYTHON_BIN" -m venv "$VENV"
fi

echo "Upgrading pip"
"$VENV/bin/python" -m pip install --upgrade pip

echo "Installing pitch_to_midi requirements"
"$VENV/bin/python" -m pip install -r requirements.txt

echo "Done. Use:"
echo "  $VENV/bin/python sequence_pitch_pipeline.py --preview-only --write-previews 5"
echo "  $VENV/bin/python sequence_pitch_pipeline.py --epochs 50 --batch-size 8 --train-batches 500 --val-batches 80 --test-batches 80 --phrase-seconds 4 --architecture cnn_gru --save-model sequence_pitch_large.keras"