# Overnight-ish CPU run for the generated sequence transcription pipeline.
# This machine's native Windows TensorFlow build reports no GPU support, so keep
# this moderate. For larger runs, WSL2/GPU or TensorFlow-DirectML will matter.

Set-Location $PSScriptRoot

.\.venv\Scripts\python.exe .\sequence_pitch_pipeline.py `
  --epochs 12 `
  --batch-size 2 `
  --train-batches 60 `
  --val-batches 12 `
  --test-batches 12 `
  --phrase-seconds 2.5 `
  --architecture cnn `
  --save-model .\sequence_pitch_overnight.keras `
  --write-previews 8 `
  --preview-dir .\sequence_previews
