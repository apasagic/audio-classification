#!/usr/bin/env bash
set -euo pipefail
cd /root/audio-classification/pitch_to_midi
source .venv/bin/activate
export PITCH_TO_MIDI_SF2=/usr/share/sounds/sf2/FluidR3_GM.sf2
export PYTHONUNBUFFERED=1
python sequence_pitch_pipeline.py \
  --epochs 40 \
  --batch-size 8 \
  --train-batches 200 \
  --val-batches 40 \
  --test-batches 40 \
  --phrase-seconds 2.5 \
  --architecture cnn_gru \
  --frequency-features flatten \
  --learning-rate 0.005 \
  --dropout 0.20 \
  --feature-type cqt \
  --augment \
  --tinysol-dir ./datasets/tinysol \
  --nsynth-dir ./datasets/nsynth_test/nsynth-test \
  --real-sample-probability 0.5 \
  --patience 7 \
  --reduce-lr-patience 3 \
  --min-learning-rate 0.00001 \
  --checkpoint-model ./cqt_gru_best.keras \
  --save-model ./cqt_gru_final.keras \
  --csv-log ./experiments/cqt_gru_gpu_full_history.csv \
  --history-svg ./experiments/cqt_gru_gpu_full_curves.svg \
  --results-json ./experiments/cqt_gru_gpu_full.json \
  --sample-predictions 5
