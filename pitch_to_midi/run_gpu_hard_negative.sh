#!/usr/bin/env bash
set -euo pipefail
cd /root/audio-classification/pitch_to_midi
source .venv/bin/activate
export PITCH_TO_MIDI_SF2=/usr/share/sounds/sf2/FluidR3_GM.sf2
export PYTHONUNBUFFERED=1

# Train a separate candidate. The current cqt_gru_best.keras is not overwritten.
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
  --hard-negative-probability 0.15 \
  --hard-negative-test-batches 20 \
  --patience 7 \
  --reduce-lr-patience 3 \
  --min-learning-rate 0.00001 \
  --checkpoint-model ./cqt_gru_hard_negative_best.keras \
  --save-model ./cqt_gru_hard_negative_final.keras \
  --csv-log ./experiments/cqt_gru_hard_negative_history.csv \
  --history-svg ./experiments/cqt_gru_hard_negative_curves.svg \
  --results-json ./experiments/cqt_gru_hard_negative.json \
  --sample-predictions 5
