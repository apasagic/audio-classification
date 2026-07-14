"""Whole-file and growing-buffer inference for trained sequence models."""

import argparse
import json
import time
from pathlib import Path

import librosa
import numpy as np
import sounddevice as sd
from tensorflow import keras

from sequence_pitch_pipeline import HOP_LENGTH, SR, audio_to_model_input, compact_segments


def load_audio(path):
    audio, _ = librosa.load(path, sr=SR, mono=True)
    peak = np.max(np.abs(audio))
    return (audio / peak if peak > 1e-8 else audio).astype(np.float32)


def predict_audio(model, audio, feature_type="raw"):
    """Transcribe the complete supplied audio buffer in one model call."""
    model_input = audio_to_model_input(audio, feature_type)
    probabilities = model.predict(model_input[None, ...], verbose=0)[0]
    return probabilities.argmax(axis=-1), probabilities.max(axis=-1)


def make_payload(classes, confidence):
    segments = compact_segments(classes, max_segments=len(classes))
    for segment in segments:
        start = round(segment["start_seconds"] * SR / HOP_LENGTH)
        stop = round(segment["end_seconds"] * SR / HOP_LENGTH)
        segment["mean_confidence"] = round(float(np.mean(confidence[start:stop])), 4)
    return {"frame_hop_ms": HOP_LENGTH * 1000 / SR, "segments": segments}


def growing_buffer_preview(model, seconds, update_seconds, device=None, feature_type="raw"):
    """Record continuously and re-run inference on everything captured so far."""
    captured = []
    started = time.time()

    def callback(indata, frames, callback_time, status):
        if status:
            print(status)
        captured.append(indata[:, 0].copy())

    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=int(SR * 0.05),
                        device=device, callback=callback):
        next_update = started + update_seconds
        while time.time() - started < seconds:
            time.sleep(0.05)
            if time.time() >= next_update and captured:
                audio = np.concatenate(captured)
                classes, confidence = predict_audio(model, audio, feature_type)
                recent = make_payload(classes, confidence)["segments"][-8:]
                print(json.dumps({"captured_seconds": round(len(audio) / SR, 2), "recent": recent}, indent=2))
                next_update += update_seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio")
    parser.add_argument("--feature-type", choices=["raw", "cqt", "stft"], default="raw")
    parser.add_argument("--output-json")
    parser.add_argument("--microphone-seconds", type=float, default=0)
    parser.add_argument("--update-seconds", type=float, default=1.0)
    parser.add_argument("--device", type=int)
    args = parser.parse_args()

    if not args.audio and not args.microphone_seconds:
        parser.error("provide --audio or --microphone-seconds")
    model = keras.models.load_model(args.model)
    if args.audio:
        classes, confidence = predict_audio(model, load_audio(args.audio), args.feature_type)
        result = make_payload(classes, confidence)
        print(json.dumps(result, indent=2))
        if args.output_json:
            Path(args.output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.microphone_seconds:
        growing_buffer_preview(model, args.microphone_seconds, args.update_seconds, args.device, args.feature_type)


if __name__ == "__main__":
    main()
