"""Process-isolated CQT+BiGRU inference used by the Tkinter GUI."""

import argparse
import json

import librosa
from tensorflow import keras

from live_piano_roll import transcribe_recorded_audio
from sequence_pitch_pipeline import SR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--quiet-rms", required=True, type=float)
    args = parser.parse_args()

    audio, _ = librosa.load(args.audio, sr=SR, mono=True)
    model = keras.models.load_model(args.model)
    events, metrics = transcribe_recorded_audio(model, audio, args.quiet_rms)
    print(json.dumps({"events": events, "metrics": metrics}))


if __name__ == "__main__":
    main()
