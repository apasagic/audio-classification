"""
Frame-wise pitch transcription starter for generated note sequences.

This is closer to transcription than cnn_pitch_pipeline.py:
- one training example is a multi-note phrase
- audio may contain pauses, legato transitions, and overlapping note tails
- the target is one class per STFT frame: silence or MIDI note

It is still monophonic: each frame has one label. That matches the first goal of
humming/singing/whistling one melody line into MIDI.
"""

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
from scipy.io import wavfile
from tensorflow import keras

from midi_renderer import fix_length, get_soundfont_path, render_note


SR = 16_000
PHRASE_SECONDS = 4.0
N_FFT = 1024
HOP_LENGTH = 128
MIDI_NOTES = np.arange(36, 73)  # C2..C5
SILENCE_CLASS = 0
NUM_CLASSES = len(MIDI_NOTES) + 1


def midi_to_class(midi_note):
    return int(midi_note - MIDI_NOTES[0] + 1)


def class_to_midi(class_index):
    return None if class_index == SILENCE_CLASS else int(MIDI_NOTES[class_index - 1])


def normalize(audio):
    return (audio / (np.max(np.abs(audio)) + 1e-8)).astype(np.float32)


def add_reverb(audio, rng):
    tail_len = int(SR * rng.uniform(0.04, 0.22))
    impulse = np.exp(-np.linspace(0, 6, tail_len)).astype(np.float32)
    wet = np.convolve(audio, impulse, mode="full")[: len(audio)]
    return normalize(audio + rng.uniform(0.04, 0.20) * wet)


def augment_phrase(audio, rng):
    """Augment without moving labels in time."""
    audio = audio * rng.uniform(0.35, 1.25)
    if rng.random() < 0.75:
        audio = audio + rng.normal(0, rng.uniform(0.001, 0.035), len(audio))
    if rng.random() < 0.35:
        # Small cents-level variation keeps the MIDI label effectively unchanged.
        audio = librosa.effects.pitch_shift(audio, sr=SR, n_steps=rng.uniform(-0.18, 0.18))
    if rng.random() < 0.50:
        audio = add_reverb(audio, rng)
    if rng.random() < 0.25:
        audio = np.tanh(audio * rng.uniform(1.0, 1.8))
    return normalize(audio)


def audio_to_logstft(audio):
    spec = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH, window="hann")
    db = librosa.amplitude_to_db(np.abs(spec), ref=np.max)
    image = ((db + 80.0) / 80.0).clip(0.0, 1.0)
    return image[..., None].astype(np.float32)


def render_sequence(soundfont_path, rng, phrase_seconds=PHRASE_SECONDS):
    """Create a synthetic monophonic phrase and frame-independent note events."""
    total_samples = int(SR * phrase_seconds)
    audio = np.zeros(total_samples, dtype=np.float32)
    events = []
    t = float(rng.uniform(0.05, 0.35))

    while t < phrase_seconds - 0.20:
        midi_note = int(rng.choice(MIDI_NOTES))
        note_duration = float(rng.uniform(0.18, 0.85))
        end_t = min(t + note_duration, phrase_seconds)
        velocity = float(rng.uniform(0.45, 1.0))

        # Render slightly past the symbolic end so natural/synth tails can overlap.
        render_duration = min(note_duration + rng.uniform(0.04, 0.25), phrase_seconds - t)
        note_audio = render_note(midi_note, SR, render_duration, soundfont_path) * velocity
        start_sample = int(t * SR)
        stop_sample = min(start_sample + len(note_audio), total_samples)
        audio[start_sample:stop_sample] += note_audio[: stop_sample - start_sample]
        events.append({"start": t, "end": end_t, "midi": midi_note, "velocity": velocity})

        # Negative gap means legato/overlap; positive gap means a pause.
        t = end_t + float(rng.uniform(-0.08, 0.35))

    return normalize(audio), events


def frame_labels(num_frames, events):
    labels = np.zeros(num_frames, dtype=np.int32)
    frame_times = librosa.frames_to_time(np.arange(num_frames), sr=SR, hop_length=HOP_LENGTH)
    for event in events:
        active = (frame_times >= event["start"]) & (frame_times < event["end"])
        labels[active] = midi_to_class(event["midi"])
    return labels


class GeneratedSequenceData(keras.utils.Sequence):
    def __init__(self, batches, batch_size, soundfont_path, seed, phrase_seconds, augment=True, **kwargs):
        super().__init__(**kwargs)
        self.batches = batches
        self.batch_size = batch_size
        self.soundfont_path = soundfont_path
        self.phrase_seconds = phrase_seconds
        self.rng = np.random.default_rng(seed)
        self.augment = augment

    def __len__(self):
        return self.batches

    def __getitem__(self, _):
        X, y = [], []
        for _ in range(self.batch_size):
            audio, events = render_sequence(self.soundfont_path, self.rng, self.phrase_seconds)
            if self.augment:
                audio = augment_phrase(audio, self.rng)
            image = audio_to_logstft(audio)
            X.append(image)
            y.append(frame_labels(image.shape[1], events))
        return np.array(X), np.array(y)


def build_model(input_shape, architecture="cnn"):
    """Frequency-compressing CNN that preserves time frames."""
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPooling2D(pool_size=(2, 1))(x)
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 1))(x)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 1))(x)
    x = keras.layers.Permute((2, 1, 3))(x)      # time, freq, channels
    x = keras.layers.TimeDistributed(keras.layers.GlobalAveragePooling1D())(x)
    if architecture == "cnn_gru":
        x = keras.layers.Bidirectional(keras.layers.GRU(48, return_sequences=True))(x)
    else:
        x = keras.layers.Conv1D(96, 5, padding="same", activation="relu")(x)
        x = keras.layers.Conv1D(96, 5, padding="same", activation="relu")(x)
    x = keras.layers.Dropout(0.20)(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def write_previews(out_dir, count, soundfont_path, phrase_seconds):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(123)
    for i in range(count):
        audio, events = render_sequence(soundfont_path, rng, phrase_seconds)
        augmented = augment_phrase(audio.copy(), rng)
        wavfile.write(out / f"sequence_{i+1:02d}.wav", SR, (augmented * 32767).astype(np.int16))
        (out / f"sequence_{i+1:02d}.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    print(f"Wrote {count} sequence previews to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-batches", type=int, default=200)
    parser.add_argument("--val-batches", type=int, default=40)
    parser.add_argument("--test-batches", type=int, default=40)
    parser.add_argument("--save-model", default="sequence_pitch_model.keras")
    parser.add_argument("--phrase-seconds", type=float, default=3.0)
    parser.add_argument("--architecture", choices=["cnn", "cnn_gru"], default="cnn")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--write-previews", type=int, default=0)
    parser.add_argument("--preview-dir", default="sequence_previews")
    args = parser.parse_args()

    soundfont_path = get_soundfont_path()
    if soundfont_path is None:
        print("No SoundFont/native FluidSynth found; using internal synth fallback.")

    if args.write_previews:
        write_previews(args.preview_dir, args.write_previews, soundfont_path, args.phrase_seconds)
        if args.preview_only:
            return

    train = GeneratedSequenceData(args.train_batches, args.batch_size, soundfont_path, seed=10, phrase_seconds=args.phrase_seconds, augment=True)
    val = GeneratedSequenceData(args.val_batches, args.batch_size, soundfont_path, seed=20, phrase_seconds=args.phrase_seconds, augment=True)
    test = GeneratedSequenceData(args.test_batches, args.batch_size, soundfont_path, seed=30, phrase_seconds=args.phrase_seconds, augment=False)

    X0, y0 = train[0]
    print("input shape:", X0.shape, "label shape:", y0.shape, "classes:", NUM_CLASSES)
    model = build_model(X0.shape[1:], architecture=args.architecture)
    model.summary()
    model.fit(train, validation_data=val, epochs=args.epochs)
    print("test:", model.evaluate(test, verbose=0))
    model.save(args.save_model)
    print("saved:", args.save_model)


if __name__ == "__main__":
    main()

