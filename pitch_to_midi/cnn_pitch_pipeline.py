"""
Starter CNN pitch classifier using generated audio and STFT images.

This is still a simple single-label classifier:
- one short audio example contains one active note
- the target is one MIDI note class
- no generated audio files are stored

Later, this can grow into onset/duration detection by training on longer
spectrogram windows with frame-level labels instead of one label per clip.
"""

import argparse

import librosa
import numpy as np
from tensorflow import keras

from midi_renderer import fix_length, get_soundfont_path, render_note


SR = 16_000
DURATION = 1.0
N_FFT = 1024
HOP_LENGTH = 128
MIDI_NOTES = np.arange(36, 73)  # C2 through C5


def normalize(audio):
    """Keep values stable after augmentation."""
    return (audio / (np.max(np.abs(audio)) + 1e-8)).astype(np.float32)


def add_reverb(audio, rng):
    """Tiny synthetic room tail, cheap but useful."""
    tail_len = int(SR * rng.uniform(0.03, 0.18))
    impulse = np.exp(-np.linspace(0, 5, tail_len)).astype(np.float32)
    wet = np.convolve(audio, impulse, mode="full")[: len(audio)]
    return normalize(0.85 * audio + rng.uniform(0.03, 0.18) * wet)


def augment(audio, rng):
    """Label-preserving augmentation for one-note pitch classification."""
    audio = audio * rng.uniform(0.35, 1.25)
    audio = np.roll(audio, rng.integers(-int(0.08 * SR), int(0.08 * SR) + 1))

    if rng.random() < 0.65:
        audio = audio + rng.normal(0, rng.uniform(0.002, 0.04), len(audio))
    if rng.random() < 0.35:
        audio = librosa.effects.time_stretch(audio, rate=rng.uniform(0.88, 1.12))
        audio = fix_length(audio, SR, DURATION)
    if rng.random() < 0.35:
        # Small cents-level pitch variation; stays within the same MIDI label.
        audio = librosa.effects.pitch_shift(audio, sr=SR, n_steps=rng.uniform(-0.18, 0.18))
    if rng.random() < 0.45:
        audio = add_reverb(audio, rng)

    return normalize(audio)


def audio_to_stft_image(audio):
    """Convert audio into a normalized log-STFT image: freq x time x channel."""
    spec = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH, window="hann")
    db = librosa.amplitude_to_db(np.abs(spec), ref=np.max)
    image = ((db + 80.0) / 80.0).clip(0.0, 1.0)
    return image[..., None].astype(np.float32)


def note_to_index(midi_note):
    return int(midi_note - MIDI_NOTES[0])


class GeneratedPitchData(keras.utils.Sequence):
    """Keras asks for a batch; we synthesize it in RAM on demand."""

    def __init__(self, batches, batch_size, soundfont_path, seed, augment_audio=True, **kwargs):
        super().__init__(**kwargs)
        self.batches = batches
        self.batch_size = batch_size
        self.soundfont_path = soundfont_path
        self.rng = np.random.default_rng(seed)
        self.augment_audio = augment_audio

    def __len__(self):
        return self.batches

    def __getitem__(self, _):
        X, y = [], []
        for _ in range(self.batch_size):
            midi_note = int(self.rng.choice(MIDI_NOTES))
            duration = float(self.rng.uniform(0.45, DURATION))
            audio = render_note(midi_note, SR, duration, self.soundfont_path)
            audio = fix_length(audio, SR, DURATION)
            if self.augment_audio:
                audio = augment(audio, self.rng)
            X.append(audio_to_stft_image(audio))
            y.append(note_to_index(midi_note))
        return np.array(X), keras.utils.to_categorical(y, num_classes=len(MIDI_NOTES))


def build_model(input_shape):
    """Small 2D CNN: STFT image in, one MIDI-note class out."""
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv2D(24, 3, padding="same", activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(48, 3, padding="same", activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(96, 3, padding="same", activation="relu"),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.20),
        keras.layers.Dense(96, activation="relu"),
        keras.layers.Dense(len(MIDI_NOTES), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-batches", type=int, default=50)
    parser.add_argument("--val-batches", type=int, default=10)
    parser.add_argument("--test-batches", type=int, default=10)
    args = parser.parse_args()

    soundfont_path = get_soundfont_path()
    if soundfont_path is None:
        print("No SoundFont/native FluidSynth found; using internal synth fallback.")

    train = GeneratedPitchData(args.train_batches, args.batch_size, soundfont_path, seed=1, augment_audio=True)
    val = GeneratedPitchData(args.val_batches, args.batch_size, soundfont_path, seed=2, augment_audio=True)
    test = GeneratedPitchData(args.test_batches, args.batch_size, soundfont_path, seed=3, augment_audio=False)

    X0, _ = train[0]
    model = build_model(X0.shape[1:])
    model.fit(train, validation_data=val, epochs=args.epochs)
    print("test:", model.evaluate(test, verbose=0))

    X_demo, y_demo = test[0]
    pred = model.predict(X_demo[:8], verbose=0)
    true_notes = MIDI_NOTES[np.argmax(y_demo[:8], axis=1)]
    pred_notes = MIDI_NOTES[np.argmax(pred, axis=1)]
    for true_note, pred_note in zip(true_notes, pred_notes):
        print(f"true MIDI {true_note:2d} -> predicted MIDI {pred_note:2d}")


if __name__ == "__main__":
    main()
