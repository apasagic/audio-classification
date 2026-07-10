"""
Pitch-to-MIDI classification skeleton using generated audio batches.

Training flow:
1. Pick random MIDI-note labels.
2. Render those labels to audio in memory using a SoundFont synth.
3. Apply label-preserving audio augmentation.
4. Convert audio to spectrogram features.
5. Feed each fresh batch into a tiny neural network.

No generated audio files are saved. No giant dataset is stored.
"""

import numpy as np
import librosa
from tensorflow import keras

from augment_audio import random_augment
from midi_renderer import get_soundfont_path, midi_to_hz, render_note


SR = 16_000
DURATION = 0.75
N_FFT = 1024
HOP_LENGTH = 256
MIDI_NOTES = np.arange(48, 73)  # C3 to C5, inclusive
CQT_BINS = len(MIDI_NOTES)


class PitchBatchGenerator(keras.utils.Sequence):
    """Generate a fresh training batch every time Keras asks for one."""

    def __init__(self, batches_per_epoch, batch_size, soundfont_path, **kwargs):
        super().__init__(**kwargs)
        self.batches_per_epoch = batches_per_epoch
        self.batch_size = batch_size
        self.soundfont_path = soundfont_path

    def __len__(self):
        """Number of batches Keras will use per epoch."""
        return self.batches_per_epoch

    def __getitem__(self, batch_index):
        """Build one batch in RAM: audio -> features -> labels."""
        X = []
        y = []

        for _ in range(self.batch_size):
            midi_note = int(np.random.choice(MIDI_NOTES))
            audio = render_note(midi_note, SR, DURATION, self.soundfont_path)
            audio = random_augment(audio)

            X.append(audio_to_features(audio))
            y.append(note_to_class_index(midi_note))

        X = np.array(X, dtype=np.float32)
        y = keras.utils.to_categorical(y, num_classes=len(MIDI_NOTES))
        return X, y


def note_to_class_index(midi_note):
    """Map MIDI note 48..72 to class index 0..24."""
    return int(np.where(MIDI_NOTES == midi_note)[0][0])


def audio_to_features(audio):
    """Convert audio to pitch-aligned CQT features.

    CQT bins are spaced like musical notes, so this is much easier for a
    first pitch classifier than raw FFT bins.
    """
    cqt = librosa.cqt(
        y=audio,
        sr=SR,
        hop_length=HOP_LENGTH,
        fmin=midi_to_hz(MIDI_NOTES[0]),
        n_bins=CQT_BINS,
        bins_per_octave=12,
    )
    db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    db = (db + 80.0) / 80.0
    return db.T.astype(np.float32)


def build_model(input_shape):
    """Tiny classifier: spectrogram in, MIDI-note class probabilities out."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.Conv1D(32, kernel_size=5, padding="same", activation="relu"),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(len(MIDI_NOTES), activation="softmax"),
        ]
    )

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    soundfont_path = get_soundfont_path()
    if soundfont_path is None:
        print('No usable SoundFont/FluidSynth found; using internal harmonic synth fallback.')

    train_data = PitchBatchGenerator(batches_per_epoch=20, batch_size=16, soundfont_path=soundfont_path)
    val_data = PitchBatchGenerator(batches_per_epoch=4, batch_size=16, soundfont_path=soundfont_path)

    # Ask one generated batch for its shape, then build the model around it.
    X_first, _ = train_data[0]
    model = build_model(input_shape=X_first.shape[1:])

    model.fit(train_data, validation_data=val_data, epochs=8)

    X_demo, y_demo = val_data[0]
    predictions = model.predict(X_demo[:8])

    true_notes = MIDI_NOTES[np.argmax(y_demo[:8], axis=1)]
    predicted_notes = MIDI_NOTES[np.argmax(predictions, axis=1)]

    for true_note, predicted_note in zip(true_notes, predicted_notes):
        print(f"true MIDI: {true_note:2d}  predicted MIDI: {predicted_note:2d}")


if __name__ == "__main__":
    main()



