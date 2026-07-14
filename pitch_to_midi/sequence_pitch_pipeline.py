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
from tinysol_renderer import load_nsynth_index, load_tinysol_index, render_tinysol_sequence


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


def audio_to_features(audio, feature_type="cqt"):
    if feature_type == "cqt":
        # Two bins per semitone make pitch intervals uniform across octaves.
        spec = librosa.cqt(
            audio,
            sr=SR,
            hop_length=HOP_LENGTH,
            fmin=librosa.midi_to_hz(MIDI_NOTES[0] - 6),
            n_bins=98,
            bins_per_octave=24,
        )
    else:
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


def audio_to_model_input(audio, feature_type="cqt"):
    """Return either a learned raw-waveform input or a fixed audio feature map."""
    if feature_type == "raw":
        return audio[:, None].astype(np.float32)
    return audio_to_features(audio, feature_type)


class GeneratedSequenceData(keras.utils.Sequence):
    def __init__(self, batches, batch_size, soundfont_path, seed, phrase_seconds, augment=True, silence_weight=1.0, transition_weight=1.0, transition_radius=2, feature_type="cqt", sample_index=None, real_probability=0.0, **kwargs):
        super().__init__(**kwargs)
        self.batches = batches
        self.batch_size = batch_size
        self.soundfont_path = soundfont_path
        self.phrase_seconds = phrase_seconds
        self.rng = np.random.default_rng(seed)
        self.augment = augment
        self.silence_weight = silence_weight
        self.transition_weight = transition_weight
        self.transition_radius = transition_radius
        self.feature_type = feature_type
        self.sample_index = sample_index
        self.real_probability = real_probability

    def __len__(self):
        return self.batches

    def __getitem__(self, _):
        X, y = [], []
        for _ in range(self.batch_size):
            if self.sample_index and self.rng.random() < self.real_probability:
                audio, events = render_tinysol_sequence(
                    self.sample_index, self.rng, self.phrase_seconds, SR,
                )
            else:
                audio, events = render_sequence(self.soundfont_path, self.rng, self.phrase_seconds)
            if self.augment:
                audio = augment_phrase(audio, self.rng)
            model_input = audio_to_model_input(audio, self.feature_type)
            num_frames = int(np.ceil(len(audio) / HOP_LENGTH)) if self.feature_type == "raw" else model_input.shape[1]
            X.append(model_input)
            y.append(frame_labels(num_frames, events))
        X, y = np.array(X), np.array(y)
        if self.silence_weight != 1.0 or self.transition_weight != 1.0:
            weights = np.where(y == SILENCE_CLASS, self.silence_weight, 1.0).astype(np.float32)
            if self.transition_weight != 1.0:
                for row in range(len(y)):
                    transitions = np.flatnonzero(y[row, 1:] != y[row, :-1]) + 1
                    for frame in transitions:
                        start = max(0, frame - self.transition_radius)
                        stop = min(y.shape[1], frame + self.transition_radius + 1)
                        weights[row, start:stop] *= self.transition_weight
            return X, y, weights
        return X, y


def build_model(
    input_shape,
    architecture="cnn",
    frequency_features="flatten",
    learning_rate=1e-3,
    dropout=0.20,
):
    """Frequency-compressing CNN that preserves time frames."""
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPooling2D(pool_size=(2, 1))(x)
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 1))(x)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 1))(x)
    x = keras.layers.Permute((2, 1, 3))(x)      # time, freq, channels
    if frequency_features == "average":
        # Small and fast, but averaging removes absolute frequency position.
        x = keras.layers.TimeDistributed(keras.layers.GlobalAveragePooling1D())(x)
    else:
        # Keep frequency-bin position: pitch depends on where energy occurs.
        x = keras.layers.TimeDistributed(keras.layers.Flatten())(x)
        x = keras.layers.TimeDistributed(keras.layers.Dense(128, activation="relu"))(x)
    if architecture == "cnn_gru":
        x = keras.layers.Bidirectional(keras.layers.GRU(48, return_sequences=True))(x)
    else:
        x = keras.layers.Conv1D(96, 5, padding="same", activation="relu")(x)
        x = keras.layers.Conv1D(96, 5, padding="same", activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_raw_model(learning_rate=1e-3, dropout=0.20, architecture="raw_tcn"):
    """Fully neural, length-agnostic waveform-to-frame-label transcriber."""
    inputs = keras.layers.Input(shape=(None, 1), name="waveform")
    x = inputs
    # 2**7 = 128 samples per output frame: exactly HOP_LENGTH (8 ms).
    for filters, kernel in [(24, 15), (32, 11), (48, 9), (64, 7), (80, 5), (96, 5), (128, 3)]:
        x = keras.layers.Conv1D(filters, kernel, strides=2, padding="same", use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("gelu")(x)
    for dilation in (1, 2, 4, 8):
        residual = x
        x = keras.layers.Conv1D(128, 5, padding="same", dilation_rate=dilation, activation="gelu")(x)
        x = keras.layers.Dropout(dropout)(x)
        x = keras.layers.Add()([x, residual])
    if architecture == "raw_gru":
        x = keras.layers.Bidirectional(keras.layers.GRU(64, return_sequences=True))(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax", name="frame_labels")(x)
    model = keras.Model(inputs, outputs, name=architecture)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def class_name(class_index):
    midi = class_to_midi(int(class_index))
    return "silence" if midi is None else f"midi_{midi}"


def compact_segments(classes, max_segments=16):
    """Compress frame predictions into readable consecutive label runs."""
    classes = np.asarray(classes, dtype=np.int32)
    boundaries = np.flatnonzero(np.r_[True, classes[1:] != classes[:-1], True])
    segments = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        segments.append({
            "label": class_name(classes[start]),
            "start_seconds": round(float(start * HOP_LENGTH / SR), 3),
            "end_seconds": round(float(stop * HOP_LENGTH / SR), 3),
        })
    return segments[:max_segments]


def boundary_errors_seconds(truth, prediction, kind):
    """Nearest same-kind boundary error for one phrase."""
    truth = np.asarray(truth)
    prediction = np.asarray(prediction)
    if kind == "onset":
        true_positions = np.flatnonzero((truth[:-1] == SILENCE_CLASS) & (truth[1:] != SILENCE_CLASS)) + 1
        predicted_positions = np.flatnonzero((prediction[:-1] == SILENCE_CLASS) & (prediction[1:] != SILENCE_CLASS)) + 1
    else:
        true_positions = np.flatnonzero((truth[:-1] != SILENCE_CLASS) & (truth[1:] == SILENCE_CLASS)) + 1
        predicted_positions = np.flatnonzero((prediction[:-1] != SILENCE_CLASS) & (prediction[1:] == SILENCE_CLASS)) + 1
    if not len(true_positions):
        return []
    if not len(predicted_positions):
        return [len(truth) * HOP_LENGTH / SR] * len(true_positions)
    return [
        float(np.min(np.abs(predicted_positions - position)) * HOP_LENGTH / SR)
        for position in true_positions
    ]


def diagnose_model(model, data, sample_count=3):
    true_frames, predicted_frames, samples = [], [], []
    onset_errors, offset_errors = [], []
    for batch_index in range(len(data)):
        batch = data[batch_index]
        X, y = batch[:2]
        predicted = model.predict(X, verbose=0).argmax(axis=-1)
        true_frames.append(y.reshape(-1))
        predicted_frames.append(predicted.reshape(-1))
        for row in range(len(X)):
            onset_errors.extend(boundary_errors_seconds(y[row], predicted[row], "onset"))
            offset_errors.extend(boundary_errors_seconds(y[row], predicted[row], "offset"))
            if len(samples) < sample_count:
                samples.append({
                    "truth": compact_segments(y[row]),
                    "prediction": compact_segments(predicted[row]),
                })

    truth = np.concatenate(true_frames)
    prediction = np.concatenate(predicted_frames)
    note_mask = truth != SILENCE_CLASS
    predicted_note_mask = prediction != SILENCE_CLASS
    true_midi = truth[note_mask] + MIDI_NOTES[0] - 1
    predicted_midi = prediction[note_mask] + MIDI_NOTES[0] - 1
    semitone_error = np.abs(predicted_midi - true_midi)
    semitone_error[~predicted_note_mask[note_mask]] = 999

    labels, counts = np.unique(prediction, return_counts=True)
    order = np.argsort(counts)[::-1]
    top_predictions = [
        {"label": class_name(labels[i]), "fraction": round(float(counts[i] / len(prediction)), 4)}
        for i in order[:10]
    ]
    return {
        "frame_accuracy": float(np.mean(prediction == truth)),
        "silence_fraction_truth": float(np.mean(truth == SILENCE_CLASS)),
        "silence_fraction_predicted": float(np.mean(prediction == SILENCE_CLASS)),
        "note_only_accuracy": float(np.mean(prediction[note_mask] == truth[note_mask])),
        "note_within_one_semitone": float(np.mean(semitone_error <= 1)),
        "mean_semitone_error_for_predicted_notes": float(np.mean(
            semitone_error[semitone_error < 999]
        )) if np.any(semitone_error < 999) else None,
        "unique_predicted_classes": int(len(labels)),
        "onset_mae_ms": float(np.mean(onset_errors) * 1000) if onset_errors else None,
        "offset_mae_ms": float(np.mean(offset_errors) * 1000) if offset_errors else None,
        "onsets_within_40ms": float(np.mean(np.asarray(onset_errors) <= 0.040)) if onset_errors else None,
        "offsets_within_40ms": float(np.mean(np.asarray(offset_errors) <= 0.040)) if offset_errors else None,
        "top_predictions": top_predictions,
        "samples": samples,
    }


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


def write_history_svg(history, path):
    """Write loss and accuracy curves without requiring matplotlib."""
    series = history.history
    panels = [("Loss", "loss", "val_loss"), ("Accuracy", "accuracy", "val_accuracy")]
    width, height, margin, gap = 960, 380, 55, 60
    panel_width = (width - 2 * margin - gap) / 2
    colors = {"train": "#2563eb", "validation": "#dc2626"}
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:Segoe UI,Arial;font-size:13px;fill:#111827}.title{font-size:18px;font-weight:600}.grid{stroke:#d1d5db;stroke-width:1}.axis{stroke:#111827;stroke-width:1.5}.train{stroke:#2563eb;fill:none;stroke-width:2.5}.validation{stroke:#dc2626;fill:none;stroke-width:2.5}</style>']
    for panel, (title, train_key, val_key) in enumerate(panels):
        left = margin + panel * (panel_width + gap)
        top, plot_height = 55, 255
        values = list(series.get(train_key, [])) + list(series.get(val_key, []))
        if not values:
            continue
        low = 0.0 if title == "Accuracy" else min(values)
        high = 1.0 if title == "Accuracy" else max(values)
        if high <= low:
            high = low + 1.0
        lines.append(f'<text class="title" x="{left}" y="28">{title}</text>')
        for tick in range(6):
            y = top + plot_height * tick / 5
            value = high - (high - low) * tick / 5
            lines += [f'<line class="grid" x1="{left}" y1="{y}" x2="{left + panel_width}" y2="{y}"/>',
                      f'<text x="{left - 8}" y="{y + 4}" text-anchor="end">{value:.3f}</text>']
        lines += [f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>',
                  f'<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + panel_width}" y2="{top + plot_height}"/>']
        for key, css in ((train_key, "train"), (val_key, "validation")):
            vals = series.get(key, [])
            if not vals:
                continue
            points = []
            for index, value in enumerate(vals):
                x = left + (panel_width * index / max(1, len(vals) - 1))
                y = top + plot_height * (high - value) / (high - low)
                points.append(f"{x:.1f},{y:.1f}")
            lines.append(f'<polyline class="{css}" points="{" ".join(points)}"/>')
        lines += [f'<text x="{left + panel_width / 2}" y="{top + plot_height + 38}" text-anchor="middle">Epoch</text>',
                  f'<line class="train" x1="{left}" y1="350" x2="{left + 24}" y2="350"/><text x="{left + 30}" y="354">training</text>',
                  f'<line class="validation" x1="{left + 115}" y1="350" x2="{left + 139}" y2="350"/><text x="{left + 145}" y="354">validation</text>']
    lines.append('</svg>')
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print("history plot:", output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-batches", type=int, default=200)
    parser.add_argument("--val-batches", type=int, default=40)
    parser.add_argument("--test-batches", type=int, default=40)
    parser.add_argument("--save-model", default="sequence_pitch_model.keras")
    parser.add_argument("--phrase-seconds", type=float, default=3.0)
    parser.add_argument("--architecture", choices=["cnn", "cnn_gru", "raw_tcn", "raw_gru"], default="cnn")
    parser.add_argument("--frequency-features", choices=["flatten", "average"], default="flatten")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--silence-weight", type=float, default=1.0)
    parser.add_argument("--transition-weight", type=float, default=1.0)
    parser.add_argument("--transition-radius", type=int, default=2)
    parser.add_argument("--feature-type", choices=["cqt", "stft", "raw"], default="cqt")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--reduce-lr-patience", type=int, default=0)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--checkpoint-model")
    parser.add_argument("--csv-log")
    parser.add_argument("--history-svg")
    parser.add_argument("--sample-predictions", type=int, default=3)
    parser.add_argument("--results-json")
    parser.add_argument("--tinysol-dir")
    parser.add_argument("--nsynth-dir")
    parser.add_argument("--real-sample-probability", type=float, default=0.0)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--write-previews", type=int, default=0)
    parser.add_argument("--preview-dir", default="sequence_previews")
    args = parser.parse_args()

    if args.architecture.startswith("raw_") and args.feature_type != "raw":
        parser.error("raw_tcn/raw_gru require --feature-type raw")
    if args.feature_type == "raw" and not args.architecture.startswith("raw_"):
        parser.error("--feature-type raw requires --architecture raw_tcn or raw_gru")

    soundfont_path = get_soundfont_path()
    train_sample_index = val_sample_index = test_sample_index = None
    nsynth_sample_index = None
    if args.tinysol_dir:
        train_sample_index = load_tinysol_index(args.tinysol_dir, folds=[3, 4, 5])
        val_sample_index = load_tinysol_index(args.tinysol_dir, folds=[2])
        test_sample_index = load_tinysol_index(args.tinysol_dir, folds=[1])
        print("TinySOL pitches:", len(train_sample_index), len(val_sample_index), len(test_sample_index))
    if args.nsynth_dir:
        nsynth_sample_index = load_nsynth_index(args.nsynth_dir)
        print("NSynth pitches:", len(nsynth_sample_index))
    if soundfont_path is None:
        print("No SoundFont/native FluidSynth found; using internal synth fallback.")

    if args.write_previews:
        write_previews(args.preview_dir, args.write_previews, soundfont_path, args.phrase_seconds)
        if args.preview_only:
            return

    common = {
        "batch_size": args.batch_size,
        "soundfont_path": soundfont_path,
        "phrase_seconds": args.phrase_seconds,
        "feature_type": args.feature_type,
    }
    train = GeneratedSequenceData(
        args.train_batches, seed=10, augment=args.augment,
        silence_weight=args.silence_weight,
        transition_weight=args.transition_weight,
        transition_radius=args.transition_radius,
        sample_index=train_sample_index,
        real_probability=args.real_sample_probability, **common,
    )
    # Validation and test stay fixed and clean so experiments are comparable.
    val = GeneratedSequenceData(
        args.val_batches, seed=20, augment=False, sample_index=val_sample_index,
        real_probability=1.0 if val_sample_index else 0.0, **common,
    )
    test = GeneratedSequenceData(
        args.test_batches, seed=30, augment=False, sample_index=test_sample_index,
        real_probability=1.0 if test_sample_index else 0.0, **common,
    )

    first_batch = train[0]
    X0, y0 = first_batch[:2]
    print("input shape:", X0.shape, "label shape:", y0.shape, "classes:", NUM_CLASSES)
    if args.feature_type == "raw":
        model = build_raw_model(architecture=args.architecture, learning_rate=args.learning_rate, dropout=args.dropout)
    else:
        # Time is dynamic, so one saved model can transcribe arbitrary durations.
        model = build_model(
            (X0.shape[1], None, X0.shape[3]),
            architecture=args.architecture,
            frequency_features=args.frequency_features,
            learning_rate=args.learning_rate,
            dropout=args.dropout,
        )
    model.summary()
    callbacks = []
    if args.checkpoint_model:
        checkpoint_path = Path(args.checkpoint_model)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1,
        ))
    if args.patience:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True, verbose=1,
        ))
    if args.reduce_lr_patience:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=args.reduce_lr_patience,
            min_lr=args.min_learning_rate, verbose=1,
        ))
    if args.csv_log:
        csv_path = Path(args.csv_log)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(keras.callbacks.CSVLogger(csv_path))
    history = model.fit(train, validation_data=val, epochs=args.epochs, callbacks=callbacks)
    if args.history_svg:
        write_history_svg(history, args.history_svg)
    test_result = model.evaluate(test, verbose=0)
    print("test:", test_result)

    diagnostic_data = GeneratedSequenceData(
        args.test_batches, seed=300, augment=False, sample_index=test_sample_index,
        real_probability=1.0 if test_sample_index else 0.0, **common,
    )
    diagnostics = diagnose_model(model, diagnostic_data, args.sample_predictions)
    print("diagnostics:", json.dumps(diagnostics, indent=2))

    nsynth_test_result = nsynth_diagnostics = None
    if nsynth_sample_index:
        nsynth_data = GeneratedSequenceData(
            args.test_batches, seed=400, augment=False, sample_index=nsynth_sample_index,
            real_probability=1.0, **common,
        )
        nsynth_test_result = model.evaluate(nsynth_data, verbose=0)
        nsynth_diagnostics = diagnose_model(model, nsynth_data, args.sample_predictions)
        print("nsynth_test:", nsynth_test_result)
        print("nsynth_diagnostics:", json.dumps(nsynth_diagnostics, indent=2))

    result = {
        "config": vars(args),
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
        "test": [float(value) for value in test_result],
        "diagnostics": diagnostics,
        "nsynth_test": [float(value) for value in nsynth_test_result] if nsynth_test_result else None,
        "nsynth_diagnostics": nsynth_diagnostics,
    }
    if args.results_json:
        result_path = Path(args.results_json)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("results:", args.results_json)
    if not args.no_save:
        model.save(args.save_model)
        print("saved:", args.save_model)


if __name__ == "__main__":
    main()

