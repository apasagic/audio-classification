"""TinySOL sample-bank loading and in-memory phrase rendering."""

import csv
import json
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np


@lru_cache(maxsize=256)
def _load_note(path, sample_rate):
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    audio, _ = librosa.effects.trim(audio, top_db=45)
    peak = np.max(np.abs(audio)) + 1e-8
    return (audio / peak).astype(np.float32)


def load_tinysol_index(dataset_dir, folds, midi_min=36, midi_max=72):
    """Return {midi_pitch: [wav paths]} for selected official folds."""
    root = Path(dataset_dir)
    metadata_path = root / "TinySOL_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"TinySOL metadata not found: {metadata_path}")

    folds = {int(value) for value in folds}
    index = {}
    with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            fold = int(row["Fold"])
            midi = int(row["Pitch ID"])
            path = root / Path(row["Path"])
            if fold in folds and midi_min <= midi <= midi_max and path.exists():
                index.setdefault(midi, []).append(str(path))
    if not index:
        raise ValueError(f"No TinySOL samples found in folds {sorted(folds)}")
    return index


def load_nsynth_index(dataset_dir, midi_min=36, midi_max=72):
    """Return a pitch index for an extracted NSynth JSON/WAV split."""
    root = Path(dataset_dir)
    metadata_path = root / "examples.json"
    audio_dir = root / "audio"
    if not metadata_path.exists():
        raise FileNotFoundError(f"NSynth metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    index = {}
    for key, row in metadata.items():
        midi = int(row["pitch"])
        path = audio_dir / f"{key}.wav"
        if midi_min <= midi <= midi_max and path.exists():
            index.setdefault(midi, []).append(str(path))
    if not index:
        raise ValueError(f"No NSynth samples found in {root}")
    return index


def render_tinysol_sequence(index, rng, phrase_seconds, sample_rate):
    """Compose a monophonic phrase from labeled real instrument notes."""
    total_samples = int(sample_rate * phrase_seconds)
    phrase = np.zeros(total_samples, dtype=np.float32)
    events = []
    pitches = np.array(sorted(index), dtype=np.int32)
    time = float(rng.uniform(0.05, 0.35))

    while time < phrase_seconds - 0.20:
        midi = int(rng.choice(pitches))
        duration = float(rng.uniform(0.18, 0.85))
        end_time = min(time + duration, phrase_seconds)
        tail = float(rng.uniform(0.03, 0.16))
        render_duration = min(duration + tail, phrase_seconds - time)
        path = str(rng.choice(index[midi]))
        source = _load_note(path, sample_rate).copy()

        target_length = int(sample_rate * render_duration)
        if len(source) < target_length:
            source = np.pad(source, (0, target_length - len(source)))
        else:
            source = source[:target_length]
        fade = min(len(source), max(1, int(sample_rate * 0.025)))
        source[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        source *= float(rng.uniform(0.45, 1.0))

        start_sample = int(time * sample_rate)
        stop_sample = min(start_sample + len(source), total_samples)
        phrase[start_sample:stop_sample] += source[: stop_sample - start_sample]
        events.append({"start": time, "end": end_time, "midi": midi, "source": path})
        time = end_time + float(rng.uniform(-0.06, 0.30))

    peak = np.max(np.abs(phrase)) + 1e-8
    return (phrase / peak).astype(np.float32), events
