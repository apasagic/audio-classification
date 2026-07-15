"""Fast regression tests for deterministic pitch-to-MIDI transformations."""

import numpy as np

from midi_renderer import fix_length, midi_to_hz
from sequence_pitch_pipeline import (
    HOP_LENGTH,
    SILENCE_CLASS,
    SR,
    audio_to_features,
    boundary_errors_seconds,
    class_to_midi,
    compact_segments,
    frame_labels,
    midi_to_class,
    normalize,
)


def test_midi_class_mapping_round_trip():
    for midi_note in (36, 48, 60, 72):
        assert class_to_midi(midi_to_class(midi_note)) == midi_note
    assert class_to_midi(SILENCE_CLASS) is None


def test_a4_converts_to_440_hz():
    assert midi_to_hz(69) == 440.0


def test_normalize_handles_silence_and_scales_peak():
    silence = normalize(np.zeros(32, dtype=np.float32))
    audio = normalize(np.array([-0.25, 0.5], dtype=np.float32))

    assert np.all(np.isfinite(silence))
    assert np.all(silence == 0)
    assert np.isclose(np.max(np.abs(audio)), 1.0)
    assert audio.dtype == np.float32


def test_silence_and_idle_noise_create_zero_feature_maps():
    rng = np.random.default_rng(42)
    silence = np.zeros(SR, dtype=np.float32)
    idle_noise = rng.normal(0, 0.000111, SR).astype(np.float32)

    assert np.all(audio_to_features(silence, "cqt") == 0)
    assert np.all(audio_to_features(idle_noise, "cqt") == 0)

def test_fix_length_pads_and_trims():
    short = fix_length(np.ones(4), sr=10, duration=1.0)
    long = fix_length(np.arange(20), sr=10, duration=1.0)

    assert len(short) == 10
    assert np.all(short[4:] == 0)
    assert len(long) == 10
    assert np.isclose(np.max(np.abs(long)), 1.0)


def test_frame_labels_marks_only_event_frames():
    events = [{"start": 0.016, "end": 0.040, "midi": 60}]
    labels = frame_labels(8, events)

    assert labels.tolist() == [0, 0, midi_to_class(60), midi_to_class(60), midi_to_class(60), 0, 0, 0]


def test_compact_segments_preserves_boundaries():
    segments = compact_segments([0, 0, midi_to_class(60), midi_to_class(60), 0])

    assert [segment["label"] for segment in segments] == ["silence", "midi_60", "silence"]
    assert segments[1]["start_seconds"] == 2 * HOP_LENGTH / SR
    assert segments[1]["end_seconds"] == 4 * HOP_LENGTH / SR


def test_boundary_error_is_measured_in_seconds():
    truth = [0, 0, 1, 1, 0]
    prediction = [0, 0, 0, 1, 0]

    assert boundary_errors_seconds(truth, prediction, "onset") == [HOP_LENGTH / SR]
    assert boundary_errors_seconds(truth, prediction, "offset") == [0.0]
