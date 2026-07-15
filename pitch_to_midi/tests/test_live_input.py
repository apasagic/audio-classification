"""Regression tests for microphone selection and neural silence gating."""

import numpy as np

from live_piano_roll import NeuralPitchDetector, choose_default_input, transcribe_recorded_audio


def test_os_default_input_is_used_when_available():
    devices = [("6: Stereo Mix", 6), ("8: Microphone USB", 8)]

    assert choose_default_input(devices, default_index=8) == "8: Microphone USB"


def test_physical_microphone_is_preferred_without_os_default():
    devices = [("6: Stereomix", 6), ("7: Eingang", 7), ("8: Mikrofon USB", 8)]

    assert choose_default_input(devices, default_index=-1) == "8: Mikrofon USB"


def test_quiet_current_chunk_does_not_invoke_neural_model():
    class ModelMustNotRun:
        def predict(self, *_args, **_kwargs):
            raise AssertionError("neural model should be gated before inference")

    detector = NeuralPitchDetector.__new__(NeuralPitchDetector)
    detector.model = ModelMustNotRun()
    audio_with_old_signal = np.ones(16_000, dtype=np.float32) * 0.1

    assert detector.predict(audio_with_old_signal, quiet_rms=0.001, current_rms=0.0) is None

def test_recorded_silence_rejects_even_confident_note_predictions():
    class AlwaysPredictNote:
        def predict(self, features, verbose=0):
            frame_count = features.shape[2]
            probabilities = np.zeros((1, frame_count, 38), dtype=np.float32)
            probabilities[..., 1] = 1.0
            return probabilities

    events, metrics = transcribe_recorded_audio(
        AlwaysPredictNote(), np.zeros(16_000, dtype=np.float32), quiet_rms=0.0005,
    )

    assert events == []
    assert metrics["accepted_frames"] == 0
