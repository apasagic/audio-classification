"""Local smoke test for the versioned neural-model interface."""

from pathlib import Path

import numpy as np
import pytest


MODEL_PATH = Path(__file__).resolve().parents[1] / "cqt_gru_best.keras"


@pytest.mark.model
def test_cqt_gru_model_contract():
    if not MODEL_PATH.exists():
        pytest.skip("model artifact is not available in this checkout")

    from tensorflow import keras
    from sequence_pitch_pipeline import NUM_CLASSES, SR, audio_to_model_input

    model = keras.models.load_model(MODEL_PATH)
    audio = np.zeros(SR * 4, dtype=np.float32)
    features = audio_to_model_input(audio, "cqt")
    probabilities = model.predict(features[None, ...], verbose=0)

    assert probabilities.shape[0] == 1
    assert probabilities.shape[1] == features.shape[1]
    assert probabilities.shape[2] == NUM_CLASSES
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0, atol=1e-5)
    assert np.all(probabilities.argmax(axis=-1) == 0)
