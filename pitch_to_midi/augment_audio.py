"""
Basic audio augmentation for pitch classification.

These functions try to preserve the MIDI-note label.
That means: add noise, change loudness, shift in time.

Do NOT pitch-shift here unless you also change the target MIDI label.
"""

import numpy as np


def random_gain(audio, min_gain=0.4, max_gain=1.2):
    """Make the note randomly quieter or louder."""
    return audio * np.random.uniform(min_gain, max_gain)


def add_noise(audio, min_level=0.0, max_level=0.05):
    """Add white noise so the model learns not to trust tiny details."""
    noise_level = np.random.uniform(min_level, max_level)
    noise = np.random.normal(0, noise_level, len(audio))
    return audio + noise


def random_time_shift(audio, max_shift_fraction=0.08):
    """Move the note slightly earlier/later without changing pitch."""
    max_shift = int(len(audio) * max_shift_fraction)
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(audio, shift)


def soft_clip(audio, drive=1.5):
    """Add mild saturation, like a simple instrument/effect color."""
    return np.tanh(audio * drive)


def normalize(audio):
    """Keep audio safely between -1 and 1."""
    return audio / (np.max(np.abs(audio)) + 1e-8)


def random_augment(audio):
    """Apply a tiny random augmentation chain."""
    audio = random_gain(audio)
    audio = add_noise(audio)
    audio = random_time_shift(audio)

    if np.random.rand() < 0.5:
        audio = soft_clip(audio, drive=np.random.uniform(1.0, 2.0))

    return normalize(audio).astype(np.float32)
