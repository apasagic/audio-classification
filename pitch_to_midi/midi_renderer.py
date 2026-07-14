"""
MIDI-note rendering helpers.

Preferred renderer:
- pretty_midi + FluidSynth + .sf2 SoundFont.

Fallback renderer:
- a tiny internal harmonic synth.

Both render from symbolic MIDI notes to audio in memory, so no giant audio dataset is stored.
"""

import os
from functools import lru_cache

import numpy as np


DEFAULT_PROGRAMS = [0, 24, 40, 56, 73]
_DLL_DIRECTORY_HANDLE = None


def midi_to_hz(midi_note):
    """Convert a MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi_note - 69) / 12))


def render_note(midi_note, sr, duration, soundfont_path=None):
    """Render one note, using SoundFont if possible, otherwise fallback synth."""
    if soundfont_path is not None:
        try:
            return render_note_with_soundfont(midi_note, sr, duration, soundfont_path)
        except ImportError as err:
            print(f"FluidSynth unavailable, using fallback synth: {err}")

    return render_note_with_internal_synth(midi_note, sr, duration)


def render_note_with_soundfont(midi_note, sr, duration, soundfont_path, programs=None):
    """Render a cached SoundFont note while preserving program diversity."""
    global _DLL_DIRECTORY_HANDLE
    dll_dir = os.environ.get("FLUIDSYNTH_DIR")
    if dll_dir and hasattr(os, "add_dll_directory") and _DLL_DIRECTORY_HANDLE is None:
        _DLL_DIRECTORY_HANDLE = os.add_dll_directory(dll_dir)
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")

    programs = programs or DEFAULT_PROGRAMS
    program = int(np.random.choice(programs))
    # FluidSynth startup dominates training. Cache one long sample for every
    # pitch/program pair (37 x 5), then create duration variation by cropping.
    cached = _render_soundfont_cached(int(midi_note), int(sr), str(soundfont_path), program)
    result = fix_length(cached, sr, duration).copy()
    release = min(len(result), max(1, int(sr * 0.05)))
    result[-release:] *= np.linspace(1.0, 0.0, release, dtype=np.float32)
    return fix_length(result, sr, duration)


@lru_cache(maxsize=256)
def _render_soundfont_cached(midi_note, sr, soundfont_path, program):
    import pretty_midi

    bank_duration = 1.20
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    instrument = pretty_midi.Instrument(program=program)
    instrument.notes.append(pretty_midi.Note(
        velocity=100, pitch=midi_note, start=0.05, end=1.10,
    ))
    midi.instruments.append(instrument)
    audio = midi.fluidsynth(fs=sr, sf2_path=soundfont_path)
    result = fix_length(audio, sr, bank_duration)
    result.setflags(write=False)
    return result

def render_note_with_internal_synth(midi_note, sr, duration):
    """A simple non-sine synth for testing the ML pipeline without FluidSynth."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freq = midi_to_hz(midi_note)

    # Random harmonic balance gives simple timbre variation.
    audio = np.zeros_like(t)
    for harmonic in range(1, 7):
        level = np.random.uniform(0.05, 1.0) / harmonic
        phase = np.random.uniform(0, 2 * np.pi)
        audio += level * np.sin(2 * np.pi * freq * harmonic * t + phase)

    # A little square-ish color, still tied to the same fundamental pitch.
    if np.random.rand() < 0.5:
        audio += 0.15 * np.sign(np.sin(2 * np.pi * freq * t))

    attack = max(1, int(sr * np.random.uniform(0.005, 0.05)))
    release = max(1, int(sr * np.random.uniform(0.05, 0.20)))
    envelope = np.ones_like(audio)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-release:] = np.linspace(1, 0, release)

    return fix_length(audio * envelope, sr, duration)


def fix_length(audio, sr, duration):
    """Pad/trim and normalize audio to a fixed length."""
    target_len = int(sr * duration)
    audio = np.asarray(audio, dtype=np.float32)

    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    return (audio / (np.max(np.abs(audio)) + 1e-8)).astype(np.float32)


def get_soundfont_path():
    """Read the SoundFont path from an environment variable, if available."""
    path = os.environ.get("PITCH_TO_MIDI_SF2", "")
    return path if path and os.path.exists(path) else None
