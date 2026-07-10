"""
MIDI-note rendering helpers.

Preferred renderer:
- pretty_midi + FluidSynth + .sf2 SoundFont.

Fallback renderer:
- a tiny internal harmonic synth.

Both render from symbolic MIDI notes to audio in memory, so no giant audio dataset is stored.
"""

import os
import numpy as np


DEFAULT_PROGRAMS = [0, 24, 40, 56, 73]


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
    """Render one MIDI note to audio using a SoundFont.

    Requires pretty_midi, pyFluidSynth, the native FluidSynth library, and a .sf2 file.
    """
    import pretty_midi

    programs = programs or DEFAULT_PROGRAMS
    program = int(np.random.choice(programs))
    velocity = int(np.random.randint(55, 115))

    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    instrument = pretty_midi.Instrument(program=program)
    instrument.notes.append(
        pretty_midi.Note(
            velocity=velocity,
            pitch=int(midi_note),
            start=0.05,
            end=float(duration),
        )
    )
    midi.instruments.append(instrument)

    audio = midi.fluidsynth(fs=sr, sf2_path=soundfont_path)
    return fix_length(audio, sr, duration)


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
