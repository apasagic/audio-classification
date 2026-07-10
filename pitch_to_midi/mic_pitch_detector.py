"""
Simple microphone pitch detector.

This does NOT use the neural net yet.
It is a small realtime-ish baseline:
1. Record a short microphone chunk.
2. Estimate the fundamental frequency with librosa.yin().
3. Convert frequency in Hz to nearest MIDI note.
4. Print the detected note.

Later we can replace or combine this with the trained NN and show MIDI notes live.
"""

import time

import librosa
import numpy as np
import sounddevice as sd


SR = 16_000             # microphone sample rate
CHUNK_SECONDS = 0.25    # shorter = faster display, longer = more stable
MIN_HZ = 65             # about C2
MAX_HZ = 1_050          # about C6
CONFIDENCE_RMS = 0.01   # ignore very quiet input


def hz_to_midi(hz):
    """Convert frequency in Hz to nearest MIDI note number."""
    return int(round(69 + 12 * np.log2(hz / 440.0)))


def midi_to_name(midi_note):
    """Convert MIDI note number to a readable note name."""
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = midi_note // 12 - 1
    return f"{names[midi_note % 12]}{octave}"


def estimate_pitch(audio):
    """Estimate one pitch from one audio chunk."""
    rms = float(np.sqrt(np.mean(audio**2)))
    if rms < CONFIDENCE_RMS:
        return None

    # yin() returns a pitch estimate for each analysis frame.
    pitches = librosa.yin(audio, fmin=MIN_HZ, fmax=MAX_HZ, sr=SR)

    # Use the median because it is less jumpy than a single frame.
    hz = float(np.median(pitches))
    if not np.isfinite(hz):
        return None

    midi_note = hz_to_midi(hz)
    return hz, midi_note, rms


def main():
    print("Listening. Press Ctrl+C to stop.")

    try:
        while True:
            # Record one mono chunk from the default microphone.
            audio = sd.rec(int(SR * CHUNK_SECONDS), samplerate=SR, channels=1, dtype="float32")
            sd.wait()
            audio = audio[:, 0]

            result = estimate_pitch(audio)
            if result is None:
                print("quiet / no stable pitch")
            else:
                hz, midi_note, rms = result
                print(f"{hz:7.1f} Hz  MIDI {midi_note:3d}  {midi_to_name(midi_note):>3s}  rms={rms:.3f}")

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
