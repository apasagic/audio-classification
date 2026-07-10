"""
Live microphone piano-roll view with record/stop/replay and tweakable grouping.

The sliders are intentionally exposed because pitch tracking is sensitive to mic,
room, voice/instrument, and note range. Tune until the piano roll behaves.
"""

import json
import queue
import faulthandler
import logging
import sys
from pathlib import Path
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk

import librosa
import numpy as np
import sounddevice as sd


SR = 16_000
CHUNK_SECONDS = 0.05
ANALYSIS_SECONDS = 0.35
YIN_FRAME_LENGTH = 2048
MIN_HZ = 60
MAX_HZ = 1_050
MIN_MIDI = 36             # C2
MAX_MIDI = 72             # C5
WINDOW_SECONDS = 8.0

GRID_W = 980
GRID_H = 520
LEFT_W = 64
TOP_H = 24
ROW_H = GRID_H / (MAX_MIDI - MIN_MIDI + 1)

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
TIMBRES = ["Piano", "Violin", "Flute", "Synth"]

LOG_PATH = Path(__file__).with_name("live_piano_roll.log")
DEFAULT_SESSION_PATH = Path(__file__).with_name("last_piano_roll_session.json")


def setup_logging():
    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s [%(threadName)s] %(message)s",
    )
    sys.stderr = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr
    faulthandler.enable(file=sys.stderr, all_threads=True)
    logging.info("Starting live_piano_roll")


def list_input_devices():
    devices = []
    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] > 0:
            devices.append((f"{index}: {device['name']}", index))
    return devices



def list_output_devices():
    devices = []
    for index, device in enumerate(sd.query_devices()):
        if device["max_output_channels"] > 0:
            devices.append((f"{index}: {device['name']}", index))
    return devices

def hz_to_midi(hz):
    return int(round(69 + 12 * np.log2(hz / 440.0)))


def midi_to_hz(midi_note):
    return 440.0 * (2.0 ** ((midi_note - 69) / 12))


def midi_to_name(midi_note):
    return f"{NOTE_NAMES[midi_note % 12]}{midi_note // 12 - 1}"


def estimate_pitch(audio, quiet_rms):
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = np.clip(audio, -1.0, 1.0)
    rms = float(np.sqrt(np.mean(audio**2)))
    if rms < quiet_rms:
        return None

    pitches = librosa.yin(
        audio,
        fmin=MIN_HZ,
        fmax=MAX_HZ,
        sr=SR,
        frame_length=YIN_FRAME_LENGTH,
        hop_length=max(128, YIN_FRAME_LENGTH // 4),
    )
    hz = float(np.median(pitches))
    if not np.isfinite(hz):
        return None

    midi_note = hz_to_midi(hz)
    if midi_note < MIN_MIDI or midi_note > MAX_MIDI:
        return None

    return hz, midi_note, rms


def group_events(events, max_gap, min_note_seconds, pitch_tolerance, min_change_seconds, include_active=True):
    """Turn tiny detector chunks into longer note blocks.

    Output notes are: (start, end, midi_note, velocity)
    `include_active=True` draws the currently-forming note immediately.
    """
    if not events:
        return []

    events = sorted(events, key=lambda event: event[0])
    notes = []
    start, last_t, midi_note, _, peak_rms = events[0]
    cluster_midis = [midi_note]
    pending = []

    def stable_midi():
        return int(round(float(np.median(cluster_midis))))

    def append_note(end_time, force=False):
        duration = end_time - start
        if force or duration >= min_note_seconds:
            notes.append((start, end_time, stable_midi(), peak_rms))

    for event in events[1:]:
        t_event, next_midi, _, next_rms = event
        gap = t_event - last_t
        current_midi = stable_midi()

        if gap <= max_gap and abs(next_midi - current_midi) <= pitch_tolerance:
            cluster_midis.append(next_midi)
            peak_rms = max(peak_rms, next_rms)
            last_t = t_event
            pending.clear()
            continue

        pending.append(event)
        pending_duration = pending[-1][0] - pending[0][0] + CHUNK_SECONDS
        pending_midis = [item[1] for item in pending]
        pending_midi = int(round(float(np.median(pending_midis))))

        # If the possible new pitch has not persisted yet, bridge it visually.
        if pending_duration < min_change_seconds:
            last_t = t_event
            continue

        # Commit the old note at the start of the persistent new pitch.
        append_note(pending[0][0])
        start = pending[0][0]
        last_t = pending[-1][0]
        midi_note = pending_midi
        cluster_midis = pending_midis[:]
        peak_rms = max(item[3] for item in pending)
        pending.clear()

    # Draw/keep the current note while recording, even before min duration.
    append_note(last_t + CHUNK_SECONDS, force=include_active)
    return notes



def group_connected_events(events, max_gap, min_note_seconds, pitch_tolerance=0):
    """Merge adjacent chunks, but split when pitch really changes."""
    if not events:
        return []

    events = sorted(events, key=lambda event: event[0])
    notes = []
    start = events[0][0]
    last_t = events[0][0]
    midis = [events[0][1]]
    peak_rms = events[0][3]

    def block_midi():
        return int(round(float(np.median(midis))))

    for t_event, midi_note, _, rms in events[1:]:
        gap = t_event - last_t
        same_pitch = abs(midi_note - block_midi()) <= pitch_tolerance

        if gap <= max_gap and same_pitch:
            last_t = t_event
            midis.append(midi_note)
            peak_rms = max(peak_rms, rms)
            continue

        end = last_t + CHUNK_SECONDS
        if end - start >= min_note_seconds:
            notes.append((start, end, block_midi(), peak_rms))

        start = t_event
        last_t = t_event
        midis = [midi_note]
        peak_rms = rms

    end = last_t + CHUNK_SECONDS
    if end - start >= min_note_seconds:
        notes.append((start, end, block_midi(), peak_rms))

    return notes

def synth_note(midi_note, duration, timbre):
    """Tiny built-in synth with deliberately different rough timbres."""
    n = max(1, int(SR * duration))
    t = np.linspace(0, duration, n, endpoint=False)
    freq = midi_to_hz(midi_note)
    audio = np.zeros_like(t)

    if timbre == "Piano":
        # Bright attack, fast decay, no vibrato.
        for i, level in enumerate([1.0, 0.55, 0.32, 0.18, 0.08, 0.04], start=1):
            audio += level * np.sin(2 * np.pi * freq * i * t)
        envelope = np.exp(-3.8 * t / max(duration, 1e-6))
        attack_s, release_s = 0.004, 0.08
    elif timbre == "Violin":
        # Sustained harmonics plus vibrato, closer to a bowed tone.
        vibrato = 0.007 * np.sin(2 * np.pi * 5.5 * t)
        phase = 2 * np.pi * freq * t * (1.0 + vibrato)
        for i, level in enumerate([1.0, 0.85, 0.62, 0.42, 0.30, 0.18, 0.10], start=1):
            audio += level * np.sin(i * phase)
        envelope = np.ones_like(t) * 0.92
        attack_s, release_s = 0.080, 0.16
    elif timbre == "Flute":
        # Mostly fundamental, soft breath noise, gentle vibrato.
        vibrato = 0.004 * np.sin(2 * np.pi * 4.0 * t)
        phase = 2 * np.pi * freq * t * (1.0 + vibrato)
        audio = np.sin(phase) + 0.14 * np.sin(2 * phase) + 0.04 * np.sin(3 * phase)
        rng = np.random.default_rng(midi_note + int(duration * 1000))
        audio += 0.015 * rng.normal(0, 1, size=len(t))
        envelope = np.ones_like(t) * 0.85
        attack_s, release_s = 0.050, 0.10
    else:
        # Obvious synth: saw-ish plus a square layer.
        phase = (freq * t) % 1.0
        saw = 2.0 * phase - 1.0
        square = np.sign(np.sin(2 * np.pi * freq * t))
        audio = 0.72 * saw + 0.28 * square
        envelope = np.ones_like(t) * 0.95
        attack_s, release_s = 0.006, 0.05

    attack = max(1, int(SR * min(attack_s, duration / 2)))
    release = max(1, int(SR * min(release_s, duration / 2)))
    envelope = envelope.copy()
    envelope[:attack] *= np.linspace(0, 1, attack)
    envelope[-release:] *= np.linspace(1, 0, release)
    audio *= envelope
    return (audio / (np.max(np.abs(audio)) + 1e-8)).astype(np.float32)

def scale_notes(notes, speed):
    speed = max(0.25, float(speed))
    return [(s / speed, e / speed, m, v) for s, e, m, v in notes]


def render_notes(notes, timbre):
    if not notes:
        return np.zeros(int(SR * 0.25), dtype=np.float32)
    last_time = max(end for _, end, _, _ in notes)
    audio = np.zeros(int(SR * (last_time + 0.35)), dtype=np.float32)
    for start_t, end_t, midi_note, velocity in notes:
        duration = max(0.05, end_t - start_t)
        start = int(SR * start_t)
        level = float(np.clip(velocity / 0.08, 0.20, 0.70))
        note = synth_note(midi_note, duration, timbre) * level
        end = min(start + len(note), len(audio))
        audio[start:end] += note[: end - start]
    return (audio / (np.max(np.abs(audio)) + 1e-8)).astype(np.float32)


class PianoRollApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Live Pitch to MIDI - Piano Roll")
        self.root.geometry(f"{GRID_W + LEFT_W + 24}x{GRID_H + TOP_H + 176}")

        self.input_devices = list_input_devices()
        default_device = next(
            (label for label, index in self.input_devices if index == 5),
            self.input_devices[0][0] if self.input_devices else "No input devices",
        )
        self.input_device = tk.StringVar(value=default_device)
        self.selected_input_index = self.device_index_from_label(self.input_devices, default_device)

        self.output_devices = list_output_devices()
        default_output_index = sd.default.device[1] if sd.default.device[1] is not None else None
        default_output = next(
            (label for label, index in self.output_devices if index == default_output_index),
            self.output_devices[0][0] if self.output_devices else "Default output",
        )
        self.output_device = tk.StringVar(value=default_output)
        self.selected_output_index = self.device_index_from_label(self.output_devices, default_output)

        self.quiet_rms = tk.DoubleVar(value=0.0005)
        self.max_gap = tk.DoubleVar(value=0.45)
        self.min_note_seconds = tk.DoubleVar(value=0.03)
        self.pitch_tolerance = tk.IntVar(value=1)
        self.min_change_seconds = tk.DoubleVar(value=0.25)
        self.playback_bpm = tk.DoubleVar(value=120.0)
        self.fit_replay = tk.BooleanVar(value=True)
        self.show_raw = tk.BooleanVar(value=False)
        self.merge_connected = tk.BooleanVar(value=True)

        self.controls = ttk.Frame(self.root)
        self.controls.pack(fill="x", padx=10, pady=6)
        self.timbre = tk.StringVar(value="Piano")
        ttk.Button(self.controls, text="Start Recording", command=self.start_recording).pack(side="left", padx=(0, 6))
        ttk.Button(self.controls, text="Stop Recording", command=self.stop_recording).pack(side="left", padx=6)
        ttk.Button(self.controls, text="Replay", command=self.replay).pack(side="left", padx=6)
        ttk.Button(self.controls, text="Reset", command=self.reset).pack(side="left", padx=6)
        ttk.Button(self.controls, text="Save", command=self.save_session).pack(side="left", padx=6)
        ttk.Button(self.controls, text="Load", command=self.load_session).pack(side="left", padx=6)
        ttk.Label(self.controls, text="Timbre").pack(side="left", padx=(18, 6))
        ttk.OptionMenu(self.controls, self.timbre, self.timbre.get(), *TIMBRES).pack(side="left")
        ttk.Label(self.controls, text="Input").pack(side="left", padx=(18, 6))
        input_labels = [label for label, _ in self.input_devices] or ["No input devices"]
        ttk.OptionMenu(self.controls, self.input_device, self.input_device.get(), *input_labels, command=self.set_input_device).pack(side="left")
        ttk.Label(self.controls, text="Output").pack(side="left", padx=(18, 6))
        output_labels = [label for label, _ in self.output_devices] or ["Default output"]
        ttk.OptionMenu(self.controls, self.output_device, self.output_device.get(), *output_labels, command=self.set_output_device).pack(side="left")
        ttk.Checkbutton(self.controls, text="Raw pitch chunks", variable=self.show_raw).pack(side="left", padx=(18, 0))
        ttk.Checkbutton(self.controls, text="Merge connected", variable=self.merge_connected).pack(side="left", padx=(10, 0))
        ttk.Checkbutton(self.controls, text="Fit replay", variable=self.fit_replay).pack(side="left", padx=(10, 0))
        self.rms_label = ttk.Label(self.controls, text="RMS: 0.00000")
        self.rms_label.pack(side="left", padx=(18, 0))

        self.slider_value_labels = {}
        self.slider_frame = ttk.Frame(self.root)
        self.slider_frame.pack(fill="x", padx=10, pady=(0, 6))
        self.add_slider("Quiet RMS", self.quiet_rms, 0.00005, 0.03, 0)
        self.add_slider("Max gap", self.max_gap, 0.02, 0.80, 1)
        self.add_slider("Min note", self.min_note_seconds, 0.00, 0.50, 2)
        self.add_slider("Pitch tol", self.pitch_tolerance, 0, 3, 3)
        self.add_slider("Min change", self.min_change_seconds, 0.00, 0.80, 4)
        self.add_slider("Replay BPM", self.playback_bpm, 40, 240, 5)

        self.canvas = tk.Canvas(self.root, bg="#111318", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.results = queue.Queue()
        self.audio_buffer = np.zeros(int(SR * ANALYSIS_SECONDS), dtype=np.float32)
        self.running = True
        self.mode = "idle"
        self.status = "Press Start Recording. If no notes appear, try Input 5 or 20 and watch input rms."
        self.latest_rms = 0.0
        self.captured_events = []
        self.record_started_at = None
        self.replay_started_at = None
        self.replay_duration = 0.0
        self.replay_speed = 1.0
        self.playback_error = None
        self.audio_lock = threading.Lock()
        self.current_quiet_rms = self.quiet_rms.get()
        self.root.report_callback_exception = self.log_tk_exception

        self.worker = threading.Thread(target=self.audio_loop, daemon=True)
        self.worker.start()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.draw_loop()

    def add_slider(self, label, variable, from_, to, column):
        box = ttk.Frame(self.slider_frame)
        box.grid(row=0, column=column, sticky="ew", padx=6)
        self.slider_frame.columnconfigure(column, weight=1)
        header = ttk.Frame(box)
        header.pack(fill="x")
        ttk.Label(header, text=label).pack(side="left")
        value_label = ttk.Label(header, width=11, anchor="e")
        value_label.pack(side="right")
        self.slider_value_labels[label] = (value_label, variable)
        ttk.Scale(box, from_=from_, to=to, variable=variable, orient="horizontal", command=lambda _=None: self.update_slider_labels()).pack(fill="x")
        self.update_slider_labels()

    def slider_text(self, label, value):
        if label == "Quiet RMS":
            return f"{value:.5f}"
        if label == "Pitch tol":
            return f"{int(round(value))} st"
        return f"{value * 1000:.0f} ms"

    def update_slider_labels(self):
        for label, (widget, variable) in self.slider_value_labels.items():
            widget.config(text=self.slider_text(label, float(variable.get())))

    def log_tk_exception(self, exc_type, exc, tb):
        logging.exception("Tkinter callback crashed", exc_info=(exc_type, exc, tb))
        self.status = f"GUI error written to {LOG_PATH.name}: {exc}"

    def device_index_from_label(self, devices, label):
        for device_label, index in devices:
            if device_label == label:
                return index
        return None

    def stop_audio(self):
        with self.audio_lock:
            sd.stop()


    def set_input_device(self, label):
        self.selected_input_index = self.device_index_from_label(self.input_devices, label)
        self.audio_buffer[:] = 0
        self.stop_audio()
        logging.info("Input changed to %s index=%s", label, self.selected_input_index)
        self.status = f"Input set to {label}"

    def set_output_device(self, label):
        self.selected_output_index = self.device_index_from_label(self.output_devices, label)
        self.stop_audio()
        logging.info("Output changed to %s index=%s", label, self.selected_output_index)
        self.status = f"Output set to {label}"

    def input_device_index(self):
        return self.selected_input_index

    def input_sample_rate(self):
        try:
            return int(sd.query_devices(self.selected_input_index, "input")["default_samplerate"])
        except Exception:
            return SR

    def output_device_index(self):
        return self.selected_output_index

    def current_notes(self, include_active=True):
        if self.show_raw.get():
            return [(t, t + CHUNK_SECONDS, midi, rms) for t, midi, _, rms in self.captured_events]

        if self.merge_connected.get():
            return group_connected_events(
                self.captured_events,
                self.max_gap.get(),
                self.min_note_seconds.get(),
                self.pitch_tolerance.get(),
            )

        return group_events(
            self.captured_events,
            self.max_gap.get(),
            self.min_note_seconds.get(),
            self.pitch_tolerance.get(),
            self.min_change_seconds.get(),
            include_active=include_active,
        )

    def settings_payload(self):
        return {
            "quiet_rms": float(self.quiet_rms.get()),
            "max_gap": float(self.max_gap.get()),
            "min_note_seconds": float(self.min_note_seconds.get()),
            "pitch_tolerance": int(round(float(self.pitch_tolerance.get()))),
            "min_change_seconds": float(self.min_change_seconds.get()),
            "playback_bpm": float(self.playback_bpm.get()),
            "fit_replay": bool(self.fit_replay.get()),
            "show_raw": bool(self.show_raw.get()),
            "merge_connected": bool(self.merge_connected.get()),
            "timbre": self.timbre.get(),
        }

    def apply_settings(self, settings):
        self.quiet_rms.set(float(settings.get("quiet_rms", self.quiet_rms.get())))
        self.max_gap.set(float(settings.get("max_gap", self.max_gap.get())))
        self.min_note_seconds.set(float(settings.get("min_note_seconds", self.min_note_seconds.get())))
        self.pitch_tolerance.set(int(settings.get("pitch_tolerance", self.pitch_tolerance.get())))
        self.min_change_seconds.set(float(settings.get("min_change_seconds", self.min_change_seconds.get())))
        self.playback_bpm.set(float(settings.get("playback_bpm", self.playback_bpm.get())))
        self.fit_replay.set(bool(settings.get("fit_replay", self.fit_replay.get())))
        self.show_raw.set(bool(settings.get("show_raw", self.show_raw.get())))
        self.merge_connected.set(bool(settings.get("merge_connected", self.merge_connected.get())))
        if settings.get("timbre") in TIMBRES:
            self.timbre.set(settings["timbre"])
        self.update_slider_labels()

    def save_session(self):
        path = filedialog.asksaveasfilename(
            title="Save piano roll session",
            initialfile=DEFAULT_SESSION_PATH.name,
            defaultextension=".json",
            filetypes=[("Piano roll session", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        payload = {
            "version": 1,
            "settings": self.settings_payload(),
            "captured_events": self.captured_events,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.status = f"Saved {len(self.captured_events)} pitch chunks to {Path(path).name}"

    def load_session(self):
        path = filedialog.askopenfilename(
            title="Load piano roll session",
            initialfile=DEFAULT_SESSION_PATH.name,
            filetypes=[("Piano roll session", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.apply_settings(payload.get("settings", {}))
        self.captured_events = [tuple(event) for event in payload.get("captured_events", [])]
        self.record_started_at = None
        self.replay_started_at = None
        self.mode = "stopped" if self.captured_events else "idle"
        self.status = f"Loaded {len(self.captured_events)} pitch chunks from {Path(path).name}"

    def reset(self):
        """Clear the phrase and return to idle without killing the GUI."""
        self.stop_audio()
        self.captured_events.clear()
        self.audio_buffer[:] = 0
        self.record_started_at = None
        self.replay_started_at = None
        self.replay_duration = 0.0
        self.replay_speed = 1.0
        self.playback_error = None
        self.mode = "idle"
        self.status = "Reset. Press Start Recording."

    def start_recording(self):
        self.stop_audio()
        self.captured_events.clear()
        self.audio_buffer[:] = 0
        self.record_started_at = time.time()
        self.replay_started_at = None
        self.mode = "recording"
        self.playback_error = None
        self.status = f"Recording on {self.input_device.get()}... input rms will appear below"

    def stop_recording(self):
        if self.mode == "recording":
            self.mode = "stopped"
            self.status = f"Stopped. Captured {len(self.current_notes(include_active=False))} grouped notes."

    def replay(self):
        notes = self.current_notes(include_active=False)
        if not notes:
            self.status = "Nothing captured. Try Input 5 or 20, lower Quiet RMS, or enable Raw pitch chunks."
            return
        self.stop_audio()
        self.mode = "replaying"
        self.replay_started_at = time.time()
        self.replay_speed = max(0.25, float(self.playback_bpm.get()) / 120.0)
        self.replay_duration = max(end for _, end, _, _ in notes) / self.replay_speed
        self.playback_error = None
        timbre = self.timbre.get()
        output_label = self.output_device.get()
        output_index = self.output_device_index()
        playback_notes = scale_notes(notes, self.replay_speed)
        self.status = f"Replaying {len(notes)} notes as {timbre} at {self.playback_bpm.get():.0f} BPM to {output_label}"
        threading.Thread(target=self._playback_worker, args=(playback_notes, timbre, output_index), daemon=True).start()
    def _playback_worker(self, notes, timbre, output_index):
        try:
            audio = render_notes(notes, timbre)
            if np.max(np.abs(audio)) < 1e-6:
                self.playback_error = "Replay generated silence."
                return
            with self.audio_lock:
                sd.play(audio, samplerate=SR, device=output_index)
                sd.wait()
        except Exception as exc:
            self.playback_error = str(exc)

    def audio_loop(self):
        while self.running:
            try:
                input_sr = self.input_sample_rate()
                with self.audio_lock:
                    audio = sd.rec(
                        int(input_sr * CHUNK_SECONDS),
                        samplerate=input_sr,
                        channels=1,
                        dtype="float32",
                        device=self.input_device_index(),
                    )
                    sd.wait()
                chunk = audio[:, 0]
                chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
                chunk = np.clip(chunk, -1.0, 1.0)

                # Record from the selected device at its native rate, then convert
                # to the detector rate. This avoids Windows devices returning bad
                # data when forced to 16 kHz.
                if input_sr != SR:
                    chunk = librosa.resample(chunk, orig_sr=input_sr, target_sr=SR)

                self.latest_rms = float(np.sqrt(np.mean(chunk**2)))
                self.audio_buffer = np.roll(self.audio_buffer, -len(chunk))
                self.audio_buffer[-len(chunk):] = chunk
                result = estimate_pitch(self.audio_buffer, self.current_quiet_rms)
                self.results.put((time.time(), result, None))
            except Exception as exc:
                logging.exception("Audio loop error")
                self.results.put((time.time(), None, str(exc)))
                time.sleep(0.20)

    def close(self):
        self.running = False
        self.stop_audio()
        self.root.destroy()

    def plot_metrics(self):
        width = max(360, self.canvas.winfo_width())
        height = max(260, self.canvas.winfo_height())
        grid_w = max(240, width - LEFT_W - 24)
        grid_h = max(140, height - TOP_H - 56)
        row_h = grid_h / (MAX_MIDI - MIN_MIDI + 1)
        return width, height, grid_w, grid_h, row_h

    def y_for_midi(self, midi_note):
        _, _, _, _, row_h = self.plot_metrics()
        return TOP_H + (MAX_MIDI - midi_note) * row_h

    def phrase_duration(self, notes):
        if not notes:
            return 1.0
        return max(WINDOW_SECONDS, max(end for _, end, _, _ in notes))

    def x_flowing(self, t_note, playhead_time):
        _, _, grid_w, _, _ = self.plot_metrics()
        return LEFT_W + grid_w - ((playhead_time - t_note) / WINDOW_SECONDS) * grid_w

    def x_fitted(self, t_note, duration):
        _, _, grid_w, _, _ = self.plot_metrics()
        return LEFT_W + (t_note / duration) * grid_w

    def drain_results(self):
        while not self.results.empty():
            item = self.results.get_nowait()
            if len(item) == 3:
                t_abs, result, error = item
            else:
                t_abs, result = item
                error = None

            if error is not None:
                self.status = f"Audio input error: {error}"
                continue

            if self.mode != "recording":
                continue
            if result is None:
                self.status = f"Recording... no pitch. input rms={self.latest_rms:.5f}, Quiet RMS={self.quiet_rms.get():.5f}, Input={self.input_device.get()}"
                continue
            hz, midi_note, rms = result
            t_recording = t_abs - self.record_started_at
            self.captured_events.append((t_recording, midi_note, hz, rms))
            self.status = f"Recording: {hz:7.1f} Hz   MIDI {midi_note:2d}   {midi_to_name(midi_note)}   rms={rms:.4f}"

        if self.mode == "replaying" and self.replay_started_at is not None:
            elapsed = time.time() - self.replay_started_at
            if elapsed >= self.replay_duration + 0.30:
                self.mode = "stopped"
                if self.playback_error:
                    self.status = f"Replay error: {self.playback_error}"
                else:
                    self.status = "Replay finished."

    def visible_notes(self):
        include_active = self.mode == "recording"
        notes = self.current_notes(include_active=include_active)
        if self.mode == "recording" and self.record_started_at is not None:
            playhead = time.time() - self.record_started_at
            visible = [note for note in notes if 0 <= playhead - note[0] <= WINDOW_SECONDS]
            return visible, "flow", playhead
        if self.mode == "replaying" and self.replay_started_at is not None:
            source_playhead = (time.time() - self.replay_started_at) * self.replay_speed
            duration = self.phrase_duration(notes)
            if self.fit_replay.get() or duration <= WINDOW_SECONDS * 1.5:
                visible = [(s, min(e, source_playhead), m, v) for s, e, m, v in notes if s <= source_playhead]
                return visible, "fit_replay", source_playhead
            visible = [(s, min(e, source_playhead), m, v) for s, e, m, v in notes if s <= source_playhead and source_playhead - s <= WINDOW_SECONDS]
            return visible, "flow", source_playhead
        return notes, "fit", None
    def draw_grid(self):
        width, height, grid_w, grid_h, row_h = self.plot_metrics()
        self.canvas.create_rectangle(0, 0, width, height, fill="#111318", outline="")
        for midi_note in range(MIN_MIDI, MAX_MIDI + 1):
            y = self.y_for_midi(midi_note)
            color = "#30343d" if midi_note % 12 == 0 else "#22262d"
            self.canvas.create_line(LEFT_W, y, LEFT_W + grid_w, y, fill=color)
            if midi_note % 12 == 0:
                self.canvas.create_text(LEFT_W - 10, y + row_h / 2, text=midi_to_name(midi_note), fill="#c9d1d9", anchor="e", font=("Segoe UI", 9))
        for i in range(17):
            x = LEFT_W + i * grid_w / 16
            color = "#3a404a" if i % 4 == 0 else "#242932"
            self.canvas.create_line(x, TOP_H, x, TOP_H + grid_h, fill=color)
        self.canvas.create_rectangle(LEFT_W, TOP_H, LEFT_W + grid_w, TOP_H + grid_h, outline="#4b5563")
        self.canvas.create_text(LEFT_W, TOP_H + grid_h + 26, text=self.status, fill="#e5e7eb", anchor="w", font=("Segoe UI", 12))

    def draw_notes(self):
        _, _, grid_w, _, row_h = self.plot_metrics()
        notes, layout, playhead = self.visible_notes()
        all_notes = self.current_notes(include_active=True)
        duration = self.phrase_duration(all_notes)
        for start, end, midi_note, _ in notes:
            if end <= start:
                continue
            if layout == "flow":
                x = self.x_flowing(start, playhead)
                width = max(5, ((end - start) / WINDOW_SECONDS) * grid_w)
            else:
                x = self.x_fitted(start, duration)
                width = max(5, ((end - start) / duration) * grid_w)
            y = self.y_for_midi(midi_note)
            self.canvas.create_rectangle(x, y + 1, x + width, y + row_h - 1, fill="#22c55e", outline="#86efac")
        if layout == "fit_replay" and playhead is not None:
            px = self.x_fitted(min(playhead, duration), duration)
            self.canvas.create_line(px, TOP_H, px, TOP_H + self.plot_metrics()[3], fill="#38bdf8", width=2)
        if self.mode in {"recording", "replaying"}:
            label = "REC" if self.mode == "recording" else "PLAY"
            color = "#ef4444" if self.mode == "recording" else "#38bdf8"
            self.canvas.create_text(LEFT_W + grid_w - 12, TOP_H + 18, text=label, fill=color, anchor="e", font=("Segoe UI", 13, "bold"))

    def draw_loop(self):
        self.current_quiet_rms = self.quiet_rms.get()
        self.drain_results()
        self.rms_label.config(text=f"RMS: {self.latest_rms:.5f}")
        self.canvas.delete("all")
        self.draw_grid()
        self.draw_notes()
        if self.running:
            self.root.after(33, self.draw_loop)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    setup_logging()
    try:
        PianoRollApp().run()
    except Exception:
        logging.exception("Application crashed")
        raise



