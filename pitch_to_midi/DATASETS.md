# External audio datasets

The project keeps downloaded data under `pitch_to_midi/datasets/`, which is ignored by Git.

## TinySOL

- Official record: https://zenodo.org/records/3659365
- License: CC BY 4.0
- Content: 2,478 isolated notes, 14 orchestral instruments, 44.1 kHz mono WAV.
- Metadata: exact pitch/MIDI ID, instrument, dynamics, and official five-fold split.
- Current split: folds 3-5 train, fold 2 validation, fold 1 test.
- Local directory: `datasets/tinysol`.

Citation: Cella et al., *TinySOL: an audio dataset of isolated musical notes*, 2020.

## NSynth test split

- Official page: https://magenta.withgoogle.com/datasets/nsynth
- License: CC BY 4.0
- Content used here: official 4,096-note test split, 16 kHz WAV plus JSON metadata.
- The official train, validation, and test instruments do not overlap.
- Current use: evaluation only; never included in training.
- Local directory: `datasets/nsynth_test/nsynth-test`.

Citation: Engel et al., *Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders*, 2017.

## SoundFont domain

- Renderer: official FluidSynth Windows x64 binary, https://www.fluidsynth.org/
- SoundFont: `pretty_midi` bundled `TimGM6mb.sf2`.
- Programs are randomized by `midi_renderer.py`.
- Configure Windows runs with `FLUIDSYNTH_DIR` and `PITCH_TO_MIDI_SF2`.

## Current domain protocol

Training batches can mix generated SoundFont phrases and TinySOL training folds through
`--real-sample-probability`. Validation uses TinySOL fold 2; primary test uses TinySOL fold 1;
`--nsynth-dir` adds an independent NSynth diagnostic without training on NSynth.
