# Testing and MLOps baseline

The test suite protects the deterministic code around the neural model and documents the model's expected interface. This is the first CI/CD building block: every later deployment should run the same checks from a clean checkout before an artifact is released.

## Install and run

```powershell
cd pitch_to_midi
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -m "not model"
```

Run the local model-contract smoke test when `cqt_gru_best.keras` is available:

```powershell
.\.venv\Scripts\python.exe -m pytest -m model
```

Run everything:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Why two layers?

- **Unit tests** exercise small deterministic transformations and should finish quickly on any developer machine or CI runner.
- **Model-contract tests** verify that a separately stored model accepts the expected CQT tensor and produces one normalized probability vector per time frame.
- **Training evaluation** measures model quality on fixed datasets. It is slower and belongs in a dedicated ML validation pipeline rather than in every code commit.

This distinction prevents a common MLOps failure: deploying code that passes ordinary tests but is incompatible with the selected model artifact.

## Portfolio value

The suite demonstrates test isolation, reproducible development dependencies, numerical assertions, explicit model contracts, and a path toward automated quality gates. The next step is to execute the fast layer in GitHub Actions on every push and pull request.
## Current behavioral baselines

The trained CQT + Bi-GRU is also checked on behaviors that aggregate validation accuracy can hide:

- Digital silence and Realtek-level idle noise must produce all-zero CQT feature maps.
- The trained model must classify every frame of the digital-silence feature map as silence.
- Recent microphone silence must prevent neural inference.
- Recorded-file silence must produce zero accepted note events even if a model proposes notes.
- Eight generated WAV/JSON preview pairs currently achieve 88.6% frame accuracy, 93.3% exact note-frame accuracy, and 93.6% within-one-semitone accuracy through the whole-file path.

These are regression baselines, not final production targets. Future real-microphone recordings should be added as versioned evaluation fixtures rather than relying only on generated audio.
