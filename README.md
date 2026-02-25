## Audio Classification

This repository explores audio (drum sound) classification using deep learning. The project focuses on experimenting with different model architectures and problem formulations to better understand what works (and what doesn’t) when classifying short percussive sounds such as kicks, snares, and cymbals.
The work was motivated by hands-on experimentation and discussion around the challenges of using sequence models (RNN/LSTM) for drum sound classification, and how alternative formulations can lead to more stable and interpretable results.

## 📌 Project Goals

- Learn how to preprocess audio for machine learning (spectrograms, feature extraction).
- Compare sequence-based models and non-sequence models for short audio events.
- Explore how problem formulation impacts model performance more than model complexity.
- Build a small but flexible codebase that can be extended with new models and datasets.

## 🧠 Key Insight

Although audio is inherently sequential, not all audio classification problems benefit from recurrent models. For short, transient sounds like drum hits:
Most discriminative information lies in the attack phase (first ~200–300 ms).
Treating audio as a fixed-length feature representation can outperform sequence modeling.
Simplifying the task (e.g., binary classification) can dramatically improve learning stability.

## Input data format

Data is 

<img width="1021" height="855" alt="image" src="https://github.com/user-attachments/assets/a008b2a2-7942-4e43-ae9f-9566e61b4335" />




## 🧩 Problem Formulations (Two Branches)

This repository supports two different ways of formulating the classification problem, which can be treated as conceptual “branches” of the project.

🔹 Branch 1: Multi-Class Drum Classification

   Problem statement:

   Given a short audio clip, predict which drum class it belongs to
   (e.g., kick, snare, cymbal, hi-hat).

   Characteristics:

   - Multi-class classification
   - Higher ambiguity between similar sounds
   - More sensitive to dataset imbalance and labeling noise
   - Useful for understanding model limitations

Typical label set: { kick, snare, cymbal, hihat, tom }

🔹 Branch 2: Binary / One-vs-Rest Classification

Problem statement:

Given a short audio clip, determine whether it belongs to a specific drum class or not
(e.g., “kick” vs “not kick”).

Characteristics:

Binary classification

Easier optimization and interpretation

Often better accuracy with limited data

Useful as a stepping stone before multi-class classification

Example tasks:

kick vs not-kick
snare vs not-snare
🏗 Available Model Types

The repository includes implementations and experiments with the following model families:

🔸 1. Feed-Forward (Dense) Models

Operate on fixed-size feature vectors (e.g., flattened spectrograms)

Simple baseline

Often surprisingly effective for short, percussive sounds

Use case:
Quick experiments, debugging pipelines, baseline performance.

🔸 2. Recurrent Models (RNN / LSTM)

Process audio features as temporal sequences

Designed to model time dependencies

Observations:

Harder to train reliably on short drum sounds

Sensitive to sequence length and padding

Often unnecessary when temporal structure is minimal

Use case:
Exploratory learning and understanding sequence modeling limitations.

🔸 3. Spectrogram-Based Models (CNN-style inputs)

Treat spectrograms as 2D representations

Enable spatial pattern learning in time–frequency space

Note:
While full CNN architectures may not be fully implemented yet, the codebase is structured to support them easily.

📁 Repository Structure
├── Drum_classification.ipynb   # End-to-end demo notebook
├── augmentations.py            # Audio data augmentation utilities
├── main_model_train.py         # Training entry point
├── model_prediction.py         # Inference / evaluation script
├── models.py                   # Model definitions
├── utils.py                    # Audio loading and preprocessing
└── .gitignore
🚀 Typical Workflow

Load and preprocess audio files (e.g., STFT or mel spectrograms)

Choose a problem formulation (multi-class or binary)

Select a model type

Train using main_model_train.py

Evaluate or predict using model_prediction.py

🛠 Requirements (Example)
pip install numpy librosa tensorflow scikit-learn matplotlib
📌 Lessons Learned

Problem formulation matters more than model complexity

RNNs are not always the right choice for short audio events

Feature representation is often the biggest performance driver

Starting with binary classification helps isolate issues early

🔮 Possible Extensions

Full CNN architectures for spectrogram inputs

Transformer-based audio models

Better dataset handling and evaluation metrics

Real-time audio classification

📜 License

MIT License (or update as needed)
