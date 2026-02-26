## Audio Classification

This repository explores audio (drum sound) classification using deep learning. The project focuses on experimenting with different model architectures and problem formulations to better understand what works (and what doesn’t) when classifying short percussive sounds such as kicks, snares, and cymbals.
The work was motivated by hands-on experimentation and discussion around the challenges of using sequence models (RNN/LSTM) for drum sound classification, and how alternative formulations can lead to more stable and interpretable results.

Motivation also came from personal involvment with music and sound and recurring usecase of having to quickly draft music ideas.
This was an attempt to explore how simple would it be to make simple music drafts transcribed as MIDI, by simply tapping, humming, whistling, beatboxing or simply feeding an existing recording.

Problem is multi-fauceted ofcourse, and at the moment, main focus is on drum detection/classification, but further efforts would be directed into note transcription and instrument categorization for other instruments.

## 📌 Repository Idea

Purpose of this project was following:

- Utilize ML algorithms to detect and categorize the events of drum hits (Kick, Snare, HiHat etc)
- Test performance of different types of NN such as simple FF, temporal CNN, 2D-CNN...
- Show that a relatively robust solution can be obtained with relatively scarce and uniform data samples (~160 each)
- How the improvement in robustness can be achieve via data-augmentation
- Explore how problem formulation impacts model performance more than model complexity (binary vs multilabel classification)
- Build a small but flexible codebase that can be extended with new models and datasets (easy to fine tune using few new samples)

## 🧠 Key Insight

Although audio is inherently sequential, not all audio classification problems benefit from recurrent models. For short, transient sounds like drum hits:
Most discriminative information lies in the attack phase (first ~200–300 ms).
Treating audio as a fixed-length feature representation can outperform sequence modeling.
Simplifying the task (e.g., binary classification) can dramatically improve learning stability.

## ♬ Input data format

Training data is generated from  audio files (either .wav or .ogg) loaded using Librosa library.
Input audio data is typically of 4 different types of drums, as seen below (this may be extended later for note transcription tasks):

<img width="610" height="420" alt="image" src="https://github.com/user-attachments/assets/a008b2a2-7942-4e43-ae9f-9566e61b4335" />

Audio data is then converted into spectrograms using STFT with NFFT of 1024 bins, for further processing in the frequency domain:

<img width="475" height="200" alt="image" src="https://github.com/user-attachments/assets/6c20765d-4bde-4cbe-b0c2-06b5c3840a80" /><img width="475" height="200" alt="image" src="https://github.com/user-attachments/assets/678e26be-e8df-4b14-a82f-e642d6bac108" />

Individual audio files are loaded randomly in a loop and concatenated with periods of silence of random length in between.
Both recorded and artifical background noises are added to the silence to improve data realism.
Some further data augumentation techniques are applied on each batch sample, such as filtering, random masking, time warping etc, which increases robustness of the trained algorithm.
During concatenation, labels are generated from the file/folder names. 
Resulting training and validation data batch samples look somewhat like the example shown below (before augmentation):

<img width="520" height="350" alt="image" src="https://github.com/user-attachments/assets/2aa32381-f353-46b5-afeb-0fe833b5ea51" />


## 🧩 Problem Formulations (Two Branches)

This repository supports two different ways of formulating the classification problem, which can be treated as conceptual “branches” of the project.
At this point, individual drums appear in data one at a time and there is no overlap. Other approaches shall be further explored in the future.

🔹 Branch 1: Multi-Class Drum Classification

   Problem statement:

   Given a short audio clip, predict which drum class it belongs to
   (e.g., kick, snare, cymbal, hi-hat).

   Characteristics:

   - Multi-class classification
   - Higher ambiguity between similar sounds
   - More sensitive to dataset imbalance and labeling noise
   - Useful for understanding model limitations

Typical label set: 
       
      { kick, snare, cymbal, hihat, tom }

🔹 Branch 2: Binary / One-vs-Rest Classification

   Problem statement:

   Given a short audio clip, determine whether it belongs to a specific drum class or not
   (e.g., “kick” vs “not kick”).

   Characteristics:

   - Binary classification
   - Easier optimization and interpretation
   - Often better accuracy with limited data
   - Useful as a stepping stone before multi-class classification

   Example tasks:

    ''kick vs not-kick
    snare vs not-snare''

## 🏗 Available Model Types

The repository includes implementations and experiments with the following model families:

🔸 1. Feed-Forward (Dense) Models

  - Operate on fixed-size feature vectors (e.g., flattened spectrograms)
  - Simple baseline
  - Often surprisingly effective for short, percussive sounds
 
  Use case:
  Quick experiments, debugging pipelines, baseline performance.

🔸 2. Recurrent Models (RNN / LSTM)

Process audio features as temporal sequences
Designed to model time dependencies

Observations:
 - Harder to train reliably on short drum sounds
 - Sensitive to sequence length and padding
 - Often unnecessary when temporal structure is minimal

Use case:
Exploratory learning and understanding sequence modeling limitations.

🔸 3. Spectrogram-Based Models (CNN-style inputs)

Treat spectrograms as 2D representations

Enable spatial pattern learning in time–frequency space

Note:
While full CNN architectures may not be fully implemented yet, the codebase is structured to support them easily.

## 📁 Repository Structure

```text
├── Drum_classification.ipynb   # End-to-end demo notebook
├── augmentations.py            # Audio data augmentation utilities
├── main_model_train.py         # Training entry point
├── model_prediction.py         # Inference / evaluation script
├── models.py                   # Model definitions
├── utils.py                    # Audio loading and preprocessing
└── .gitignore
```

## 🚀 Typical Workflow

Load and preprocess audio files (e.g., STFT or mel spectrograms)
Choose a problem formulation (multi-class or binary)
Select a model type
Train using main_model_train.py
Evaluate or predict using model_prediction.py

EDIT: Repo is to be migrated and run on AWS with a GUI which will allow easier exploration of the model training, fine-tuning and evaluation .

## 🛠 Requirements (Example)
pip install numpy librosa tensorflow scikit-learn matplotlib

📌 Results

TBD

🔮 Possible Extensions

Full CNN architectures for spectrogram inputs
Transformer-based audio models
Simultaneous drum hits
Velocity/pitch detection in other instruments
Fine tuning
Develop interface for fine-tuning and parametrization and host on AWS
Real-time audio classification

📜 License

MIT License (or update as needed)
