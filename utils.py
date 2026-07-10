from tracemalloc import start
import numpy as np
import math
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import random

from sklearn.preprocessing import OneHotEncoder
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from augmentations import augment_spectrogram



def detect_onsets(probs, threshold=0.5, min_distance=3):
    """
    Detect onset peaks per class.

    Returns:
        list of (frame_index, class_index, probability)
    """
    events = []

    num_classes = probs.shape[1]

    for c in range(num_classes):
        class_probs = probs[:, c]

        peaks, _ = find_peaks(
            class_probs,
            height=threshold,
            distance=min_distance
        )

        for p in peaks:
            events.append((p, c, class_probs[p]))

    return events

def smooth_probabilities(probs, sigma=1.5):
    """
    Smooth per-frame probabilities over time.

    probs shape: (time, classes)
    """
    smoothed = np.zeros_like(probs)

    for c in range(probs.shape[1]):
        smoothed[:, c] = gaussian_filter1d(probs[:, c], sigma=sigma)

    return smoothed

def compute_frame_energy(magnitude_db):
    """
    Estimate energy per frame from spectrogram.
    """
    return np.mean(magnitude_db, axis=1)


def apply_energy_gate(probs, energy, threshold_percentile=20):
    """
    Zero out predictions during low-energy frames.
    """
    threshold = np.percentile(energy, threshold_percentile)

    mask = energy > threshold

    gated = probs.copy()
    gated[~mask] = 0

    return gated

def probs_to_class_labels(probs, classes, threshold=0.5):
    """
    Convert sigmoid outputs to per-frame class labels.

    probs shape: (time, num_classes)

    Returns:
        list of string labels per frame
    """
    labels = []

    for t in range(probs.shape[0]):
        frame_probs = probs[t]

        # find active classes
        active = np.where(frame_probs >= threshold)[0]

        if len(active) == 0:
            labels.append("None")
        else:
            # if multiple active Ã¢â€ â€™ take strongest
            best = active[np.argmax(frame_probs[active])]
            labels.append(classes[best])

    return np.array(labels)

def convert_to_multilabel(labels, class_list):
    """
    Convert label arrays to multi-label binary format.

    Works with:
        (num_frames,)
        (num_windows, window_size)

    Returns:
        (num_frames, num_classes) OR
        (num_windows, window_size, num_classes)
    """

    labels = np.array(labels)
    num_classes = len(class_list)

    # --- Case 1: single sequence ---
    if labels.ndim == 1:
        num_frames = labels.shape[0]
        binary = np.zeros((num_frames, num_classes), dtype=np.float32)

        for i, lbl in enumerate(labels):
            if lbl != 'None' and lbl != 'none':
                if lbl in class_list:
                    idx = class_list.index(lbl)
                    binary[i, idx] = 1

        return binary

    # --- Case 2: batched windows ---
    elif labels.ndim == 2:
        num_windows, window_size = labels.shape
        binary = np.zeros((num_windows, window_size, num_classes), dtype=np.float32)

        for w in range(num_windows):
            for t in range(window_size):
                lbl = labels[w, t]
                if lbl != 'None' and lbl != 'none':
                    if lbl in class_list:
                        idx = class_list.index(lbl)
                        binary[w, t, idx] = 1

        return binary

    else:
        raise ValueError("labels must be 1D or 2D array of strings")


def extract_drum_events(probs, magnitude_db):
    """
    Full onset detection pipeline.
    """

    # 1. Smooth probabilities
    probs = smooth_probabilities(probs, sigma=1.5)

    # 2. Compute energy
    energy = compute_frame_energy(magnitude_db)

    # 3. Apply energy gate
    probs = apply_energy_gate(probs, energy)

    # 4. Peak detection
    events = detect_onsets(probs)

    return events

def check_model_predictions(model, stft, labels):

    # Create subplots
    fig, axs = plt.subplots(2, figsize=(12, 8))

    # Plot the spectrogram on the first subplot
    img = axs[0].imshow(stft.T, aspect='auto', origin='lower', cmap='jet')

    # Set labels and title for the first subplot
    axs[0].set_xlabel('Samples')
    axs[0].set_ylabel('Frequency Bin')
    axs[0].set_title('STFT Spectrogram')
    axs[0].grid(True)

    axs[1].plot(labels)
    axs[1].set_xlabel('Samples')
    #axs[1].set_title('Drug labels')
    axs[1].grid(True)

    # Show the plot
    plt.show()

def load_audio_file(file_path):
    audio, sr = librosa.load(file_path, mono=True, dtype=np.float32)
    return audio,sr

def compute_stft(audio, n_fft=2048, hop_length=512, win_length=2048):
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    stft = stft.astype(np.float32)
    magnitude, phase = librosa.magphase(stft)
    magnitude_db = librosa.amplitude_to_db(np.abs(magnitude), ref=np.max)
    return magnitude_db, phase

def find_events_from_predictions(predictions, threshold=0.5, sigma = 2, distance = 5):

   predictions = model.predict(magnitude_db)[0]

   # Smooth
   smoothed = gaussian_filter1d(predictions, sigma=sigma, axis=0)

   events = {}

   for drum_id in range(smoothed.shape[1]):
      curve = smoothed[:, drum_id]
      peaks, _ = find_peaks(curve,
                          height=threshold,
                          distance=distance)
      events[drum_id] = peaks

   return events


def plot_spectrogram(magnitude_db, sr, hop_length, title='Spectrogram'):
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(magnitude_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()


def load_wav_data(directory):
    data = []
    labels = []

    for root, dirs, files in os.walk(directory):

        for file in files:

          if file.endswith(".wav"):

            file_path = os.path.join(root, file)

            # Get a label Y from a file name
            label = os.path.basename(os.path.dirname(file_path))  # Assuming each subdirectory represents a class

            #if the label starts with #, skip it (used for temporary files or ignored samples)
            if(label[0]=="#"):
               continue

            labels.append(label)

            audio, sr = load_audio_file(file_path)
            magnitude_db, phase = compute_stft(audio[:8000], n_fft=2048)
            data.append(magnitude_db)



    return np.array(data), np.array(labels)

def create_one_hot_encoder_batch(labels):
    labels_flat = labels.reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = encoder.fit_transform(labels_flat)
    labels_one_hot = one_hot_encoded.reshape(labels.shape[0], labels.shape[1], -1)
    return labels_one_hot, encoder

def generate_realistic_silence(freq_bins, length):
    base = np.random.normal(loc=-80, scale=2, size=(length,freq_bins))
    smooth = gaussian_filter1d(base, sigma=5, axis=0)
    return smooth

def generate_time_windows(audio_array, labels, window_size, max_length, sr, add_augumentation=False):

   #windows = np.array(no_windows,audio_array.shape[0],audio_array.shape[1])

   total_length = 0
   total_audio = np.array([])
   onset_frames = int(0.15*sr//512)  # 100 ms onset frames

   i = 0

   chunks = []
   labels_chunks = []

   while total_length < max_length:

      audio_sample_ind = random.randint(0, audio_array.shape[0]-1)
      audio_sample = audio_array[audio_sample_ind, :, :]
      audio_sample = augmentations.augment_spectrogram(audio_sample) if add_augumentation else audio_sample

      chunks.append(audio_sample)

      # Get the label for this specific audio sample
      sample_label = labels[audio_sample_ind]
      labels_chunks.extend([sample_label] * onset_frames + ['None'] * (audio_sample.shape[1] - onset_frames))

      #silence_window = -80*np.ones((audio_array.shape[1], random.randint(0, 5)))
      silence_window = generate_realistic_silence(audio_array.shape[1], random.randint(0, 20))

      chunks.append(silence_window)
      labels_chunks.extend(['None'] * silence_window.shape[1])

      total_length += 2 * (audio_sample.shape[1] + silence_window.shape[1]) / 87
      #total_length = i*2

      i += 1

      if(i%10==0):
        print(f"{i}: Generated {total_length} seconds of audio data")

   total_audio = np.concatenate(chunks, axis=1)

   windows = np.array(total_audio)
   labels = np.array(labels_chunks)

   n_windows = windows.shape[1]//window_size

   if(n_windows != 0):
      windows_trimmed = windows[:,:window_size * n_windows].T.reshape(-1, window_size, audio_array.shape[1])
      windows_trimmed = np.transpose(windows_trimmed, (0, 2, 1))
      labels_trimmed = labels[:window_size * n_windows].reshape(n_windows, window_size)
   else:
      windows_trimmed = windows.T.reshape(1, windows.shape[1], windows.shape[0])
      labels_trimmed = labels.reshape(1, labels.shape[0])

   windows = windows.astype(np.float32)

   mean = np.mean(windows_trimmed, axis=(0,2), keepdims=True)
   std  = np.std(windows_trimmed, axis=(0,2), keepdims=True)

   windows_trimmed = (windows_trimmed - mean) / std

   norm_params = (mean, std)

   return windows_trimmed, labels_trimmed, norm_params
