from tracemalloc import start
import numpy as np
import math
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import random
from sklearn.preprocessing import OneHotEncoder 

def load_audio_file(file_path):
    audio, sr = librosa.load(file_path, mono=True, dtype=np.float32)
    return audio,sr


def compute_stft(audio, n_fft=2048, hop_length=512, win_length=2048):
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    stft = stft.astype(np.float32)
    magnitude, phase = librosa.magphase(stft)
    magnitude_db = librosa.amplitude_to_db(np.abs(magnitude), ref=np.max)
    return magnitude_db, phase


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
            audio, sr = load_audio_file(file_path)
            magnitude_db, phase = compute_stft(audio[:8000], n_fft=2048)
            data.append(magnitude_db)
            # Get a label Y from a file name
            label = os.path.basename(os.path.dirname(file_path))  # Assuming each subdirectory represents a class            
            labels.append(label)


    return np.array(data), np.array(labels)

def create_one_hot_encoder_batch(labels):
    labels_flat = labels.reshape(-1, 1)
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded = encoder.fit_transform(labels_flat)
    labels_one_hot = one_hot_encoded.reshape(labels.shape[0], labels.shape[1], -1)
    return labels_one_hot

def generate_time_windows(audio_array, labels, window_size, max_length, sr):

   #windows = np.array(no_windows,audio_array.shape[0],audio_array.shape[1])

   total_length = 0
   total_audio = np.array([])
   i = 0

   chunks = []
   labels_chunks = []
   
   while total_length < max_length:
      
      audio_sample_ind = random.randint(0, audio_array.shape[0]-1)
      audio_sample = audio_array[audio_sample_ind, :, :]
      
      chunks.append(audio_sample)
      
      # Get the label for this specific audio sample
      sample_label = labels[audio_sample_ind]
      labels_chunks.extend([sample_label] * audio_sample.shape[1])

      silence_window = -80*np.ones((audio_array.shape[1], random.randint(0, 2)))
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

   #np.savez_compressed(
   # "generated_windows.npz",
   # windows=windows_trimmed,
   # labels=labels_trimmed
   #)

   mean = np.mean(windows_trimmed, axis=(0, 2), keepdims=True)
   std = np.std(windows_trimmed, axis=(0, 2), keepdims=True) + 1e-8
   windows_trimmed = (windows_trimmed - mean) / std

   return windows_trimmed, labels_trimmed
   

