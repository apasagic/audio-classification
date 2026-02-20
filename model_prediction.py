import numpy as np
import tensorflow as tf
from tensorflow import keras
import utils
import os
from sklearn.preprocessing import OneHotEncoder

from utils import load_audio_file, compute_stft, plot_spectrogram, check_model_predictions, load_wav_data, create_one_hot_encoder_batch, generate_time_windows

WINDOW_SIZE = 0.025  # in seconds (100 ms)
HOP_SIZE = 0.05  # in seconds (50 ms)
N_FFT = 2048

data = np.load("generated_windows.npz", allow_pickle=True)

encoder = data["encoder"].item()
norm_params = data["norm_params"]

# Paths
model_path = "best_cnn_model.keras"
audio1_path = "audio1.ogg"
audio2_path = "audio2.ogg"
    
# Check if model exists
if not os.path.exists(model_path):
    print(f"Error: Model file {model_path} not found!")
    print("Make sure you have trained the model first.")
    
# Check if audio files exist
if not os.path.exists(audio1_path):
    print(f"Warning: {audio1_path} not found!")
if not os.path.exists(audio2_path):
    print(f"Warning: {audio2_path} not found!")
   
model = keras.models.load_model(model_path)

audio, _ = load_audio_file("audio1.ogg")
magnitude_db, _ = compute_stft(audio, n_fft=N_FFT)
magnitude_db = magnitude_db.T  # (time, features)
magnitude_db = magnitude_db.astype(np.float32)
  # Normalize using training parameters
magnitude_db = np.expand_dims(magnitude_db, axis=0)  # Add batch dimension
magnitude_db = (magnitude_db - norm_params[0].transpose(0,2,1)) / norm_params[1].transpose(0,2,1)
predictions = model.predict(magnitude_db)
predicted_classes = encoder.inverse_transform(predictions[0])
check_model_predictions(model, magnitude_db, predicted_classes[:,0])
print(f"Predicted classes for audio2.ogg: {predicted_classes[:,0]}")
