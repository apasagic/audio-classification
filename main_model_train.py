import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import random
import utils
import models

from utils import load_audio_file, compute_stft, plot_spectrogram, check_model_predictions, load_wav_data, create_one_hot_encoder_batch, generate_time_windows

from tensorflow.keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

from augmentations import add_background_noise

CREATE_WINDOWS = True

# Define constants
#SR = 22050
#DURATION = 1  # in seconds
WINDOW_SIZE = 0.025  # in seconds (100 ms)
HOP_SIZE = 0.05  # in seconds (50 ms)
N_FFT = 2048

directory = 'Drum_samples'
audio_array, labels = load_wav_data(directory)
audio_array = np.array(audio_array)

# Use the same encoder to transform the test data
#y_test_OH = encoder.transform(np.array(y_test).reshape(-1, 1))

# ---- Callbacks ----
checkpoint = ModelCheckpoint(
    "best_cnn_model.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True,   # IMPORTANT
    verbose=1
)

if(CREATE_WINDOWS):
   
  #windows, labels, norm_params = generate_time_windows(audio_array, labels, window_size = 150, max_length = 2500, sr = 22050, add_augumentation=True)
  windows_train, labels_train, norm_params = generate_time_windows(audio_array[0:int(0.8*len(audio_array))], labels, window_size = 150, max_length = 2500, sr = 22050, add_augumentation=True)
  windows_train = add_background_noise(windows_train, path_to_noise="background-sounds\\background-room.wav",norm_params=norm_params)
  windows_val, labels_val, _ = generate_time_windows(audio_array[int(0.8*len(audio_array)):], labels, window_size = 150, max_length = 800, sr = 22050, add_augumentation=False)

  Y_train, encoder = create_one_hot_encoder_batch(labels_train)
  Y_val = encoder.transform(np.array(labels_val).reshape(-1, 1))
  Y_val = Y_val.reshape(labels_val.shape[0], labels_val.shape[1], -1)

  # Print shape of Input/Output vector for training for check
  X_train = windows_train.transpose((0,2,1))  # (N, features, time) -> (N, time, features)
  X_val = windows_val.transpose((0,2,1))      # (N, features, time) -> (N, time, features)

  #np.savez_compressed(
  #  "generated_windows.npz",
  #  X_train=X_train,
  #  Y_train=Y_train,
  #  X_val=X_val,
  #  Y_val=Y_val,
  #  encoder=encoder,
  #  norm_params=norm_params
  #)

else:

  data = np.load("generated_windows.npz", allow_pickle=True)
  X_train = data["X_train"]
  Y_train = data["Y_train"]
  X_val = data["X_val"]
  Y_val = data["Y_val"]
  encoder = data["encoder"].item()
  norm_params = data["norm_params"]

model = models.modelNN(model_type='1D_CNN',out_dim=Y_train.shape[2])

model.model.fit(
    X_train,
    Y_train,
    validation_data=(X_val, Y_val),
    batch_size=16,
    epochs=250  ,
    callbacks=[early, checkpoint]
)

print("Evaluating on validation set:")

#predictions = model.model.predict(X_val[0:1,:,:])
#class_names = encoder.inverse_transform(predictions[0,:,:])
#check_model_predictions(model.model, X_val[0:1,:,:], class_names[:,0])

#############################
# Testing on new audio files
###############################

# Audio1
audio, _ = load_audio_file("audio1.ogg")
magnitude_db, _ = compute_stft(audio, n_fft=N_FFT)
magnitude_db = magnitude_db.T  # (time, features)
magnitude_db = magnitude_db.astype(np.float32)
  # Normalize using training parameters
magnitude_db = np.expand_dims(magnitude_db, axis=0)  # Add batch dimension
magnitude_db = (magnitude_db - norm_params[0].transpose(0,2,1)) / norm_params[1].transpose(0,2,1)
predictions = model.model.predict(magnitude_db)
predicted_classes = encoder.inverse_transform(predictions[0])
check_model_predictions(model.model, magnitude_db, predicted_classes[:,0])
print(f"Predicted classes for audio2.ogg: {predicted_classes[:,0]}")

# Audio2
audio, _ = load_audio_file("audio2.ogg")
magnitude_db, _ = compute_stft(audio, n_fft=N_FFT)
magnitude_db = magnitude_db.T  # (time, features)
magnitude_db = magnitude_db.astype(np.float32)
  # Normalize using training parameters
magnitude_db = np.expand_dims(magnitude_db, axis=0)  # Add batch dimension
magnitude_db = (magnitude_db - norm_params[0].transpose(0,2,1)) / norm_params[1].transpose(0,2,1)
predictions = model.model.predict(magnitude_db)
predicted_classes = encoder.inverse_transform(predictions[0])
check_model_predictions(model.model, magnitude_db, predicted_classes[:,0])
print(f"Predicted classes for audio2.ogg: {predicted_classes[:,0]}")

# Audio3
audio, _ = load_audio_file("audio3.ogg")
magnitude_db, _ = compute_stft(audio, n_fft=N_FFT)
magnitude_db = magnitude_db.T  # (time, features)
magnitude_db = magnitude_db.astype(np.float32)
  # Normalize using training parameters
magnitude_db = np.expand_dims(magnitude_db, axis=0)  # Add batch dimension
magnitude_db = (magnitude_db - norm_params[0].transpose(0,2,1)) / norm_params[1].transpose(0,2,1)
predictions = model.model.predict(magnitude_db)
predicted_classes = encoder.inverse_transform(predictions[0])
check_model_predictions(model.model, magnitude_db, predicted_classes[:,0])
print(f"Predicted classes for audio2.ogg: {predicted_classes[:,0]}")