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

from tensorflow.keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping


CREATE_WINDOWS = True

# Define constants
#SR = 22050
#DURATION = 1  # in seconds
WINDOW_SIZE = 0.025  # in seconds (100 ms)
HOP_SIZE = 0.05  # in seconds (50 ms)
N_FFT = 2048

directory = 'Drum_samples'
audio_array, labels = utils.load_wav_data(directory)
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

model = models.modelNN(model_type='1D_CNN')

if(CREATE_WINDOWS):
   windows, labels = utils.generate_time_windows(audio_array, labels, window_size = 150, max_length = 800, sr = 22050)
else:
   data = np.load("generated_windows.npz", allow_pickle=True)
   windows = data["windows"]      # shape: (N, 435, 1025)
   labels  = data["labels"]       # shape: (N, 435)

labels_one_hot = utils.create_one_hot_encoder_batch(labels)


if(windows.shape[0]!=1):
   
  # Create input output tensors
  X_train = windows[:int(0.8*windows.shape[0]),:,:]
  Y_train = labels_one_hot[:int(0.8*windows.shape[0]),:,:]

  X_val = windows[int(0.8*windows.shape[0]):,:,:]
  Y_val = labels_one_hot[int(0.8*windows.shape[0]):,:,:]

  # Print shape of Input/Output vector for training for check
  X_train = X_train.transpose((0,2,1))  # (N, features, time) -> (N, time, features)
  X_val = X_val.transpose((0,2,1))      # (N, features, time) -> (N, time, features)

else:
   
  # Create input output tensors
  X_train = windows[:,:int(0.8*windows.shape[1]),:]
  Y_train = labels_one_hot[:,:int(0.8*windows.shape[1]),:]

  X_val = windows[:,int(0.8*windows.shape[1]):,:]
  Y_val = labels_one_hot[:,int(0.8*windows.shape[1]):,:]

  # Print shape of Input/Output vector for training for check
#  X_train = X_train.transpose((0,2,1))  # (N, features, time) -> (N, time, features)
#  X_val = X_val.transpose((0,2,1))      # (N, features, time) -> (N, time, features)

print(X_train.shape)
print(Y_train.shape)
print(X_val.shape)
print(Y_val.shape)

model.model.fit(
    X_train,
    Y_train,
    validation_data=(X_val, Y_val),
    batch_size=16,
    epochs=1500,
    callbacks=[early, checkpoint]
)

print("Evaluating on validation set:")
