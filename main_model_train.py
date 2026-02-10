import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import random
import utils

from tensorflow.keras.callbacks import ModelCheckpoint

from keras.models import Sequential
from keras.layers import LSTM, SimpleRNN, Dense, Bidirectional, Dropout, TimeDistributed, Activation, Input, Reshape, Lambda, RepeatVector
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


# Define constants
#SR = 22050
#DURATION = 1  # in seconds
WINDOW_SIZE = 0.025  # in seconds (100 ms)
HOP_SIZE = 0.05  # in seconds (50 ms)
N_FFT = 2048

directory = 'Drum_samples'
audio_array, labels = utils.load_wav_data(directory)
audio_array = np.array(audio_array)

windows, labels = utils.generate_time_windows(audio_array, labels, window_size = 435, max_length = 10000, sr = 22050)

labels_one_hot = utils.create_one_hot_encoder_batch(labels)

# Create input output tensors
X_train = windows[:int(0.8*len(windows)),:,:]
Y_train = labels_one_hot[:int(0.8*len(labels_one_hot)),:,:]

X_val = windows[int(0.8*len(windows)):,:,:]
Y_val = labels_one_hot[int(0.8*len(labels_one_hot)):,:,:]

# Print shape of Input/Output vector for training for check
print(X_train.shape)
print(Y_train.shape)
#print(X_val.shape)
#print(Y_val.shape)
# Use the same encoder to transform the test data
#y_test_OH = encoder.transform(np.array(y_test).reshape(-1, 1))

checkpoint = ModelCheckpoint(
    "best_model.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
) 

# Early stop is used to make sure model is not overfitting. If 'val_loss' is not improved within 10 epochs (patience=10), the training is automaticlly stopped.
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')


model_FF = Sequential([
    Input(shape=(None, 1025)),      # (time, features)
    Dense(1025, activation='relu'),
    Dense(512, activation='relu'),
    Dense(5, activation='softmax')  # per-frame prediction
])

model_FF.compile(
    optimizer=Adam(learning_rate=5e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model_FF.summary()

model_FF.fit(
    X_train.transpose((0,2,1)),
    Y_train,
    batch_size=X_train.shape[0],   # IMPORTANT: now you actually have batches
    epochs=2500
)

print("Evaluating on validation set:")
