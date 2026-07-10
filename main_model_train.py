import numpy as np

import models
from utils import (
    check_model_predictions,
    compute_stft,
    convert_to_multilabel,
    generate_time_windows,
    load_audio_file,
    load_wav_data,
    probs_to_class_labels,
)

from tensorflow.keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping


CREATE_WINDOWS = True
N_FFT = 2048
CLASSES = ["kick", "snare", "hihat"]


def normalize_prediction_input(magnitude_db, norm_params):
    """Match the normalization/layout used for training windows."""
    x = magnitude_db.T.astype(np.float32)  # (time, features)
    x = np.expand_dims(x, axis=0)          # (batch, time, features)
    mean, std = norm_params
    return (x - mean.transpose(0, 2, 1)) / std.transpose(0, 2, 1)


def predict_audio_file(model, file_path, norm_params, threshold=0.5):
    audio, _ = load_audio_file(file_path)
    magnitude_db, _ = compute_stft(audio, n_fft=N_FFT)
    x = normalize_prediction_input(magnitude_db, norm_params)
    predictions = model.model.predict(x)
    predicted_classes = probs_to_class_labels(predictions[0], CLASSES, threshold=threshold)
    check_model_predictions(model.model, x[0], predicted_classes)
    print(f"Predicted classes for {file_path}: {predicted_classes}")


# Load drum samples and create synthetic training windows.
directory = "Drum_samples"
audio_array, labels = load_wav_data(directory)
audio_array = np.array(audio_array)

checkpoint = ModelCheckpoint(
    "best_cnn_model.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1,
)

early = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True,
    verbose=1,
)

if CREATE_WINDOWS:
    windows_train, labels_train, norm_params = generate_time_windows(
        audio_array[0:int(0.8 * len(audio_array))],
        labels,
        window_size=150,
        max_length=2500,
        sr=22050,
        add_augumentation=True,
    )
    windows_val, labels_val, _ = generate_time_windows(
        audio_array[int(0.8 * len(audio_array)):],
        labels,
        window_size=150,
        max_length=800,
        sr=22050,
        add_augumentation=False,
    )

    Y_train = convert_to_multilabel(labels_train, CLASSES)
    Y_val = convert_to_multilabel(labels_val, CLASSES)
    X_train = windows_train.transpose((0, 2, 1))
    X_val = windows_val.transpose((0, 2, 1))
else:
    data = np.load("generated_windows.npz", allow_pickle=True)
    X_train = data["X_train"]
    Y_train = data["Y_train"]
    X_val = data["X_val"]
    Y_val = data["Y_val"]
    norm_params = data["norm_params"]

model = models.modelNN(model_type="1D_CNN", out_dim=Y_train.shape[2])

model.model.fit(
    X_train,
    Y_train,
    validation_data=(X_val, Y_val),
    batch_size=16,
    epochs=250,
    callbacks=[early, checkpoint],
)

print("Evaluating on validation set:")
val_predictions = model.model.predict(X_val[0:1, :, :])
val_labels = probs_to_class_labels(val_predictions[0], CLASSES, threshold=0.5)
check_model_predictions(model.model, X_val[0], val_labels)

for audio_path in ["audio1.ogg", "audio2.ogg", "audio4.ogg"]:
    predict_audio_file(model, audio_path, norm_params, threshold=0.5)
