

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

from keras.models import Sequential
from keras.layers import LSTM, SimpleRNN, Dense, Bidirectional, Dropout, TimeDistributed, Activation, Input, Reshape, Lambda, RepeatVector
from keras.layers import Conv1D, Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten
from keras.optimizers import Adam

class modelNN:
    def __init__(self,model_type):
        # ---- Model Init ----
        if(model_type=='1D_CNN'):

            model_CNN = Sequential([
            Input(shape=(None, 1025)),   # (time, features)

            #Conv1D(128, kernel_size=5, padding="same", activation="relu"),
            #Dropout(0.3),

            #Conv1D(64, kernel_size=3, padding="same", activation="relu"),

            #Dense(5, activation="softmax")   # per-frame prediction
           
            Conv1D(256, kernel_size=5, padding="same", activation="linear", use_bias=True,data_format="channels_last"),
            Dropout(0.1),

            Conv1D(128, kernel_size=5, padding="same", activation="linear", use_bias=True,data_format="channels_last"),
            Dropout(0.1),

            Conv1D(64, kernel_size=3, padding="same", activation="linear", use_bias=True,data_format="channels_last"),

            Dense(5, activation="softmax")   # per-frame prediction
            ])

            model_CNN.compile(
                optimizer=Adam(learning_rate=1e-3),
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )

            model_CNN.summary()
            self.model = model_CNN

        elif(model_type=='FF'):

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
            self.model = model_FF

        elif(model_type=='2D_CNN'):

            model_2D_CNN = Sequential([
            Input(shape=(None, 1025, 1)),      # (time, frequency, channels)
            
            Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.3),

            Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.3),

            Conv2D(256, kernel_size=(3, 3), padding="same", activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.3),

            Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu"),
            Dropout(0.3),

            # Global average pooling to handle variable time dimension
            Lambda(lambda x: tf.reduce_mean(x, axis=[2])),  # Average over frequency dimension
            
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(5, activation='softmax')  # per-frame prediction
            ])

            model_2D_CNN.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
            )
            
            model_2D_CNN.summary()
            self.model = model_2D_CNN

    # ---- Training model ----
    def train_model(self,X_train,Y_train,X_val,Y_val,early,checkpoint,batch_size=32,epochs=2500):

        history = self.model.fit(
            X_train,
            Y_train,
            validation_data=(X_val, Y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[early, checkpoint]
        )

        return history

