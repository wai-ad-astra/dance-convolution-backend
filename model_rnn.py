n_timesteps= 50
n_features = 2
n_outputs = 2
import json
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense,Input,Conv1D,Dropout,MaxPooling1D,Flatten
from tensorflow.keras.models import Model,Sequential

model = Sequential()
model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])