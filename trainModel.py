import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import cv2
import pickle
import numpy as np
import time

# Load data
X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

# Pixels
X = X/255.0

# Layers
dense_layers = [1]
layer_sizes = [64]
conv_layers = [2]

# Create the convolutional neural network
for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            # tensorboard --logdir=logs\\
            # name = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            # tensorboard = TensorBoard(log_dir="logs\\{}".format(name))
            # callbacks=[tensorboard]

            model = Sequential()

            # Convolutional Layer
            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            # Convolutional Layer
            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            # Flatten the data
            model.add(Flatten())

            # Dense Layer
            for l in range(dense_layer):
                model.add(Dense(32))
                model.add(Activation('relu'))
                model.add(Dropout(0.2))

            # Output Layer
            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            # Compile
            model.compile(loss='binary_crossentropy',
                          optimizer='adam', metrics=['accuracy'])

            # Train model
            model.fit(X, y, batch_size=32, epochs=10, validation_split=0.1)

# Save model
model.save("model.h5")
