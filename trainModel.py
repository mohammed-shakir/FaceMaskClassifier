from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

# Load data
try:
    with open("images_x.pickle", "rb") as file:
        images_x = pickle.load(file)
    with open("labels_y.pickle", "rb") as file:
        labels_y = pickle.load(file)
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

images_x = images_x/255.0

# Layers
dense_layers = [1] # Tested with 0, 1, 2
layer_sizes = [128] # Tested with 32, 64, 128
conv_layers = [3] # Tested with 1, 2, 3
epochs = [10] # Tested with 10, 20, 30, 50, 100

# Create the convolutional neural network
for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            for epoch in epochs:
                # tensorboard --logdir=logs\\
                name = "{}-conv-{}-nodes-{}-dense-{}-epochs-{}".format(conv_layer, layer_size, dense_layer, epoch, int(time.time()))
                print(name)
                tensorboard = TensorBoard(log_dir=f"logs/{name}")
                callbacks=[tensorboard]

                model = Sequential()

                # Convolutional Layer
                model.add(Conv2D(layer_size, (3, 3), input_shape=images_x.shape[1:]))
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

                model.compile(loss='binary_crossentropy',
                            optimizer='adam', metrics=['accuracy'])

                model.fit(images_x, labels_y, batch_size=32, epochs=epoch, validation_split=0.1, callbacks=callbacks)

model.save("model.h5")