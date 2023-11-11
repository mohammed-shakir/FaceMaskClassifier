import cv2
import os
import numpy as np
import pickle
import random

# Data
data = "images"

# Categories
categories = ["type1", "type2"]

# Image size
img_size = 50

# Training data
training_data = []

# Images
X = []

# Labels
y = []


def create_training_data():
    # Create training data
    for category in categories:
        path = os.path.join(data, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(
                    path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass


create_training_data()

# Shuffle images
random.shuffle(training_data)

# Set X as images and y as labels
for features, label in training_data:
    X.append(features)
    y.append(label)

# Convert X and y to a numpy array
X = np.array(X).reshape(-1, img_size, img_size, 1)
y = np.array(y)

# Save X
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

# Save y
pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
