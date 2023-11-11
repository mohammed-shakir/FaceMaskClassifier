import cv2
import os
import numpy as np
import pickle
import random

data = "dataset"
categories = ["with_mask", "without_mask"]
img_size = 50
training_data = []
images_x = []
labels_y = []

def create_training_data():
    for category in categories:
        path = os.path.join(data, category)
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist")

        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img_array is None:
                    raise ValueError(f"Unable to read image {img_path}")

                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                print(f"Error reading image {img_path}: {e}")

create_training_data()

random.shuffle(training_data)

for features, label in training_data:
    images_x.append(features)
    labels_y.append(label)

images_x = np.array(images_x).reshape(-1, img_size, img_size, 1)
labels_y = np.array(labels_y)

# Save images_x
with open("images_x.pickle", "wb") as pickle_out:
    pickle.dump(images_x, pickle_out)

# Save labels_y
with open("labels_y.pickle", "wb") as pickle_out:
    pickle.dump(labels_y, pickle_out)
