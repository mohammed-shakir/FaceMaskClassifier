import cv2
import tensorflow as tf
import os
import random

categories = ["with_mask", "without_mask"]

def prepare(filepath):
    img_size = 50
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img_array is None:
        return None
    img_array = img_array / 255.0
    new_array = cv2.resize(img_array, (img_size, img_size))
    return new_array.reshape(-1, img_size, img_size, 1)

def extract_label(filename):
    for category in categories:
        if category in filename:
            return category
    return None

def prepare_all(dir):
    files = os.listdir(dir)
    combined = []
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            im = prepare(os.path.join(dir, file))
            if im is not None:
                label = extract_label(file)
                combined.append((im, label, file))
    random.shuffle(combined)
    x_test = [item[0] for item in combined]
    y_test = [item[1] for item in combined]
    test_files = [item[2] for item in combined]
    return x_test, y_test, test_files

model = tf.keras.models.load_model("model.h5")

# Test model images
x_test, y_test, test_files = prepare_all("testImages/")
correct_predictions = 0
total_predictions = len(x_test)

for i in range(total_predictions):
    prediction = model.predict(x_test[i])
    predicted_category = categories[1 if prediction[0][0] > 0.5 else 0]
    actual_category = y_test[i]
    print(f"File: {test_files[i]} - Prediction: {predicted_category} | Actual: {actual_category} \n")
    if predicted_category == actual_category:
        correct_predictions += 1

accuracy = correct_predictions / total_predictions
print(f"\nModel Evaluation:")
print(f"Correct Predictions: {correct_predictions}")
print(f"Incorrect Predictions: {total_predictions - correct_predictions}")
print(f"Accuracy: {accuracy:.2f}")
