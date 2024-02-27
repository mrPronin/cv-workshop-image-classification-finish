# test_model.py

import tensorflow as tf
import numpy as np
import os
from constants import (
    # max_images_per_label,
    data_set_folder,
    data_set_path,
    image_size,
)


def load_and_preprocess_image(img_path):
    # Load the image file, resizing it to the input size of our model
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=image_size)
    # Convert the image to array
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    # Normalize the image pixels
    img_array /= 255.0

    # Expand dimensions to fit the batch size
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict_image(model, img_path, labels):
    # Preprocess the image
    img_array = load_and_preprocess_image(img_path)

    # Make predictions
    predictions = model.predict(img_array)

    percentages = tf.nn.softmax(predictions[0])
    for i, label in enumerate(labels):
        print(f"{label}: {percentages[i] * 100:.2f}%")


def main():
    model_path = f"./data/trained-model-{data_set_folder}.keras"
    model = tf.keras.models.load_model(model_path)

    labels = [
        d
        for d in os.listdir(data_set_path)
        if os.path.isdir(os.path.join(data_set_path, d))
    ]

    # img_path = "data/PetImages/Dog/1275.jpg"
    img_path = "data/PetImages/Cat/1585.jpg"
    predict_image(model, img_path, labels)
    # print(predicted_label)


if __name__ == "__main__":
    main()
