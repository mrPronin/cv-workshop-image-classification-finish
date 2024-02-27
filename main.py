# main.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Input, Flatten, Dropout, Dense
import numpy as np
from PIL import Image
from constants import max_images_per_label, data_set_folder, data_set_path, image_size
from utils import encode_labels, map_labels


def check_images(data_set_path, max_images_per_label=None):
    # Dynamically list directories in the dataset path
    labels = [
        d
        for d in os.listdir(data_set_path)
        if os.path.isdir(os.path.join(data_set_path, d))
    ]
    for label in labels:
        label_path = os.path.join(data_set_path, label)
        files = os.listdir(label_path)
        if max_images_per_label:
            files = files[:max_images_per_label]
        for img_file in files:
            img_path = os.path.join(label_path, img_file)
            try:
                # print(f"File name: {img_file}")
                Image.open(img_path)
            except:  # noqa: E722
                # os.remove(img_path)
                print(img_path)


def load_and_prepare_data(data_set_path, max_images_per_label=None):
    # Dynamically list directories in the dataset path
    labels = [
        d
        for d in os.listdir(data_set_path)
        if os.path.isdir(os.path.join(data_set_path, d))
    ]

    # create a dataframe of image path and label
    img_list, label_list = [], []
    for label in labels:
        label_path = os.path.join(data_set_path, label)
        files = os.listdir(label_path)
        if max_images_per_label:
            files = files[:max_images_per_label]
        for img_file in files:
            img_list.append(os.path.join(label_path, img_file))
            label_list.append(label)
    return pd.DataFrame({"img": img_list, "label": label_list}), labels


def show_sample_images(df, labels):
    fig, ax = plt.subplots(ncols=len(labels), figsize=(20, 4))
    fig.suptitle("Category")
    random_num = 12
    for i, label in enumerate(labels):
        path = df[df["label"] == label]["img"].iloc[random_num]
        ax[i].set_title(label)
        ax[i].imshow(plt.imread(path))
    plt.show()


def prepare_images(df, image_size=(96, 96)):
    # Prepare a model training dataset
    X, y = [], []
    for _, row in df.iterrows():
        try:
            img = cv2.imread(row["img"])
            img = cv2.resize(img, image_size)
            img = img / 255.0
            X.append(img)
            y.append(row["encode_label"])
        except Exception:
            print(f"Read failed for:{row['img']}")
    return np.array(X), np.array(y)


def split_data(X, y):
    # Split into training (80%) and temporary test (20%)
    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.2)
    # Split the temporary test set into validation (50% of X_temp)
    # and test (50% of X_temp)
    # This means we're effectively splitting the original dataset into
    # 80% train, 10% validation, and 10% test
    X_test, X_val, y_test, y_val = train_test_split(
        X_test_val, y_test_val, test_size=0.5
    )
    return X_train, X_test, X_val, y_train, y_test, y_val


def prepare_model(input_shape, num_classes):
    # Use VGG16 as a base model
    base_model = VGG16(input_shape=input_shape, include_top=False, weights="imagenet")

    print("Base model summary:")
    base_model.summary()

    # Freeze all except last three
    for layer in base_model.layers[:-3]:
        layer.trainable = False

    model = Sequential(
        [
            Input(shape=input_shape),
            base_model,
            Flatten(),
            Dropout(0.2),
            Dense(256, activation="relu"),
            Dropout(0.2),
            Dense(num_classes, activation="softmax"),
        ]
    )

    print("Model summary:")
    model.summary()

    return model


def train_model(model):
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["acc"]
    )
    return model


def plot_metrics(history):
    # Plot accuracy
    plt.plot(history.history["acc"], marker="o")
    plt.plot(history.history["val_acc"], marker="o")
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="lower right")
    plt.show()

    # Plot loss
    plt.plot(history.history["loss"], marker="o")
    plt.plot(history.history["val_loss"], marker="o")
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper right")
    plt.show()


def main():
    check_images(data_set_path, max_images_per_label=max_images_per_label)
    df, labels = load_and_prepare_data(
        data_set_path, max_images_per_label=max_images_per_label
    )

    # count the number of images of each category
    print(df["label"].value_counts())

    show_sample_images(df, labels)

    # know image shape
    print(f"Image shape: {plt.imread(df['img'][0]).shape}")

    # Create a dataframe for mapping label
    label_mapping = map_labels(labels)
    df = encode_labels(df, label_mapping)
    print(df.head())

    X, y = prepare_images(df, image_size)
    X_train, X_test, X_val, y_train, y_test, y_val = split_data(X, y)

    model = prepare_model((*image_size, 3), len(labels))
    model = train_model(model)

    history = model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))

    # model evaluation
    model.evaluate(X_test, y_test)

    # store trained model
    model_filename = f"./data/trained-model-{data_set_folder}.keras"
    model.save(model_filename)

    # load model
    # tensorflow.keras.models.load_model(model_filename)

    # visualize the model
    plot_metrics(history)


if __name__ == "__main__":
    main()
