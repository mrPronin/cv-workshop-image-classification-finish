# Image classification

Current repository is based on [Image Classification for beginner](https://medium.com/mlearning-ai/image-classification-for-beginner-a6de7a69bc78) tutorial (with some refactoring).

This repository will also be the basis for the "Workshop 1. Computer Vision | Image classification".

## How to setup

1. Install [Miniconda](https://docs.anaconda.com/free/miniconda/) for your operation system. Current repository using conda 24.1.0
2. Clone [this repository](https://github.com/mrPronin/cv-workshop-image-classification-start) to your workshop folder
3. Run next command to create an environment from an environment.yml file

```bash
conda env create -y -f environment.yml
```
Verify that the new environment was installed correctly:
```bash
conda env list
```

4. Activate a new environment
```bash
conda activate cv-workshop-01-3_11
```

5. We will use two data sets for the workshop - [Rice Image Dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset/data) and [Cats-vs-Dogs](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset/discussion/438473).

- Create `data` subfolder in the project folder.
- Download [Cats-vs-Dogs](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset/discussion/438473) data extract to the `data` folder.
- Download [Rice Image Dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset/data) data extract to the `data` folder.

6. Create `constants.py`
```python
# constants.py

max_images_per_label = 500
data_set_folder = "dogs-cats"
# data_set_folder = "rice-image-dataset"
data_set_path = f"data/{data_set_folder}/"
image_size = (160, 160)
# image_size = (96, 96)
```

7. Create `utils.py`
```python
# utils.py

def encode_labels(df, label_mapping):
    df["encode_label"] = df["label"].map(label_mapping)
    return df


def map_labels(labels):
    return {label: idx for idx, label in enumerate(labels)}
```

8. Create `main.py`
```python
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
from constants import (
    max_images_per_label,
    data_set_folder,
    data_set_path,
    image_size
)
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
    X_train, X_test_val, y_train, y_test_val = train_test_split(
        X,
        y,
        test_size=0.2
    )
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
    base_model = VGG16(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

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
    check_images(
        data_set_path,
        max_images_per_label=max_images_per_label
    )
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

    history = model.fit(
        X_train,
        y_train,
        epochs=5,
        validation_data=(X_val, y_val)
    )

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
```

9. Create `test_model.py`
```python
# test_model.py

import tensorflow as tf
import numpy as np
import os
from constants import (
    # max_images_per_label,
    data_set_folder,
    data_set_path,
    image_size
)


def load_and_preprocess_image(img_path):
    # Load the image file, resizing it to the input size of our model
    img = tf.keras.preprocessing.image.load_img(
        img_path,
        target_size=image_size
    )
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

    # img_path = "data/dogs-cats/Dog/1275.jpg"
    img_path = "data/dogs-cats/Cat/1584.jpg"
    predict_image(model, img_path, labels)
    # print(predicted_label)


if __name__ == "__main__":
    main()
```

10. Train the model
```python
python ./main.py
```

11. Test the model with selected image
```python
python ./test_model.py
```