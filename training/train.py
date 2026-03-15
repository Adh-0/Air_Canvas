"""
Training pipeline for the air writing character recognition CNN.

Supports the English alphabet dataset (A-Z, 26 classes).
Pre-trained weights are included in the models/ directory — this script
is only needed if you want to retrain from scratch.

Usage:
    python train.py True eng_alphabets     # train a new model
    python train.py False eng_alphabets    # load and inspect existing model
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
import keras
from keras import Input
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.utils import to_categorical


# -- Dataset configurations --

SUPPORTED_DATASETS = {
    "eng_alphabets": {
        "labels": dict(enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")),
        "input_dim": 28,
        "num_classes": 26,
    },
}


# -- Data loading & preprocessing --

class DataPipeline:
    """Loads a CSV dataset, splits it, and prepares it for training."""

    def __init__(self, dataset_key):
        if dataset_key not in SUPPORTED_DATASETS:
            valid = ", ".join(SUPPORTED_DATASETS.keys())
            raise ValueError(f"Unknown dataset '{dataset_key}'. Options: {valid}")

        cfg = SUPPORTED_DATASETS[dataset_key]
        self.name = dataset_key
        self.labels = cfg["labels"]
        self.dim = cfg["input_dim"]
        self.n_classes = cfg["num_classes"]

    def prepare(self, csv_path):
        """Read CSV, split, reshape, one-hot encode, and return train/test arrays."""
        df = pd.read_csv(csv_path).astype("float32")
        features = df.drop("0", axis=1)
        targets = df["0"]

        x_train, x_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, shuffle=True, random_state=42
        )

        x_train = x_train.values.reshape(-1, self.dim, self.dim)
        x_test = x_test.values.reshape(-1, self.dim, self.dim)

        print(f"  Train samples: {x_train.shape[0]}  |  Test samples: {x_test.shape[0]}")

        self._show_distribution(targets)
        self._show_samples(x_train)

        # add channel axis for Conv2D
        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]

        y_train_ohe = to_categorical(y_train, num_classes=self.n_classes, dtype="int")
        y_test_ohe = to_categorical(y_test, num_classes=self.n_classes, dtype="int")

        return x_train, y_train_ohe, x_test, y_test_ohe

    def _show_distribution(self, all_targets):
        counts = np.zeros(self.n_classes, dtype="int")
        for val in np.int0(all_targets):
            counts[val] += 1
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(list(self.labels.values()), counts)
        ax.set_xlabel("Count")
        ax.set_ylabel("Character")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def _show_samples(self, images):
        sample = shuffle(images[:100])[:9]
        fig, axes = plt.subplots(3, 3, figsize=(8, 8))
        for img, ax in zip(sample, axes.flatten()):
            ax.imshow(img, cmap="Greys")
            ax.axis("off")
        plt.tight_layout()
        plt.show()


# -- Model building & training --

class CharacterClassifier:
    """Builds, trains, and saves a CNN for handwritten character recognition."""

    def __init__(self, dataset_key, should_train=True):
        self.dataset_key = dataset_key
        self.should_train = should_train

        project_root = os.path.dirname(os.getcwd())
        self.weights_path = os.path.join(project_root, "models", f"model_{dataset_key}.h5")
        self.log_dir = os.path.join(project_root, f"logs_{dataset_key}")
        self.data_path = os.path.join(project_root, "data", f"{dataset_key}.csv")

    def run(self):
        pipeline = DataPipeline(self.dataset_key)
        x_train, y_train, x_test, y_test = pipeline.prepare(self.data_path)

        if self.should_train == "True":
            self._train_new_model(x_train, y_train, x_test, y_test,
                                  pipeline.dim, pipeline.n_classes)
        else:
            self._load_existing_model()

    def _build_cnn(self, input_dim, n_classes):
        inp = Input(shape=(None, input_dim, input_dim, 1))
        x = Conv2D(32, (3, 3), activation="relu")(inp)
        x = MaxPool2D((2, 2), strides=2)(x)
        x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = MaxPool2D((2, 2), strides=2)(x)
        x = Conv2D(128, (3, 3), activation="relu", padding="valid")(x)
        x = MaxPool2D((2, 2), strides=2)(x)
        x = Flatten()(x)
        x = Dense(64, activation="relu")(x)
        x = Dense(128, activation="relu")(x)
        out = Dense(n_classes, activation="softmax", dtype="float32")(x)
        return keras.Model(inputs=inp, outputs=out)

    def _train_new_model(self, x_train, y_train, x_test, y_test, dim, n_classes):
        model = self._build_cnn(dim, n_classes)
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.summary()

        cb = [
            ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=1, min_lr=1e-4),
            EarlyStopping(monitor="val_loss", patience=2, verbose=0, mode="auto"),
            TensorBoard(log_dir=self.log_dir),
        ]

        history = model.fit(
            x_train, y_train,
            epochs=10,
            callbacks=cb,
            validation_split=0.2,
            use_multiprocessing=True,
        )
        model.save(self.weights_path)

        for metric in ("val_accuracy", "accuracy", "val_loss", "loss"):
            print(f"  {metric}: {history.history[metric]}")

    def _load_existing_model(self):
        model = tf.keras.models.load_model(self.weights_path)
        model.summary()


# -- CLI entry point --

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train.py <True|False> <dataset_name>")
        print(f"  Available datasets: {', '.join(SUPPORTED_DATASETS)}")
        sys.exit(1)

    CharacterClassifier(dataset_key=sys.argv[2], should_train=sys.argv[1]).run()
