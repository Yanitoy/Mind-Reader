# ==============================
# Imports
# ==============================
import math
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# Config
# ==============================
SEED = 42
IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 20

# Choose one:
DATA_SOURCE = "csv"  # "csv" or "folders"

# CSV example (FER2013-style)
CSV_PATH = "/kaggle/input/fer2013/fer2013.csv"

# Folder example (structure: IMAGE_DIR/class_name/*.png|jpg)
IMAGE_DIR = "/kaggle/input/fer2013-images"

# Match this order to your dataset labels.
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

np.random.seed(SEED)
tf.random.set_seed(SEED)
AUTOTUNE = tf.data.AUTOTUNE

# ==============================
# Data Loading + Preprocessing
# ==============================
if DATA_SOURCE == "csv":
    df = pd.read_csv(CSV_PATH)
    if "pixels" not in df.columns or "emotion" not in df.columns:
        raise ValueError("CSV must have 'pixels' and 'emotion' columns.")

    pixel_lists = df["pixels"].str.split().tolist()
    flat_len = len(pixel_lists[0])
    side = int(math.sqrt(flat_len))
    images = np.array(pixel_lists, dtype=np.float32).reshape(-1, side, side, 1)

    if side != IMG_SIZE:
        images = tf.image.resize(images, (IMG_SIZE, IMG_SIZE)).numpy()

    images = images / 255.0

    labels_raw = df["emotion"].values
    if labels_raw.dtype.kind in {"U", "S", "O"}:
        label_names = sorted(np.unique(labels_raw))
        label_to_index = {name: idx for idx, name in enumerate(label_names)}
        labels = np.array([label_to_index[x] for x in labels_raw], dtype=np.int32)
    else:
        labels = labels_raw.astype(np.int32)
        label_names = EMOTION_LABELS

    num_classes = len(label_names)

    X_train, X_temp, y_train_raw, y_temp_raw = train_test_split(
        images,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=SEED,
    )
    X_val, X_test, y_val_raw, y_test_raw = train_test_split(
        X_temp,
        y_temp_raw,
        test_size=0.5,
        stratify=y_temp_raw,
        random_state=SEED,
    )

    y_train = tf.keras.utils.to_categorical(y_train_raw, num_classes)
    y_val = tf.keras.utils.to_categorical(y_val_raw, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test_raw, num_classes)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    train_ds = train_ds.shuffle(2000, seed=SEED).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

elif DATA_SOURCE == "folders":
    data_dir = os.path.abspath(IMAGE_DIR)
    if not os.path.isdir(data_dir):
        raise ValueError(f"Folder not found: {data_dir}")

    image_paths = []
    labels_raw = []
    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for filename in os.listdir(class_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(os.path.join(class_dir, filename))
                labels_raw.append(class_name)

    label_names = sorted(set(labels_raw))
    label_to_index = {name: idx for idx, name in enumerate(label_names)}
    labels = np.array([label_to_index[name] for name in labels_raw], dtype=np.int32)
    num_classes = len(label_names)

    image_paths = np.array(image_paths)
    indices = np.arange(len(image_paths))
    np.random.shuffle(indices)
    image_paths = image_paths[indices]
    labels = labels[indices]

    train_end = int(0.8 * len(image_paths))
    val_end = int(0.9 * len(image_paths))

    train_paths, val_paths, test_paths = (
        image_paths[:train_end],
        image_paths[train_end:val_end],
        image_paths[val_end:],
    )
    train_labels, val_labels, test_labels = (
        labels[:train_end],
        labels[train_end:val_end],
        labels[val_end:],
    )

    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=1, expand_animations=False)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = tf.cast(img, tf.float32) / 255.0
        label = tf.one_hot(label, num_classes)
        return img, label

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))

    train_ds = (
        train_ds.shuffle(2000, seed=SEED)
        .map(load_image, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )
    val_ds = val_ds.map(load_image, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    test_ds = test_ds.map(load_image, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE).prefetch(AUTOTUNE)

else:
    raise ValueError("DATA_SOURCE must be 'csv' or 'folders'.")

# ==============================
# Model
# ==============================
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ==============================
# Training
# ==============================
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

plt.figure(figsize=(8, 4))
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# ==============================
# Evaluation
# ==============================
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in test_ds], axis=0)
y_pred = np.argmax(model.predict(test_ds), axis=1)

print(classification_report(y_true, y_pred, target_names=label_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# ==============================
# Saving
# ==============================
model.save("expression_model.h5")
print("Saved model to /kaggle/working/expression_model.h5")

# ==============================
# Export Instructions (readme-style notes)
# ==============================
# 1) In Kaggle: open the "Output" panel and download expression_model.h5
#    (it is saved under /kaggle/working/).
#
# 2) Convert to TensorFlow.js on your local machine:
#    pip install tensorflowjs
#    tensorflowjs_converter --input_format=keras expression_model.h5 frontend/public/web_model
#
# 3) Expected output:
#    frontend/public/web_model/
#      model.json
#      group1-shard1ofX.bin
#
# Keep EMOTION_LABELS in frontend/src/App.jsx in the same order as label_names.
