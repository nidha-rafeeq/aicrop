# Run this in Google Colab (free GPU), NOT on your laptop.
# Before running: upload the Kaggle zip to Colab and unzip it to /content/data
# It trains the disease model and saves 2 files:
#   crop_disease_model.keras  and  class_names.json
import json
import tensorflow as tf
import os

# Folder with the unzipped dataset. If it holds a single wrapper folder
# (e.g. PlantVillage/), step inside until the disease folders are found.
DATA_DIR = "/content/data"
while True:
    items = [d for d in os.listdir(DATA_DIR) if not d.startswith(".")]
    if len(items) == 1 and os.path.isdir(os.path.join(DATA_DIR, items[0])):
        DATA_DIR = os.path.join(DATA_DIR, items[0])
    else:
        break
print("Using data folder:", DATA_DIR)
print("Number of disease folders:", len(items))
IMG_SIZE = (224, 224)
BATCH = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="training",
    seed=42, image_size=IMG_SIZE, batch_size=BATCH)
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=0.2, subset="validation",
    seed=42, image_size=IMG_SIZE, batch_size=BATCH)

class_names = train_ds.class_names
print("Classes:", class_names)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# Pre-trained MobileNetV2 (transfer learning)
base = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base.trainable = False  # keep the pre-trained knowledge

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = tf.keras.layers.RandomFlip("horizontal")(inputs)
x = tf.keras.layers.RandomRotation(0.1)(x)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)  # scaling is inside the model
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(train_ds, validation_data=val_ds, epochs=8)

loss, acc = model.evaluate(val_ds)
print(f"Validation accuracy: {acc*100:.2f}%")

model.save("crop_disease_model.keras")
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

# Download both files to your computer
from google.colab import files
files.download("crop_disease_model.keras")
files.download("class_names.json")
