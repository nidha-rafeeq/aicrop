import json
import os
import re
import numpy as np
from flask import Flask, render_template, request
from PIL import Image

app = Flask(__name__)
UPLOAD_DIR = os.path.join("static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
ALLOWED = {"jpg", "jpeg", "png"}
MIN_CONFIDENCE = 60  # percent

# ---- load model, class names and treatments once ----
import tensorflow as tf
model = tf.keras.models.load_model("model/crop_disease_model.keras")
with open("model/class_names.json") as f:
    class_names = json.load(f)


def normalize(name):
    """'Tomato___Early_blight' and 'Tomato Early Blight' -> 'tomatoearlyblight'"""
    return re.sub(r"[^a-z0-9]", "", name.lower())


with open("treatments.json", encoding="utf-8") as f:
    treatments = {normalize(k): v for k, v in json.load(f).items()}

DEFAULT_ADVICE = {
    "organic": "Remove affected leaves and keep the field clean.",
    "chemical": "Consult your local agricultural officer before using any pesticide.",
    "prevention": "Use healthy seeds, rotate crops and avoid water-logging.",
}


def predict(image_path):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img, dtype="float32"), axis=0)
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx]) * 100


def pretty(name):
    return re.sub(r"_+", " ", name).strip()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    file = request.files.get("image")
    if not file or file.filename == "":
        return render_template("index.html", error="Please choose a leaf photo.")
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED:
        return render_template("index.html", error="Only JPG or PNG images are allowed.")

    path = os.path.join(UPLOAD_DIR, "upload." + ext)
    file.save(path)

    label, confidence = predict(path)
    low_confidence = confidence < MIN_CONFIDENCE
    advice = treatments.get(normalize(label), DEFAULT_ADVICE)
    healthy = "healthy" in label.lower()

    return render_template(
        "result.html",
        image=path, disease=pretty(label), confidence=round(confidence, 1),
        advice=advice, low_confidence=low_confidence, healthy=healthy)


if __name__ == "__main__":
    app.run(debug=True)
