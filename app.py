import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

MODEL_PATH = Path("artifacts/training/model.h5")
DATA_PATH = Path("artifacts/data_ingestion/Chicken-fecal-images")
IMAGE_SIZE = (224, 224)

app = Flask(__name__)
_model = None


def get_class_names() -> list:
    if not DATA_PATH.exists():
        return ["Coccidiosis", "Healthy"]

    class_names = sorted([item.name for item in DATA_PATH.iterdir() if item.is_dir()])
    return class_names if class_names else ["Coccidiosis", "Healthy"]


def load_model() -> tf.keras.Model:
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. Run `python3 main.py` first."
            )
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def preprocess_upload(file_storage) -> tf.Tensor:
    image_bytes = file_storage.read()
    image_tensor = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image_tensor = tf.image.resize(image_tensor, IMAGE_SIZE)
    image_tensor = tf.cast(image_tensor, tf.float32) / 255.0
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    return image_tensor


def predict(file_storage) -> dict:
    model = load_model()
    class_names = get_class_names()
    processed_image = preprocess_upload(file_storage)

    probabilities = model.predict(processed_image, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])
    predicted_class = (
        class_names[predicted_index]
        if predicted_index < len(class_names)
        else str(predicted_index)
    )

    return {
        "class": predicted_class,
        "confidence": round(confidence, 4),
    }


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        error = "No file uploaded. Please choose an image first."
        if request.accept_mimetypes.best == "application/json":
            return jsonify({"error": error}), 400
        return render_template("index.html", error=error), 400

    image_file = request.files["image"]
    if image_file.filename == "":
        error = "Selected file is empty. Please choose a valid image."
        if request.accept_mimetypes.best == "application/json":
            return jsonify({"error": error}), 400
        return render_template("index.html", error=error), 400

    try:
        result = predict(image_file)
    except Exception as exc:
        error = str(exc)
        if request.accept_mimetypes.best == "application/json":
            return jsonify({"error": error}), 500
        return render_template("index.html", error=error), 500

    if request.accept_mimetypes.best == "application/json":
        return jsonify(result)
    return render_template("index.html", result=result)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)
