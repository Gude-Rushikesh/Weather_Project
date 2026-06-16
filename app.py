import base64
import io
import json
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError


ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATHS = [
    ROOT_DIR / "weather_model.keras",
    ROOT_DIR / "weather_model.h5",
]
CLASS_NAMES_PATH = ROOT_DIR / "class_names.json"
IMG_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
LOW_CONFIDENCE_THRESHOLD = 60.0

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
app.config["TEMPLATES_AUTO_RELOAD"] = True


def load_class_names():
    with CLASS_NAMES_PATH.open() as f:
        class_indices = json.load(f)
    return {int(v): k for k, v in class_indices.items()}


class LegacyBatchNormalization(tf.keras.layers.BatchNormalization):
    @classmethod
    def from_config(cls, config):
        config.pop("renorm", None)
        config.pop("renorm_clipping", None)
        config.pop("renorm_momentum", None)
        return super().from_config(config)


class LegacyDense(tf.keras.layers.Dense):
    @classmethod
    def from_config(cls, config):
        config.pop("quantization_config", None)
        return super().from_config(config)


class LegacyConv2D(tf.keras.layers.Conv2D):
    @classmethod
    def from_config(cls, config):
        config.pop("quantization_config", None)
        return super().from_config(config)


def load_weather_model():
    for path in MODEL_PATHS:
        if path.exists():
            if path.suffix == ".h5":
                return (
                    tf.keras.models.load_model(
                        path,
                        custom_objects={
                            "BatchNormalization": LegacyBatchNormalization,
                            "Dense": LegacyDense,
                            "Conv2D": LegacyConv2D,
                        },
                        compile=False,
                    ),
                    path.name,
                )
            return tf.keras.models.load_model(path, compile=False), path.name

    hf_model_path = hf_hub_download(
        repo_id="Master2316/Rushikesh-weather-vision-model",
        filename="weather_model.h5",
    )

    return (
        tf.keras.models.load_model(
            hf_model_path,
            custom_objects={
                "BatchNormalization": LegacyBatchNormalization,
                "Dense": LegacyDense,
                "Conv2D": LegacyConv2D,
            },
            compile=False,
        ),
        "weather_model.h5 (HF)",
    )


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image, dtype=np.float32)

    if USE_MOBILENET_PREPROCESSING:
        model_input = image_array
    else:
        model_input = image_array / 255.0
    return image, np.expand_dims(model_input, axis=0)


def last_visual_layer(model):
    for layer in reversed(model.layers):
        output_shape = getattr(layer.output, "shape", None)
        if output_shape is not None and len(output_shape) == 4:
            return layer.name
    return None


def build_gradcam(image, image_tensor, class_index):
    layer_name = last_visual_layer(MODEL)
    if not layer_name:
        return None

    try:
        grad_model = tf.keras.models.Model(
            MODEL.inputs,
            [MODEL.get_layer(layer_name).output, MODEL.outputs[0]],
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image_tensor)
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = np.maximum(heatmap.numpy(), 0)
        if np.max(heatmap) == 0:
            return None
        heatmap = heatmap / np.max(heatmap)

        heatmap_image = Image.fromarray(np.uint8(255 * heatmap)).resize(IMG_SIZE)
        red_layer = Image.new("RGBA", IMG_SIZE, (230, 62, 62, 0))
        alpha = heatmap_image.point(lambda value: int(value * 0.45))
        red_layer.putalpha(alpha)

        base = image.resize(IMG_SIZE).convert("RGBA")
        overlay = Image.alpha_composite(base, red_layer).convert("RGB")

        buffer = io.BytesIO()
        overlay.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return None


@app.route("/")
def index():
    return render_template("index.html", model_file=MODEL_FILE, classes=list(CLASS_NAMES.values()))


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL_FILE,
        "classes": list(CLASS_NAMES.values()),
    })


@app.route("/predict", methods=["POST"])
def predict():
    uploaded_file = request.files.get("file")
    if not uploaded_file or uploaded_file.filename == "":
        return jsonify({"error": "No image uploaded"}), 400

    if not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Upload a JPG, PNG, JPEG, or WEBP image"}), 400

    image_bytes = uploaded_file.read()
    try:
        image, image_tensor = preprocess_image(image_bytes)
    except UnidentifiedImageError:
        return jsonify({"error": "The uploaded file is not a valid image"}), 400

    probabilities = MODEL.predict(image_tensor, verbose=0)[0]
    ranked_indices = np.argsort(probabilities)[::-1]

    top_predictions = [
        {
            "class_name": CLASS_NAMES[int(index)],
            "confidence": round(float(probabilities[index]) * 100, 2),
        }
        for index in ranked_indices[:3]
    ]

    all_probabilities = {
        CLASS_NAMES[int(index)]: round(float(probabilities[index]) * 100, 2)
        for index in ranked_indices
    }
    top_prediction = top_predictions[0]
    warning = None
    if top_prediction["confidence"] < LOW_CONFIDENCE_THRESHOLD:
        warning = "Low confidence: this image may be outside the training distribution."

    gradcam = build_gradcam(image, image_tensor, ranked_indices[0])

    return jsonify({
        "prediction": top_prediction["class_name"],
        "confidence": top_prediction["confidence"],
        "top_predictions": top_predictions,
        "all_probabilities": all_probabilities,
        "warning": warning,
        "gradcam": gradcam,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Open http://localhost:{port} in your browser")
    app.run(host="0.0.0.0", port=port)
