import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, render_template
import random
import time

# Initialize Flask app
app = Flask(__name__)

# Paths
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
class_indices_path = f"{working_dir}/class_indices.json"

# Load model & class indices
model = tf.keras.models.load_model(model_path)
with open(class_indices_path, "r") as f:
    class_indices = json.load(f)

# --- Helper Functions ---
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = Image.open(image)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype("float32") / 255.0
    return img_array

def predict_image_class(model, image, class_indices, top_k=3):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)[0]  # single image -> 1D array

    # Get top-k predictions
    top_indices = predictions.argsort()[-top_k:][::-1]
    results = []
    for idx in top_indices:
        class_name = class_indices[str(idx)]
        confidence = float(predictions[idx]) * 100
        results.append({"class": class_name, "confidence": round(confidence, 2)})
    return results

# Fun facts dictionary
fun_facts = {
    "Apple Scab": "🍏 Did you know? Apple scab is caused by the fungus Venturia inaequalis and thrives in humid environments.",
    "Powdery Mildew": "🌸 Fun fact: Powdery mildew can actually survive even without free water on leaves!",
    "Tomato Late Blight": "🍅 The Irish Potato Famine in the 1840s was largely caused by late blight.",
    "Healthy": "🌱 Your plant looks healthy! Keep it happy with good sunlight and water balance."
}

# --- Routes ---
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("result.html", error="No file uploaded")

    file = request.files["file"]
    if file.filename == "":
        return render_template("result.html", error="Empty filename")

    try:
        # Ensure 'static' folder exists
        static_dir = os.path.join(working_dir, "static")
        os.makedirs(static_dir, exist_ok=True)

        # Create unique filename
        filename = f"{int(time.time())}_{file.filename}"
        image_path = os.path.join(static_dir, filename)

        # Save file
        file.seek(0)
        file.save(image_path)

        # Run prediction
        predictions = predict_image_class(model, image_path, class_indices, top_k=3)
        top_prediction = predictions[0]["class"]

        fact = fun_facts.get(
            top_prediction,
            "🌿 Plants are amazing! Did you know some can communicate using chemical signals?"
        )

        # Pass relative path to template
        image_relative_path = os.path.join("static", filename)

        return render_template(
            "result.html",
            predictions=predictions,
            image_path=image_relative_path,
            fun_fact=fact
        )

    except Exception as e:
        return render_template("result.html", error=str(e))

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
