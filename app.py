import os
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify
from PIL import Image
from flask_cors import CORS

# Initialize the Flask app and enable CORS if needed
app = Flask(__name__)
CORS(app)

# Load your trained model (make sure the path is correct)
model = tf.keras.models.load_model('potato_disease_model.keras')
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

def preprocess_image(img):
    # Ensure the image is in RGB mode
    if img.mode != "RGB":
        img = img.convert("RGB")
    # Resize to match training dimensions
    img = img.resize((256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # Do NOT normalize here because the model already includes a rescaling layer.
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected for uploading"}), 400

    try:
        # Read the image file
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        # Preprocess the image
        img_array = preprocess_image(img)
        # Run prediction
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = round(100 * np.max(predictions[0]), 2)
        
        # Return the result as JSON
        return jsonify({
            "predicted_class": predicted_class,
            "confidence": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def index():
    return "Potato Disease Prediction API is running!"

if __name__ == "__main__":
    # Run the app on port 5000
    app.run(host="0.0.0.0", port=5000, debug=True)
