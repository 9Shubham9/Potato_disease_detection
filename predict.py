from flask import Flask, request, jsonify
from predict.py import predict_disease

app = Flask(__name__)

@app.route('/')
def home():
    return "Potato Disease Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    file_path = "temp.jpg"
    file.save(file_path)
    
    predicted_class, confidence = predict_disease(file_path)

    return jsonify({
        "prediction": predicted_class,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)
