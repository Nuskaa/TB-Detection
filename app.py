import os
import pickle
import numpy as np
import cv2
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Load models safely at startup
try:
    model = pickle.load(open('model.pkl', 'rb'))
    model_lr = pickle.load(open('model_lr.pkl', 'rb'))
    model_svm = pickle.load(open('model_svm.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    print("All models loaded successfully.")
except Exception as e:
    print(f"Model loading error: {e}")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Read image from memory (no disk write needed)
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return jsonify({'error': 'Invalid or unreadable image'}), 400

    # Preprocess
    img_resized = cv2.resize(img, (64, 64))
    img_flat = img_resized.flatten().reshape(1, -1)
    img_scaled = scaler.transform(img_flat)

    # Predict
    prediction = model.predict(img_scaled)[0]
    result = "TB Detected" if prediction == 1 else "Normal"

    return render_template('index.html', result=result)


# Required for Render: bind to 0.0.0.0 and read PORT from environment
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)