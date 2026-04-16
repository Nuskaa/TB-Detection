"""
=============================================================
  TB DETECTION — FLASK APP (PRODUCTION-READY FOR RENDER)
=============================================================
Changes made for deployment:
  1. host="0.0.0.0" so Render can expose the port
  2. Port read from environment variable (Render sets PORT)
  3. debug=False in production
  4. Model path uses os.path so it works on any OS/server
  5. MAX_CONTENT_LENGTH set to prevent abuse
=============================================================
"""

import os
import io
import base64
import pickle
import traceback
from datetime import datetime

import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")

# Max upload size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
IMG_SIZE = (64, 64)

# ─────────────────────────────────────────────
# MODEL LOADING
# Use os.path.join so paths work on Linux (Render uses Linux)
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

def load_model():
    """Load model and scaler from disk. Called once at startup."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"model.pkl not found at {MODEL_PATH}\n"
            "Make sure you committed model.pkl to your GitHub repo."
        )
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"scaler.pkl not found at {SCALER_PATH}\n"
            "Make sure you committed scaler.pkl to your GitHub repo."
        )
    with open(MODEL_PATH,  "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    print(f"✅ Model loaded  : {type(model).__name__}")
    print(f"✅ Scaler loaded : {type(scaler).__name__}")
    return model, scaler

model, scaler = load_model()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess(file_bytes):
    """Raw image bytes → scaled feature vector (same as training)."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read the uploaded image.")
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE)
    flat    = resized.flatten().astype(np.float32)
    scaled  = scaler.transform(flat.reshape(1, -1))
    return scaled

def to_base64(file_bytes):
    return base64.b64encode(file_bytes).decode("utf-8")

def get_severity(label, confidence):
    if label == "TB":
        if confidence >= 85: return "Critical Risk", "critical"
        if confidence >= 70: return "High Risk",     "high"
        if confidence >= 55: return "Moderate Risk", "moderate"
        return "Low Risk", "low"
    else:
        if confidence >= 85: return "Clear",         "clear"
        if confidence >= 70: return "Very Low Risk", "very-low"
        return "Low Risk", "low"

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/home")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Validate
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded."}), 400
    file = request.files["file"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"success": False,
                        "error": "Invalid file. Upload a JPG or PNG."}), 400
    try:
        file_bytes = file.read()
        if not file_bytes:
            return jsonify({"success": False, "error": "Empty file."}), 400

        # Predict
        features   = preprocess(file_bytes)
        pred_class = model.predict(features)[0]
        proba      = model.predict_proba(features)[0]
        label      = "TB" if pred_class == 1 else "Normal"
        confidence = round(float(max(proba)) * 100, 2)
        severity, severity_key = get_severity(label, confidence)

        # Image for display
        img_b64 = to_base64(file_bytes)

        return jsonify({
            "success"     : True,
            "prediction"  : label,
            "confidence"  : confidence,
            "severity"    : severity,
            "severity_key": severity_key,
            "image_b64"   : img_b64,
            "timestamp"   : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filename"    : file.filename,
        })

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 422
    except Exception:
        traceback.print_exc()
        return jsonify({"success": False,
                        "error": "Server error. Please try again."}), 500

# Health-check endpoint — Render pings this to verify the app is alive
@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": type(model).__name__}), 200

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Render sets the PORT environment variable automatically
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") == "development"
    print(f"\n🚀 Starting TB Detection App on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)