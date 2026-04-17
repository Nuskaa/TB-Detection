"""
=============================================================
  TB DETECTION — FIXED app.py
  Works with the enhanced index.html (tabs, edge overlay,
  pipeline viewer, dual model, severity, history)
=============================================================
"""

import os
import io
import base64
import pickle
import json
import traceback
from datetime import datetime

import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template, send_file

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024   # 10 MB max upload

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
IMG_SIZE           = (64, 64)
HISTORY_FILE       = "prediction_history.json"

# ─────────────────────────────────────────────
# MODEL LOADING  (absolute paths so it works everywhere)
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_pkl(filename):
    """Load a pickle file. Returns None if file doesn't exist."""
    path = os.path.join(BASE_DIR, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

print("Loading models …")
model_svm = _load_pkl("model_svm.pkl") or _load_pkl("model.pkl")
model_lr  = _load_pkl("model_lr.pkl")
scaler    = _load_pkl("scaler.pkl")

if model_svm is None:
    raise FileNotFoundError("❌  No SVM model found. Run train_model.py first.")
if scaler is None:
    raise FileNotFoundError("❌  scaler.pkl not found. Run train_model.py first.")

print(f"✅  SVM   : {type(model_svm).__name__}")
print(f"{'✅' if model_lr else '⚠️ '}  LR    : {type(model_lr).__name__ if model_lr else 'not found (re-run train_model.py)'}")
print(f"✅  Scaler: {type(scaler).__name__}")

# In-memory store for last prediction result (used by /report)
_last_result = {}

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def decode_image(file_bytes):
    """Bytes  →  (BGR image, grayscale image)"""
    arr  = np.frombuffer(file_bytes, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode the image. Try a different file.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def extract_features(gray):
    """Grayscale image  →  scaled 1-D feature vector  (1 × 4096)"""
    resized = cv2.resize(gray, IMG_SIZE)
    flat    = resized.flatten().astype(np.float32)
    return scaler.transform(flat.reshape(1, -1))


def encode_img(img, quality=88):
    """cv2 image (BGR or gray)  →  base64 JPEG string"""
    if len(img.shape) == 2:                               # grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    _, buf = cv2.imencode(".jpg", img,
                          [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()


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


def get_image_stats(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return {
        "brightness": round(float(np.mean(gray)), 1),
        "contrast":   round(float(np.std(gray)), 1),
        "sharpness":  round(float(min(lap.var(), 9999.0)), 1),
    }


def build_pipeline(img_bgr, img_gray):
    """Return 4 preprocessing steps as base64 images for the Pipeline tab."""
    steps = []

    # ① Original (resized to 256 for display)
    steps.append({
        "label": "① Original",
        "desc":  "Raw uploaded image",
        "b64":   encode_img(cv2.resize(img_bgr, (256, 256))),
    })

    # ② Grayscale
    gray_256 = cv2.resize(img_gray, (256, 256))
    steps.append({
        "label": "② Grayscale",
        "desc":  "cv2.COLOR_BGR2GRAY",
        "b64":   encode_img(gray_256),
    })

    # ③ 64×64 (upscaled with nearest-neighbor so pixelation is visible)
    small    = cv2.resize(img_gray, IMG_SIZE)
    small_up = cv2.resize(small, (256, 256), interpolation=cv2.INTER_NEAREST)
    steps.append({
        "label": "③ Resized 64×64",
        "desc":  "Model input resolution",
        "b64":   encode_img(small_up),
    })

    # ④ Canny edge map
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)
    edges    = cv2.Canny(enhanced, 30, 100)
    steps.append({
        "label": "④ Edge Map",
        "desc":  "Canny edge detection",
        "b64":   encode_img(cv2.resize(edges, (256, 256))),
    })
    return steps


def build_edge_overlay(img_bgr, img_gray):
    """Teal Canny-edge overlay on the original X-ray → base64 JPEG."""
    img_512  = cv2.resize(img_bgr,  (512, 512))
    gray_512 = cv2.resize(img_gray, (512, 512))

    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_512)
    edges    = cv2.Canny(enhanced, 25, 90)

    kernel = np.ones((2, 2), np.uint8)
    edges  = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    overlay = img_512.copy()
    cv2.drawContours(overlay, contours, -1, (0, 221, 180), 1)
    overlay[edges > 0] = [0, 221, 180]          # teal highlights

    result = cv2.addWeighted(img_512, 0.65, overlay, 0.35, 0)
    return encode_img(result, quality=90)


# ─────────────────────────────────────────────
# PREDICTION HISTORY  (stored in a local JSON file)
# ─────────────────────────────────────────────
def load_history():
    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except Exception:
        return []

def save_history(entry):
    hist = load_history()
    hist.insert(0, entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(hist[:25], f, indent=2)   # keep last 25 records


# ─────────────────────────────────────────────
# PDF REPORT
# ─────────────────────────────────────────────
def generate_pdf(data):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph,
                                    Spacer, Table, TableStyle,
                                    HRFlowable)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units  import inch, cm

    buf  = io.BytesIO()
    doc  = SimpleDocTemplate(buf, pagesize=A4,
                             leftMargin=2*cm, rightMargin=2*cm,
                             topMargin=2*cm,  bottomMargin=2*cm)
    st    = getSampleStyleSheet()
    story = []
    is_tb = data.get("prediction") == "TB"
    pc    = colors.HexColor("#dc2626" if is_tb else "#16a34a")

    story.append(Paragraph("TB Detection Report",
        ParagraphStyle("H", parent=st["Title"], fontSize=24,
                       textColor=colors.HexColor("#0f172a"), spaceAfter=4)))
    story.append(Paragraph(
        "AI-Assisted Chest X-Ray Analysis · Scientific Python Mini Project",
        ParagraphStyle("S", parent=st["Normal"], fontSize=9,
                       textColor=colors.HexColor("#64748b"), spaceAfter=10)))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=colors.HexColor("#0ea5e9")))
    story.append(Spacer(1, 0.2*inch))

    # Info table
    rows = [
        ["Report Date",    data.get("timestamp", "N/A")],
        ["File Name",      data.get("filename",  "N/A")],
        ["Classification", data.get("prediction","N/A")],
        ["Confidence",     f"{data.get('confidence', 0):.1f}%"],
        ["Severity",       data.get("severity",  "N/A")],
    ]
    t = Table(rows, colWidths=[5*cm, 11*cm])
    t.setStyle(TableStyle([
        ("FONTNAME",      (0,0),(0,-1), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 10),
        ("BACKGROUND",    (0,0),(-1,0), colors.HexColor("#f1f5f9")),
        ("BACKGROUND",    (0,2),(-1,2), colors.HexColor("#f1f5f9")),
        ("BACKGROUND",    (0,4),(-1,4), colors.HexColor("#f1f5f9")),
        ("GRID",          (0,0),(-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ("TOPPADDING",    (0,0),(-1,-1), 7),
        ("BOTTOMPADDING", (0,0),(-1,-1), 7),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
        ("TEXTCOLOR",     (1,2),(1,2), pc),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.25*inch))

    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#e2e8f0")))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(
        "⚠ DISCLAIMER: This report is for EDUCATIONAL PURPOSES ONLY and does not "
        "constitute medical advice. Consult a qualified physician for any health decisions.",
        ParagraphStyle("D", parent=st["Normal"], fontSize=8,
                       textColor=colors.HexColor("#94a3b8"),
                       fontName="Helvetica-Oblique", leading=12)))
    doc.build(story)
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    global _last_result

    # ── Validation ───────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({"success": False,
                        "error": "No file uploaded."}), 400

    file = request.files["file"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"success": False,
                        "error": "Invalid file. Upload a JPG or PNG."}), 400

    try:
        file_bytes = file.read()
        if not file_bytes:
            return jsonify({"success": False,
                            "error": "Uploaded file is empty."}), 400

        # ── Decode ────────────────────────────────────────────
        img_bgr, img_gray = decode_image(file_bytes)

        # ── Features ──────────────────────────────────────────
        features = extract_features(img_gray)

        # ── SVM prediction ────────────────────────────────────
        svm_class = model_svm.predict(features)[0]
        svm_proba = model_svm.predict_proba(features)[0]
        svm_label = "TB" if svm_class == 1 else "Normal"
        svm_conf  = round(float(max(svm_proba)) * 100, 2)

        # ── LR prediction (optional) ──────────────────────────
        lr_label, lr_conf = None, None
        if model_lr is not None:
            lr_class = model_lr.predict(features)[0]
            lr_proba = model_lr.predict_proba(features)[0]
            lr_label = "TB" if lr_class == 1 else "Normal"
            lr_conf  = round(float(max(lr_proba)) * 100, 2)

        # Primary result = SVM
        label      = svm_label
        confidence = svm_conf
        severity, severity_key = get_severity(label, confidence)

        # ── Build visuals ─────────────────────────────────────
        img_512   = cv2.resize(img_bgr, (512, 512))
        image_b64 = encode_img(img_512, quality=90)
        edge_b64  = build_edge_overlay(img_bgr, img_gray)
        pipeline  = build_pipeline(img_bgr, img_gray)
        stats     = get_image_stats(img_gray)
        ts        = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ── Build response ────────────────────────────────────
        result = {
            "success"      : True,
            "prediction"   : label,
            "confidence"   : confidence,
            "severity"     : severity,
            "severity_key" : severity_key,
            "image_b64"    : image_b64,
            "edge_b64"     : edge_b64,
            "pipeline"     : pipeline,
            "svm"          : {"prediction": svm_label, "confidence": svm_conf},
            "lr"           : {"prediction": lr_label,  "confidence": lr_conf},
            "image_stats"  : stats,
            "timestamp"    : ts,
            "filename"     : file.filename,
        }

        # Save last result for PDF (without heavy pipeline data)
        _last_result = {k: v for k, v in result.items() if k != "pipeline"}

        # Persist to history file
        save_history({
            "timestamp" : ts,
            "filename"  : file.filename,
            "prediction": label,
            "confidence": confidence,
            "severity"  : severity,
        })

        return jsonify(result)

    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 422
    except Exception:
        traceback.print_exc()
        return jsonify({"success": False,
                        "error": "Server error. Check terminal for details."}), 500


@app.route("/report")
def report():
    if not _last_result:
        return "No prediction yet. Run a prediction first.", 400
    try:
        buf   = generate_pdf(_last_result)
        fname = f"TB_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        return send_file(buf, mimetype="application/pdf",
                         as_attachment=True, download_name=fname)
    except Exception:
        traceback.print_exc()
        return ("PDF generation failed. "
                "Run:  pip install reportlab  then restart app.py"), 500


@app.route("/history")
def history():
    return jsonify(load_history())


@app.route("/history/clear", methods=["POST"])
def clear_history():
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)
    return jsonify({"success": True})


@app.route("/health")
def health():
    return jsonify({
        "status"    : "ok",
        "svm_model" : type(model_svm).__name__,
        "lr_model"  : type(model_lr).__name__ if model_lr else "not loaded",
    }), 200


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    print(f"\n{'='*50}")
    print(f"  TB Detection App  →  http://127.0.0.1:{port}")
    print(f"{'='*50}\n")
    app.run(host="0.0.0.0", port=port, debug=True)