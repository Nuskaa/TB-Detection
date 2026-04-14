"""
=============================================================
  TB DETECTION — FLASK BACKEND (ENHANCED)
  New features:
    • Dual model (SVM + Logistic Regression)
    • Lung edge overlay (Canny + CLAHE)
    • Preprocessing pipeline (4 steps)
    • Image quality stats (brightness, contrast, sharpness)
    • Severity scoring (Critical/High/Moderate/Low)
    • PDF report generation (reportlab)
    • Prediction history (JSON file persistence)
  Routes:
    GET  /             → Homepage
    POST /predict      → Full analysis, returns JSON
    GET  /report       → Download PDF of last prediction
    GET  /history      → Return history JSON
    POST /history/clear → Clear history
=============================================================
"""

import os
import io
import base64
import pickle
import json
import traceback
import tempfile
from datetime import datetime

import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template, send_file

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
IMG_SIZE     = (64, 64)
HISTORY_FILE = "prediction_history.json"

# ─────────────────────────────────────────────
# LOAD MODELS AT STARTUP
# ─────────────────────────────────────────────
def _load(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

# Try specific names first, fall back to legacy model.pkl for SVM
model_svm = _load("model_svm.pkl") or _load("model.pkl")
model_lr  = _load("model_lr.pkl")
scaler    = _load("scaler.pkl")

if model_svm is None:
    raise FileNotFoundError("No SVM model found. Run train_model.py first.")
if scaler is None:
    raise FileNotFoundError("scaler.pkl not found. Run train_model.py first.")

# In-memory store for last prediction (used by PDF route)
_last = {}

# ─────────────────────────────────────────────
# HISTORY HELPERS
# ─────────────────────────────────────────────
def load_history():
    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except Exception:
        return []

def append_history(entry):
    h = load_history()
    h.insert(0, entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(h[:25], f, indent=2)  # keep last 25


# ─────────────────────────────────────────────
# IMAGE PROCESSING HELPERS
# ─────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _encode(img_bgr_or_gray, quality=88):
    """Encode a cv2 image (BGR or grayscale) to base64 JPEG string."""
    # Convert grayscale to BGR so imencode works uniformly
    if len(img_bgr_or_gray.shape) == 2:
        img_bgr_or_gray = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_GRAY2BGR)
    _, buf = cv2.imencode(".jpg", img_bgr_or_gray,
                          [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()


def decode_image(file_bytes):
    """Bytes → (BGR image, grayscale image). Raises ValueError on failure."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode the uploaded image.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def extract_features(gray):
    """Grayscale image → scaled feature vector (1, 4096)."""
    resized = cv2.resize(gray, IMG_SIZE)
    flat    = resized.flatten().astype(np.float32)
    return scaler.transform(flat.reshape(1, -1))


def get_image_stats(gray):
    """Return brightness, contrast, sharpness of a grayscale image."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return {
        "brightness": round(float(np.mean(gray)),   1),
        "contrast":   round(float(np.std(gray)),    1),
        "sharpness":  round(float(min(lap.var(), 9999.0)), 1),
    }


def get_severity(label, confidence):
    """
    Map prediction + confidence → human-readable severity + CSS key.
    """
    if label == "TB":
        if confidence >= 85: return "Critical Risk", "critical"
        if confidence >= 70: return "High Risk",     "high"
        if confidence >= 55: return "Moderate Risk", "moderate"
        return "Low Risk", "low"
    else:  # Normal
        if confidence >= 85: return "Clear",         "clear"
        if confidence >= 70: return "Very Low Risk", "very-low"
        return "Low Risk", "low"


def build_pipeline_steps(img_bgr, img_gray):
    """
    Return 4 preprocessing steps as base64 images.
    Each image is shown at 256×256 for display.
    """
    steps = []

    # ① Original
    steps.append({
        "label": "① Original",
        "desc":  "Raw uploaded image",
        "b64":   _encode(cv2.resize(img_bgr, (256, 256))),
    })

    # ② Grayscale
    gray_256 = cv2.resize(img_gray, (256, 256))
    steps.append({
        "label": "② Grayscale",
        "desc":  "cv2.COLOR_BGR2GRAY",
        "b64":   _encode(gray_256),
    })

    # ③ Resized to 64×64 (shown upscaled with nearest-neighbor so pixelation is visible)
    small    = cv2.resize(img_gray, IMG_SIZE)
    small_up = cv2.resize(small, (256, 256), interpolation=cv2.INTER_NEAREST)
    steps.append({
        "label": "③ Resized 64×64",
        "desc":  "Model input resolution",
        "b64":   _encode(small_up),
    })

    # ④ Canny edge map
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)
    edges   = cv2.Canny(enhanced, 30, 100)
    edges_256 = cv2.resize(edges, (256, 256))
    steps.append({
        "label": "④ Edge Map",
        "desc":  "Canny edge detection",
        "b64":   _encode(edges_256),
    })

    return steps


def build_edge_overlay(img_bgr, img_gray):
    """
    CLAHE-enhanced Canny edge detection overlaid on the original X-ray.
    Edges are highlighted in teal (#00DDB4).
    Returns base64 JPEG string at 512×512.
    """
    # Resize both to 512 for display quality
    img_512  = cv2.resize(img_bgr,  (512, 512))
    gray_512 = cv2.resize(img_gray, (512, 512))

    # CLAHE: improves contrast in dark lung regions → better edges
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_512)

    # Canny detection
    edges = cv2.Canny(enhanced, 25, 90)

    # Slight dilation so edges are visible on screen
    kernel = np.ones((2, 2), np.uint8)
    edges  = cv2.dilate(edges, kernel, iterations=1)

    # Find contours (lung boundaries)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw on overlay
    overlay = img_512.copy()
    cv2.drawContours(overlay, contours, -1, (0, 221, 180), 1)
    overlay[edges > 0] = [0, 221, 180]   # teal pixel highlight

    # Blend: 65% original + 35% overlay
    result = cv2.addWeighted(img_512, 0.65, overlay, 0.35, 0)
    return _encode(result, quality=90)


# ─────────────────────────────────────────────
# PDF REPORT GENERATOR
# ─────────────────────────────────────────────
def generate_pdf(data):
    """
    Build a professional A4 PDF report using reportlab.
    Returns a BytesIO buffer ready for send_file().
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer,
        Table, TableStyle, HRFlowable,
        Image as RLImage,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units  import inch, cm
    from reportlab.lib.enums  import TA_CENTER

    buf  = io.BytesIO()
    doc  = SimpleDocTemplate(buf, pagesize=A4,
                             leftMargin=2*cm, rightMargin=2*cm,
                             topMargin=2*cm,  bottomMargin=2*cm)
    st   = getSampleStyleSheet()
    story = []

    is_tb      = data.get("prediction") == "TB"
    pred_color = colors.HexColor("#dc2626" if is_tb else "#16a34a")

    # ── Header ──────────────────────────────────────────
    story.append(Paragraph(
        "TB Detection Report",
        ParagraphStyle("H", parent=st["Title"], fontSize=26,
                       textColor=colors.HexColor("#0f172a"), spaceAfter=4),
    ))
    story.append(Paragraph(
        "AI-Assisted Chest X-Ray Analysis · Scientific Python Mini Project",
        ParagraphStyle("S", parent=st["Normal"], fontSize=10,
                       textColor=colors.HexColor("#64748b"), spaceAfter=10),
    ))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=colors.HexColor("#0ea5e9")))
    story.append(Spacer(1, 0.2*inch))

    # ── Scan info table ──────────────────────────────────
    story.append(Paragraph("Scan Information", st["Heading2"]))
    info = Table([
        ["Report Date",   data.get("timestamp", "N/A")],
        ["File Name",     data.get("filename",  "N/A")],
        ["Primary Model", "Support Vector Machine — RBF Kernel"],
        ["Input Size",    f"64 × 64 px → 4,096 features"],
    ], colWidths=[5*cm, 11*cm])
    info.setStyle(TableStyle([
        ("FONTNAME",       (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE",       (0,0), (-1,-1), 10),
        ("BACKGROUND",     (0,0), (-1,0), colors.HexColor("#f1f5f9")),
        ("BACKGROUND",     (0,2), (-1,2), colors.HexColor("#f1f5f9")),
        ("GRID",           (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ("TOPPADDING",     (0,0), (-1,-1), 7),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 7),
        ("LEFTPADDING",    (0,0), (-1,-1), 8),
    ]))
    story.append(info)
    story.append(Spacer(1, 0.2*inch))

    # ── X-ray thumbnail ──────────────────────────────────
    img_b64 = data.get("image_b64", "")
    if img_b64:
        try:
            raw = base64.b64decode(img_b64)
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp.write(raw); tmp.close()
            story.append(Paragraph("X-Ray Image", st["Heading2"]))
            story.append(RLImage(tmp.name, width=3.2*inch, height=3.2*inch))
            os.unlink(tmp.name)
            story.append(Spacer(1, 0.2*inch))
        except Exception:
            pass

    # ── Prediction result ────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#e2e8f0")))
    story.append(Spacer(1, 0.12*inch))
    story.append(Paragraph("Diagnosis Result", st["Heading2"]))
    story.append(Paragraph(
        f"● {data.get('prediction','Unknown')}",
        ParagraphStyle("PR", parent=st["Normal"], fontSize=22,
                       textColor=pred_color, fontName="Helvetica-Bold",
                       spaceAfter=8),
    ))

    pred = data.get("prediction", "—")
    conf = data.get("confidence", 0)
    sev  = data.get("severity",   "—")
    svm  = data.get("svm", {})
    lr   = data.get("lr",  {})

    rows = [
        ["Classification", pred,                        "Confidence",   f"{conf:.1f}%"],
        ["Severity",       sev,                         "SVM Result",   f"{svm.get('prediction','N/A')} ({svm.get('confidence',0):.1f}%)"],
    ]
    if lr and lr.get("prediction"):
        rows.append(["LR Result",
                     f"{lr['prediction']} ({lr['confidence']:.1f}%)",
                     "Agreement",
                     "✓ Yes" if lr["prediction"] == pred else "✗ No"])

    res_table = Table(rows, colWidths=[4*cm, 5.5*cm, 4*cm, 2.5*cm])
    res_table.setStyle(TableStyle([
        ("FONTNAME",      (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME",      (2,0), (2,-1), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 10),
        ("BACKGROUND",    (0,0), (-1,0), colors.HexColor("#f8fafc")),
        ("GRID",          (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
    ]))
    story.append(res_table)
    story.append(Spacer(1, 0.2*inch))

    # ── Image quality stats ──────────────────────────────
    stats = data.get("image_stats", {})
    if stats:
        story.append(Paragraph("Image Quality Metrics", st["Heading2"]))
        sq = Table([
            ["Brightness",  f"{stats.get('brightness',0):.1f} / 255",
             "Contrast",    f"{stats.get('contrast',0):.1f}"],
            ["Sharpness",   f"{stats.get('sharpness',0):.1f}",  "", ""],
        ], colWidths=[4*cm, 5.5*cm, 4*cm, 2.5*cm])
        sq.setStyle(TableStyle([
            ("FONTNAME",      (0,0), (0,-1), "Helvetica-Bold"),
            ("FONTNAME",      (2,0), (2,-1), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 10),
            ("GRID",          (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
            ("TOPPADDING",    (0,0), (-1,-1), 7),
            ("BOTTOMPADDING", (0,0), (-1,-1), 7),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ]))
        story.append(sq)
        story.append(Spacer(1, 0.2*inch))

    # ── Disclaimer ───────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#e2e8f0")))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(
        "⚠ DISCLAIMER: This report is generated for EDUCATIONAL PURPOSES ONLY and "
        "does not constitute medical advice, clinical diagnosis, or treatment. "
        "Please consult a qualified medical professional for any health-related decisions. "
        "Developed as part of a Scientific Python course mini project using Scikit-learn.",
        ParagraphStyle("Disc", parent=st["Normal"], fontSize=8,
                       textColor=colors.HexColor("#94a3b8"),
                       fontName="Helvetica-Oblique", leading=12),
    ))

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
    global _last

    # ── Validate upload ──────────────────────────────────
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded."}), 400

    file = request.files["file"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"success": False,
                        "error": "Invalid file type. Upload JPG or PNG."}), 400

    try:
        file_bytes = file.read()
        if not file_bytes:
            return jsonify({"success": False, "error": "Uploaded file is empty."}), 400

        # ── Decode ───────────────────────────────────────
        img_bgr, img_gray = decode_image(file_bytes)

        # ── Feature extraction ───────────────────────────
        features = extract_features(img_gray)

        # ── SVM prediction ───────────────────────────────
        svm_class = model_svm.predict(features)[0]
        svm_proba = model_svm.predict_proba(features)[0]
        svm_label = "TB" if svm_class == 1 else "Normal"
        svm_conf  = round(float(max(svm_proba)) * 100, 2)

        # ── LR prediction (optional) ─────────────────────
        lr_label, lr_conf = None, None
        if model_lr is not None:
            lr_class  = model_lr.predict(features)[0]
            lr_proba  = model_lr.predict_proba(features)[0]
            lr_label  = "TB" if lr_class == 1 else "Normal"
            lr_conf   = round(float(max(lr_proba)) * 100, 2)

        # Primary result = SVM
        label      = svm_label
        confidence = svm_conf
        severity, severity_key = get_severity(label, confidence)

        # ── Visuals ──────────────────────────────────────
        # Main display image (512×512)
        img_512   = cv2.resize(img_bgr, (512, 512))
        image_b64 = _encode(img_512, quality=90)

        # Edge overlay
        edge_b64 = build_edge_overlay(img_bgr, img_gray)

        # Preprocessing pipeline
        pipeline = build_pipeline_steps(img_bgr, img_gray)

        # Image quality stats
        stats = get_image_stats(img_gray)

        # Timestamp
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ── Build response ────────────────────────────────
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

        # Store for PDF (without heavy pipeline images)
        _last = {k: v for k, v in result.items() if k != "pipeline"}

        # Persist history
        append_history({
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
                        "error": "Server error. Please try again."}), 500


@app.route("/report", methods=["GET"])
def report():
    """Generate and return a PDF report for the last prediction."""
    if not _last:
        return "No prediction yet. Upload an X-ray first.", 400
    try:
        pdf_buf = generate_pdf(_last)
        fname   = f"TB_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        return send_file(pdf_buf, mimetype="application/pdf",
                         as_attachment=True, download_name=fname)
    except Exception:
        traceback.print_exc()
        return "PDF generation failed. Make sure reportlab is installed.", 500


@app.route("/history", methods=["GET"])
def history():
    return jsonify(load_history())


@app.route("/history/clear", methods=["POST"])
def clear_history():
    with open(HISTORY_FILE, "w") as f:
        json.dump([], f)
    return jsonify({"success": True})


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*52)
    print("  TB DETECTION — ENHANCED FLASK APP")
    print("="*52)
    print(f"  SVM Model : {'✓ loaded' if model_svm else '✗ MISSING — run train_model.py'}")
    print(f"  LR  Model : {'✓ loaded' if model_lr  else '✗ not found (re-run train_model.py)'}")
    print(f"  Scaler    : {'✓ loaded' if scaler    else '✗ MISSING — run train_model.py'}")
    print("  Open: http://127.0.0.1:5000")
    print("="*52 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
