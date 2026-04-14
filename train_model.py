"""
=============================================================
  TB DETECTION — MODEL TRAINING (ENHANCED)
  Trains: SVM + Logistic Regression
  Saves : model_svm.pkl, model_lr.pkl, scaler.pkl
=============================================================
"""

import os
import pickle
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_DIR  = "dataset"
IMG_SIZE     = (64, 64)
TEST_SIZE    = 0.2
RANDOM_STATE = 42
LABEL_MAP    = {"TB": 1, "Normal": 0}


# ─────────────────────────────────────────────
# STEP 1 — LOAD IMAGES
# ─────────────────────────────────────────────
def load_dataset(dataset_dir):
    images, labels, filenames = [], [], []
    for class_name, class_id in LABEL_MAP.items():
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"  ⚠  Not found: {class_dir}")
            continue
        files = [f for f in os.listdir(class_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"  📂  {class_name}: {len(files)} images")
        for fname in files:
            img = cv2.imread(os.path.join(class_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            images.append(img.flatten())
            labels.append(class_id)
            filenames.append(fname)

    if not images:
        raise FileNotFoundError(
            "No images found. Structure:\n  dataset/TB/\n  dataset/Normal/")

    X = np.array(images, dtype=np.float32)
    y = np.array(labels,  dtype=np.int32)
    df = pd.DataFrame({"filename": filenames, "label": y})
    df["class"] = df["label"].map({0: "Normal", 1: "TB"})
    print(f"\n  ✅  Total: {len(X)}  |  Normal: {np.sum(y==0)}  |  TB: {np.sum(y==1)}\n")
    return X, y, df


# ─────────────────────────────────────────────
# STEP 2 — NORMALIZE
# ─────────────────────────────────────────────
def normalize(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), scaler


# ─────────────────────────────────────────────
# STEP 3 — TRAIN BOTH MODELS
# ─────────────────────────────────────────────
def train_svm(X_train, y_train):
    print("  🏋  Training SVM (RBF kernel) …")
    model = SVC(kernel="rbf", C=1.0, gamma="scale",
                probability=True, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    print("  ✅  SVM done!\n")
    return model


def train_logistic(X_train, y_train):
    print("  🏋  Training Logistic Regression …")
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, solver="lbfgs")
    model.fit(X_train, y_train)
    print("  ✅  Logistic Regression done!\n")
    return model


# ─────────────────────────────────────────────
# STEP 4 — EVALUATE
# ─────────────────────────────────────────────
def evaluate(model, X_test, y_test, name="Model"):
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"  ── {name} ──")
    print(f"  🎯  Accuracy : {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=["Normal", "TB"]))
    return y_pred, acc


# ─────────────────────────────────────────────
# STEP 5 — PLOT (side-by-side confusion matrices)
# ─────────────────────────────────────────────
def plot_results(svm_model, lr_model, X_test, y_test,
                 svm_pred, lr_pred, df, svm_acc, lr_acc):
    fig = plt.figure(figsize=(16, 10), facecolor="#0d1117")
    fig.suptitle("TB Detection — Dual Model Evaluation", fontsize=17,
                 color="white", fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
    bg, tc, gc = "#0d1117", "white", "#30363d"

    def style_ax(ax, title):
        ax.set_facecolor(bg)
        ax.set_title(title, color=tc, pad=10)
        ax.tick_params(colors=tc)
        for sp in ax.spines.values(): sp.set_edgecolor(gc)

    # Distribution
    ax0 = fig.add_subplot(gs[0, 0])
    counts = df["class"].value_counts()
    bars = ax0.bar(counts.index, counts.values,
                   color=["#22c55e", "#ef4444"], width=0.5, zorder=3)
    style_ax(ax0, "Dataset Distribution")
    ax0.set_ylabel("Count", color=tc)
    ax0.yaxis.grid(True, color=gc, zorder=0)
    for b in bars:
        ax0.text(b.get_x()+b.get_width()/2, b.get_height()+1,
                 str(int(b.get_height())), ha="center", color=tc, fontsize=10)

    # SVM Confusion
    ax1 = fig.add_subplot(gs[0, 1])
    ConfusionMatrixDisplay(confusion_matrix(y_test, svm_pred),
                          display_labels=["Normal","TB"]).plot(
        ax=ax1, colorbar=False, cmap="Blues")
    style_ax(ax1, f"SVM — {svm_acc*100:.1f}%")
    ax1.xaxis.label.set_color(tc); ax1.yaxis.label.set_color(tc)

    # LR Confusion
    ax2 = fig.add_subplot(gs[0, 2])
    ConfusionMatrixDisplay(confusion_matrix(y_test, lr_pred),
                          display_labels=["Normal","TB"]).plot(
        ax=ax2, colorbar=False, cmap="Greens")
    style_ax(ax2, f"Logistic Reg — {lr_acc*100:.1f}%")
    ax2.xaxis.label.set_color(tc); ax2.yaxis.label.set_color(tc)

    # Accuracy comparison bar
    ax3 = fig.add_subplot(gs[1, 0])
    accs = [svm_acc*100, lr_acc*100]
    names = ["SVM", "Log. Reg."]
    ax3.bar(names, accs, color=["#38bdf8", "#a78bfa"], width=0.4, zorder=3)
    ax3.set_ylim([min(accs)-5, 100])
    style_ax(ax3, "Model Accuracy Comparison")
    ax3.set_ylabel("Accuracy (%)", color=tc)
    ax3.yaxis.grid(True, color=gc, zorder=0)
    for i, v in enumerate(accs):
        ax3.text(i, v+0.3, f"{v:.1f}%", ha="center", color=tc, fontsize=11)

    # Sample images
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off"); style_ax(ax4, "Sample: Normal | TB")
    for pos, (cls, label, color) in enumerate([(0, "Normal", "#22c55e"), (1, "TB", "#ef4444")]):
        idx_arr = np.where(y_test == cls)[0]
        if len(idx_arr):
            img = X_test[idx_arr[0]].reshape(IMG_SIZE)
            ax_in = ax4.inset_axes([0.05 + pos*0.5, 0.1, 0.43, 0.75])
            ax_in.imshow(img, cmap="gray")
            ax_in.set_title(label, color=color, fontsize=9)
            ax_in.axis("off")

    # Summary
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor("#161b22"); ax5.axis("off")
    ax5.set_title("Summary", color=tc, pad=10)
    for sp in ax5.spines.values(): sp.set_edgecolor(gc)
    lines = [
        f"Image Size  : {IMG_SIZE[0]}×{IMG_SIZE[1]} px",
        f"Features    : {IMG_SIZE[0]*IMG_SIZE[1]}",
        f"Train Split : {int((1-TEST_SIZE)*100)}%",
        f"Test  Split : {int(TEST_SIZE*100)}%",
        f"SVM  Acc    : {svm_acc*100:.2f}%",
        f"LR   Acc    : {lr_acc*100:.2f}%",
        f"Winner      : {'SVM' if svm_acc >= lr_acc else 'Log.Reg.'}",
    ]
    for i, l in enumerate(lines):
        ax5.text(0.06, 0.88-i*0.12, l, transform=ax5.transAxes,
                 color=tc, fontsize=10, fontfamily="monospace")

    plt.savefig("evaluation_report.png", dpi=150,
                bbox_inches="tight", facecolor="#0d1117")
    print("  📈  evaluation_report.png saved")
    plt.show()


# ─────────────────────────────────────────────
# STEP 6 — SAVE
# ─────────────────────────────────────────────
def save_all(svm_model, lr_model, scaler):
    artifacts = [
        ("model_svm.pkl", svm_model),
        ("model_lr.pkl",  lr_model),
        ("model.pkl",     svm_model),   # backward compat alias
        ("scaler.pkl",    scaler),
    ]
    for fname, obj in artifacts:
        with open(fname, "wb") as f:
            pickle.dump(obj, f)
        print(f"  💾  Saved → {fname}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  TB DETECTION — DUAL MODEL TRAINING")
    print("="*50 + "\n")

    print("📂 STEP 1: Loading dataset …")
    X, y, df = load_dataset(DATASET_DIR)

    print("✂️  STEP 2: Train/test split …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    print(f"   Train: {len(X_train)}  |  Test: {len(X_test)}\n")

    print("📐 STEP 3: Normalizing …")
    X_tr_s, X_te_s, scaler = normalize(X_train, X_test)

    print("🤖 STEP 4: Training models …")
    svm_model = train_svm(X_tr_s, y_tr_s := y_train)
    lr_model  = train_logistic(X_tr_s, y_train)

    print("📊 STEP 5: Evaluating …\n")
    svm_pred, svm_acc = evaluate(svm_model, X_te_s, y_test, "SVM")
    lr_pred,  lr_acc  = evaluate(lr_model,  X_te_s, y_test, "Logistic Regression")

    print("🎨 STEP 6: Plotting …")
    plot_results(svm_model, lr_model, X_te_s, y_test,
                 svm_pred, lr_pred, df, svm_acc, lr_acc)

    print("\n💾 STEP 7: Saving models …")
    save_all(svm_model, lr_model, scaler)

    print("\n✅  Done! Run  python app.py  to launch the web app.\n")
