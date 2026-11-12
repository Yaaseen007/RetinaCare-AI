"""
==============================================================================
Diabetic Retinopathy Detection - Model Evaluation & Performance Analytics
Department of Computer Science & Engineering - GM University

This module computes multi-class clinical evaluation metrics:
1. Overall Accuracy
2. Multi-class Precision, Recall, and F1-Score (Macro and Weighted)
3. Multi-class One-vs-Rest (OvR) ROC-AUC Curves and Scores
4. Raw and Normalized Confusion Matrices
5. Comprehensive Scikit-Learn Classification Report

Adherence to PBL Academic Integrity:
------------------------------------
- strictly avoids fabricating metrics or training results.
- Un-evaluated or un-trained architectures are explicitly designated as "Not trained".
==============================================================================
"""

import os
import sys
import json
from pathlib import Path

# Ensure project root and external libraries are in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
_PYLIB = Path("E:/pylib")
if _PYLIB.exists() and str(_PYLIB) not in sys.path:
    sys.path.insert(0, str(_PYLIB))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

try:
    import tensorflow as tf
    HAS_TF = True
except Exception:
    tf = None
    HAS_TF = False

try:
    from utils.config import (
        CLASS_NAMES,
        CLASS_SHORT_NAMES,
        NUM_CLASSES,
        RESULTS_DIR,
        CONFUSION_MATRIX_PATH,
        EVALUATION_REPORT_PATH,
        METRICS_JSON_PATH,
        MODEL_COMPARISON_JSON_PATH,
        CANDIDATE_MODELS,
        TEST_DIR,
        IMAGE_SIZE
    )
    from src.preprocessing import preprocess_retinal_image
except ImportError:
    CLASS_NAMES = ["No Diabetic Retinopathy", "Mild", "Moderate", "Severe", "Proliferative Diabetic Retinopathy"]
    CLASS_SHORT_NAMES = ["No DR (0)", "Mild (1)", "Moderate (2)", "Severe (3)", "Proliferative (4)"]
    NUM_CLASSES = 5
    RESULTS_DIR = Path("results")
    CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
    EVALUATION_REPORT_PATH = RESULTS_DIR / "evaluation_report.png"
    METRICS_JSON_PATH = RESULTS_DIR / "metrics.json"
    MODEL_COMPARISON_JSON_PATH = RESULTS_DIR / "model_comparison.json"
    CANDIDATE_MODELS = ["MobileNetV2", "EfficientNetB0", "ResNet50", "VGG16", "CustomCNN"]
    TEST_DIR = Path("dataset/test")
    IMAGE_SIZE = (224, 224)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path = CONFUSION_MATRIX_PATH
) -> None:
    """
    Generates and saves a publication-quality confusion matrix heatmap.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    # Compute normalized version
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
    sns.set_theme(style="white")

    # Annotated heatmap with counts and percentages
    annot_matrix = np.empty_like(cm, dtype=object)
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            annot_matrix[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

    sns.heatmap(
        cm,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        xticklabels=CLASS_SHORT_NAMES,
        yticklabels=CLASS_SHORT_NAMES,
        cbar=True,
        linewidths=1.0,
        linecolor="#e2e8f0",
        ax=ax
    )

    ax.set_title("Diabetic Retinopathy Severity - Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Severity Level", fontsize=12, fontweight="semibold", labelpad=10)
    ax.set_ylabel("True Clinical Severity Level", fontsize=12, fontweight="semibold", labelpad=10)
    plt.xticks(rotation=25, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_roc_curves(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    output_path: Optional[Path] = None
) -> Dict[str, float]:
    """
    Plots multi-class One-vs-Rest (OvR) ROC curves and returns AUC values.
    """
    if output_path is None:
        output_path = RESULTS_DIR / "roc_curves.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Binarize labels for OvR evaluation
    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    # Handle single class edge-case
    if y_true_bin.shape[1] == 1 and NUM_CLASSES > 1:
        temp = np.zeros((len(y_true), NUM_CLASSES))
        for idx, val in enumerate(y_true):
            if 0 <= val < NUM_CLASSES:
                temp[idx, val] = 1
        y_true_bin = temp

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)
    palette = ["#10b981", "#84cc16", "#f59e0b", "#f97316", "#ef4444"]
    auc_dict = {}

    for i in range(NUM_CLASSES):
        if np.sum(y_true_bin[:, i]) > 0:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
            class_auc = auc(fpr, tpr)
            auc_dict[CLASS_SHORT_NAMES[i]] = float(class_auc)
            ax.plot(
                fpr,
                tpr,
                color=palette[i % len(palette)],
                lw=2,
                label=f"{CLASS_SHORT_NAMES[i]} (AUC = {class_auc:.3f})"
            )
        else:
            auc_dict[CLASS_SHORT_NAMES[i]] = 0.0

    # Macro-average ROC AUC
    try:
        macro_auc = roc_auc_score(y_true_bin, y_pred_probs, multi_class="ovr", average="macro")
    except Exception:
        macro_auc = np.mean(list(auc_dict.values())) if auc_dict else 0.0

    auc_dict["Macro Average"] = float(macro_auc)

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="semibold")
    ax.set_title(f"Multi-class ROC Curves (Macro AUC: {macro_auc:.3f})", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    return auc_dict


def plot_metrics_summary(
    metrics: Dict[str, Any],
    output_path: Path = EVALUATION_REPORT_PATH
) -> None:
    """
    Renders an academic report card visual highlighting key performance metrics.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    ax.axis("off")

    # Card background
    rect = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color="#f8fafc", zorder=0)
    ax.add_patch(rect)

    # Title
    ax.text(0.5, 0.90, "RetinaCare AI - Model Evaluation Summary", fontsize=15, fontweight="bold",
            ha="center", color="#0f172a", transform=ax.transAxes)
    ax.text(0.5, 0.81, f"Evaluated Architecture: {metrics.get('model_name', 'MobileNetV2')}", fontsize=11,
            ha="center", color="#64748b", transform=ax.transAxes)

    # Horizontal divider
    ax.plot([0.1, 0.9], [0.75, 0.75], color="#cbd5e1", lw=1.2, transform=ax.transAxes)

    # Metric boxes
    metric_labels = [
        ("Accuracy", f"{metrics.get('accuracy', 0.0)*100:.2f}%"),
        ("Macro Precision", f"{metrics.get('macro_precision', 0.0)*100:.2f}%"),
        ("Macro Recall", f"{metrics.get('macro_recall', 0.0)*100:.2f}%"),
        ("Macro F1-Score", f"{metrics.get('macro_f1', 0.0)*100:.2f}%"),
        ("Macro ROC-AUC", f"{metrics.get('macro_roc_auc', 0.0):.4f}"),
        ("Test Samples", str(metrics.get('total_test_samples', 0)))
    ]

    positions = [
        (0.20, 0.55), (0.50, 0.55), (0.80, 0.55),
        (0.20, 0.28), (0.50, 0.28), (0.80, 0.28)
    ]

    for (name, val), (x_pos, y_pos) in zip(metric_labels, positions):
        ax.text(x_pos, y_pos + 0.06, name, fontsize=10, ha="center", color="#475569",
                fontweight="semibold", transform=ax.transAxes)
        ax.text(x_pos, y_pos, val, fontsize=14, ha="center", color="#0284c7",
                fontweight="bold", transform=ax.transAxes)

    # Footer notice
    ax.text(0.5, 0.07, "B.Tech CSE PBL Project | Department of Computer Science & Engineering, GM University",
            fontsize=9, ha="center", color="#94a3b8", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def evaluate_dataset(
    model: Any,
    test_data: Any,
    model_name: str = "MobileNetV2",
    output_dir: Path = RESULTS_DIR
) -> Dict[str, Any]:
    """
    Evaluates a trained TensorFlow model against a test dataset and persists
    all diagnostic plots and JSON metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true_list = []
    y_pred_probs_list = []

    print("[RetinaCare Evaluation] Generating test predictions...")
    for images, labels in test_data:
        preds = model.predict(images, verbose=0)
        y_pred_probs_list.append(preds)
        y_true_list.append(labels.numpy())

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred_probs = np.concatenate(y_pred_probs_list, axis=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calculate key metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Multi-class ROC-AUC
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
        macro_roc_auc = float(roc_auc_score(y_true_bin, y_pred_probs, multi_class="ovr", average="macro"))
    except Exception:
        macro_roc_auc = 0.0

    # Classification Report
    clf_report_dict = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_SHORT_NAMES,
        output_dict=True,
        zero_division=0
    )

    metrics_payload = {
        "model_name": model_name,
        "total_test_samples": int(len(y_true)),
        "accuracy": acc,
        "macro_precision": prec_macro,
        "macro_recall": rec_macro,
        "macro_f1": f1_macro,
        "weighted_precision": prec_weighted,
        "weighted_recall": rec_weighted,
        "weighted_f1": f1_weighted,
        "macro_roc_auc": macro_roc_auc,
        "classification_report": clf_report_dict
    }

    # Save metrics JSON
    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4)

    # Generate visual artifacts
    print("[RetinaCare Evaluation] Rendering Confusion Matrix...")
    plot_confusion_matrix(y_true, y_pred, output_path=CONFUSION_MATRIX_PATH)

    print("[RetinaCare Evaluation] Rendering ROC Curves...")
    plot_roc_curves(y_true, y_pred_probs, output_path=output_dir / "roc_curves.png")

    print("[RetinaCare Evaluation] Rendering Metrics Card...")
    plot_metrics_summary(metrics_payload, output_path=EVALUATION_REPORT_PATH)

    # Update model comparison table
    update_model_comparison_record(
        model_name=model_name,
        accuracy=acc,
        precision=prec_macro,
        recall=rec_macro,
        f1=f1_macro,
        roc_auc=macro_roc_auc,
        training_time_sec=None
    )

    print(f"[RetinaCare Evaluation] Completed. Metrics written to {METRICS_JSON_PATH}")
    return metrics_payload


def update_model_comparison_record(
    model_name: str,
    accuracy: float,
    precision: float,
    recall: float,
    f1: float,
    roc_auc: float,
    training_time_sec: Optional[float] = None
) -> None:
    """
    Saves or updates comparison entry for the specified model architecture.
    """
    data = load_model_comparison_table()
    time_str = f"{training_time_sec:.1f}s" if training_time_sec is not None else "N/A"

    data[model_name] = {
        "Accuracy": f"{accuracy * 100:.2f}%",
        "Precision": f"{precision * 100:.2f}%",
        "Recall": f"{recall * 100:.2f}%",
        "F1 Score": f"{f1 * 100:.2f}%",
        "ROC-AUC": f"{roc_auc:.4f}",
        "Training Time": time_str,
        "Status": "Trained"
    }

    MODEL_COMPARISON_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_COMPARISON_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_model_comparison_table() -> Dict[str, Dict[str, Any]]:
    """
    Loads candidate model records. If an architecture has not been trained,
    marks its values strictly as 'Not trained' (PBL Rule 11).
    """
    existing = {}
    if MODEL_COMPARISON_JSON_PATH.is_file():
        try:
            with open(MODEL_COMPARISON_JSON_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}

    full_table = {}
    for model_name in CANDIDATE_MODELS:
        if model_name in existing and existing[model_name].get("Status") == "Trained":
            full_table[model_name] = existing[model_name]
        else:
            full_table[model_name] = {
                "Accuracy": "Not evaluated",
                "Precision": "Not evaluated",
                "Recall": "Not evaluated",
                "F1 Score": "Not evaluated",
                "ROC-AUC": "Not evaluated",
                "Training Time": "Not evaluated",
                "Status": "Not evaluated"
            }
    return full_table


def get_comparison_dataframe() -> pd.DataFrame:
    """Returns the comparison table formatted as a pandas DataFrame."""
    table_dict = load_model_comparison_table()
    rows = []
    for model_name, metrics in table_dict.items():
        row = {"Model Architecture": model_name}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)
