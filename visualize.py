from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from evaluate import HIGH_RISK, LOW_RISK, extract_ground_truth


def save_prediction_chart(predictions: Sequence[dict[str, Any]], threshold: float, output_path: str | Path = "prediction_results.png") -> None:
    event_ids = [str(item.get("event_id", f"Event {idx + 1}")) for idx, item in enumerate(predictions)]
    probabilities = [float(item.get("probability", 0.0)) for item in predictions]

    fig, ax = plt.subplots(figsize=(10, max(5.0, 0.7 * max(len(predictions), 1))))
    ax.barh(event_ids, probabilities, color="#4C78A8")
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold {threshold:.4f}")
    ax.set_title("Satellite Collision Risk Predictions")
    ax.set_xlabel("Predicted high-risk probability")
    ax.set_ylabel("Event identifiers")
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def save_confusion_matrix(events: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]], output_path: str | Path = "confusion_matrix_comparison.png") -> bool:
    true_labels = extract_ground_truth(events)
    pred_labels = [item.get("prediction") for item in predictions]
    if not true_labels or len(true_labels) != len(predictions):
        print("Ground-truth labels not available.\nSkipping evaluation metrics.")
        return False

    label_order = [LOW_RISK, HIGH_RISK]
    y_true = np.array([1 if label == HIGH_RISK else 0 for label in true_labels], dtype=int)
    y_pred = np.array([1 if str(label).upper().replace(" ", "_") == HIGH_RISK else 0 for label in pred_labels], dtype=int)
    matrix = np.array([[0, 0], [0, 0]], dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        matrix[actual, predicted] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["LOW RISK", "HIGH RISK"], yticklabels=["LOW RISK", "HIGH RISK"], ax=ax)
    ax.set_title("XGBoost + SMOTE Collision Risk Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return True


def save_feature_importance(model: Any, feature_cols: Sequence[str], output_path: str | Path = "feature_importance.png", top_n: int = 12) -> bool:
    if not hasattr(model, "feature_importances_"):
        print("Feature importance unavailable for this model. Keeping existing feature_importance.png.")
        return False

    importances = np.asarray(model.feature_importances_, dtype=float)
    names = list(feature_cols) if feature_cols else [f"feature_{idx}" for idx in range(len(importances))]
    if len(names) != len(importances):
        names = [f"feature_{idx}" for idx in range(len(importances))]

    order = np.argsort(importances)[::-1][:top_n]
    sorted_names = [names[idx] for idx in order]
    sorted_importances = importances[order]

    fig, ax = plt.subplots(figsize=(10, max(6.0, 0.45 * len(sorted_names))))
    ax.barh(sorted_names[::-1], sorted_importances[::-1], color="#3E9B6F")
    ax.set_title("Top 12 CDM Features Driving Collision Risk Prediction")
    ax.set_xlabel("Feature Importance Score")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return True
