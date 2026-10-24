from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, fbeta_score, precision_score, recall_score

HIGH_RISK = "HIGH_RISK"
LOW_RISK = "LOW_RISK"


def _normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    label = str(value).strip().upper().replace(" ", "_")
    if label in {"LOW_RISK", "LOW", "0"}:
        return LOW_RISK
    if label in {"HIGH_RISK", "HIGH", "1"}:
        return HIGH_RISK
    return label if label in {LOW_RISK, HIGH_RISK} else None


def extract_ground_truth(events: Sequence[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        actual = (
            event.get("actual_label")
            or event.get("label")
            or event.get("ground_truth")
            or event.get("target")
        )
        normalized = _normalize_label(actual)
        if normalized is not None:
            labels.append(normalized)
    return labels


def compute_metrics(events: Sequence[dict[str, Any]], predictions: Sequence[dict[str, Any]]) -> dict[str, float | list[int]] | None:
    true_labels = extract_ground_truth(events)
    if not true_labels or len(true_labels) != len(predictions):
        print("Ground-truth labels not available.\nSkipping evaluation metrics.")
        return None

    y_true = np.array([1 if label == HIGH_RISK else 0 for label in true_labels], dtype=int)
    y_pred = np.array(
        [1 if _normalize_label(prediction.get("prediction")) == HIGH_RISK else 0 for prediction in predictions],
        dtype=int,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }

    print("Evaluation metrics:")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1 Score : {metrics['f1']:.4f}")
    print(f"  F2 Score : {metrics['f2']:.4f}")
    return metrics
