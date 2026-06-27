from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_prob_fake: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    # Convert inputs to NumPy arrays so the metric functions behave consistently.
    y_true = np.asarray(y_true).astype(int)
    y_prob_fake = np.asarray(y_prob_fake).astype(float)
    y_pred = (y_prob_fake >= threshold).astype(int)

    # Report both standard accuracy and imbalance-aware metrics.
    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # Handle cases where AUC or PR-AUC cannot be computed because only one class is present.
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob_fake))
    except ValueError:
        metrics["auc"] = float("nan")

    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob_fake))
    except ValueError:
        metrics["pr_auc"] = float("nan")

    return metrics


# Search over a fixed threshold range and keep the best score for the selected metric.
def find_best_threshold(
    y_true: np.ndarray,
    y_prob_fake: np.ndarray,
    metric: str = "balanced_accuracy",
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    num_steps: int = 181,
) -> Tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob_fake = np.asarray(y_prob_fake).astype(float)

    # Use a dense threshold grid to keep selection simple and reproducible.
    thresholds = np.linspace(min_threshold, max_threshold, num_steps)
    best_threshold = 0.5
    best_score = -float("inf")

    for threshold in thresholds:
        metrics_dict = classification_metrics(
            y_true,
            y_prob_fake,
            threshold=float(threshold),
        )
        score = metrics_dict.get(metric, float("nan"))
        if np.isnan(score):
            continue
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)

    if not np.isfinite(best_score):
        return 0.5, float("nan")

    return float(best_threshold), float(best_score)
