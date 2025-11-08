"""Evaluation utilities for classification models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    report: Dict[str, float] = {}

    report["accuracy"] = float(accuracy_score(y_true, y_pred))
    report["precision"] = float(precision_score(y_true, y_pred, average="macro"))
    report["recall"] = float(recall_score(y_true, y_pred, average="macro"))
    report["f1"] = float(f1_score(y_true, y_pred, average="macro"))

    if y_proba is not None:
        try:
            if y_proba.ndim == 1 or y_proba.shape[1] == 1:
                y_scores = y_proba.ravel()
                report["roc_auc"] = float(roc_auc_score(y_true, y_scores))
                report["pr_auc"] = float(average_precision_score(y_true, y_scores))
            else:
                report["roc_auc"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovo", average="macro")
                )
                report["pr_auc"] = float(
                    average_precision_score(y_true, y_proba, average="macro")
                )
        except ValueError:
            pass

    return report


def confusion_matrix_to_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, list]:
    matrix = confusion_matrix(y_true, y_pred)
    return {
        "matrix": matrix.tolist(),
        "labels": sorted({int(label) for label in np.unique(y_true)}),
    }


def save_metrics(report: Dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def save_confusion_matrix(matrix: Dict[str, list], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(matrix, f, indent=2)

