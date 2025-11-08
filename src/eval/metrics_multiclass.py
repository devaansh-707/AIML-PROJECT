"""Metrics for multiclass and multilabel disease classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute comprehensive metrics for multiclass classification.
    
    Args:
        y_true: True class indices
        y_pred: Predicted class indices
        y_proba: Class probabilities (n_samples, n_classes)
        class_names: Disease names for each class
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    
    # ROC-AUC (one-vs-rest)
    if y_proba is not None and y_proba.shape[1] > 2:
        try:
            metrics["roc_auc_ovr"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            )
        except ValueError:
            metrics["roc_auc_ovr"] = None
    
    # Per-class metrics
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    if class_names is not None:
        metrics["per_class"] = {
            class_names[i]: {
                "f1": float(per_class_f1[i]),
                "precision": float(per_class_precision[i]),
                "recall": float(per_class_recall[i]),
            }
            for i in range(len(class_names))
        }
    
    return metrics


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute metrics for multilabel classification.
    
    Args:
        y_true: True binary labels (n_samples, n_classes)
        y_pred: Predicted binary labels (n_samples, n_classes)
        y_proba: Class probabilities (n_samples, n_classes)
        class_names: Disease names
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    
    # ROC-AUC per class
    if y_proba is not None:
        try:
            per_class_auc = roc_auc_score(y_true, y_proba, average=None)
            metrics["roc_auc_macro"] = float(np.mean(per_class_auc))
            
            if class_names is not None:
                metrics["per_class_auc"] = {
                    class_names[i]: float(per_class_auc[i])
                    for i in range(len(class_names))
                }
        except ValueError:
            metrics["roc_auc_macro"] = None
    
    return metrics


def save_confusion_matrix_multiclass(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: Path,
):
    """Save confusion matrix as CSV with disease names."""
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(output_path)


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: Path,
):
    """Save detailed classification report as JSON."""
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def save_metrics(metrics: Dict, path: Path):
    """Save metrics dictionary to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
