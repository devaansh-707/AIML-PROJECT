"""Sample weighting for imbalanced classification."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from collections import Counter


def compute_class_weights(
    y: np.ndarray,
    scheme: str = "inverse_freq"
) -> np.ndarray:
    """
    Compute sample weights based on class imbalance.
    
    Args:
        y: Class labels (integer indices)
        scheme: Weighting scheme - 'inverse_freq', 'balanced', or 'none'
    
    Returns:
        Sample weights (one per sample)
    """
    if scheme == "none":
        return np.ones(len(y))
    
    class_counts = Counter(y)
    n_samples = len(y)
    n_classes = len(class_counts)
    
    if scheme == "inverse_freq":
        # Weight inversely proportional to class frequency
        class_weight_dict = {
            cls: n_samples / (n_classes * count)
            for cls, count in class_counts.items()
        }
    elif scheme == "balanced":
        # sklearn's balanced weighting
        class_weight_dict = {
            cls: n_samples / (n_classes * count)
            for cls, count in class_counts.items()
        }
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme}")
    
    # Map to sample weights
    sample_weights = np.array([class_weight_dict[label] for label in y])
    
    # Normalize to sum to n_samples
    sample_weights = sample_weights * (n_samples / sample_weights.sum())
    
    return sample_weights


def compute_multilabel_weights(
    y_multilabel: np.ndarray,
    scheme: str = "inverse_freq"
) -> np.ndarray:
    """
    Compute sample weights for multilabel classification.
    
    Args:
        y_multilabel: Binary matrix (n_samples, n_classes)
        scheme: Weighting scheme
    
    Returns:
        Sample weights
    """
    if scheme == "none":
        return np.ones(y_multilabel.shape[0])
    
    # Count positive samples per class
    pos_counts = y_multilabel.sum(axis=0)
    n_samples = y_multilabel.shape[0]
    
    # Compute per-class weights
    class_weights = n_samples / (2 * pos_counts + 1)  # +1 to avoid division by zero
    
    # For each sample, average the weights of its positive classes
    sample_weights = []
    for row in y_multilabel:
        active_classes = np.where(row == 1)[0]
        if len(active_classes) > 0:
            weight = class_weights[active_classes].mean()
        else:
            weight = 1.0
        sample_weights.append(weight)
    
    sample_weights = np.array(sample_weights)
    sample_weights = sample_weights * (n_samples / sample_weights.sum())
    
    return sample_weights
