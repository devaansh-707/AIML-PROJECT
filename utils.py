"""Utility helpers for the disease prediction stack."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - optional dependency
        pass


def best_threshold_for_f1(y_true, y_prob):
    quantiles = np.linspace(0.01, 0.99, 199)
    thr_list = np.unique(np.quantile(y_prob, quantiles))
    best_f1, best_thr = -1.0, 0.5
    for t in thr_list:
        y_pred = (y_prob >= t).astype(int)
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    return float(best_thr), float(best_f1)


def ensure_path(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], path: Path) -> None:
    ensure_path(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
