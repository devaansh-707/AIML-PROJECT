"""Label encoder for disease names to class indices."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd


class DiseaseLabelEncoder:
    """
    Encode disease names to integer class IDs and vice versa.
    Supports multiclass (one disease) and multilabel (multiple diseases).
    """

    def __init__(self):
        self.classes_: List[str] = []
        self.class_to_idx_: Dict[str, int] = {}
        self.idx_to_class_: Dict[int, str] = {}
        self.is_fitted_ = False

    def fit(self, y: Union[pd.Series, np.ndarray, List[str]]) -> "DiseaseLabelEncoder":
        """
        Learn unique disease classes from training labels.
        
        Args:
            y: Disease names (for multiclass) or list of names
        """
        if isinstance(y, pd.Series):
            y = y.values
        
        unique_classes = sorted(set(y))
        self.classes_ = unique_classes
        self.class_to_idx_ = {cls: idx for idx, cls in enumerate(unique_classes)}
        self.idx_to_class_ = {idx: cls for cls, idx in self.class_to_idx_.items()}
        self.is_fitted_ = True
        return self

    def transform(self, y: Union[pd.Series, np.ndarray, List[str]]) -> np.ndarray:
        """Transform disease names to class indices."""
        if not self.is_fitted_:
            raise ValueError("LabelEncoder must be fitted before transform")
        
        if isinstance(y, pd.Series):
            y = y.values
        
        return np.array([self.class_to_idx_[cls] for cls in y])

    def inverse_transform(self, y_idx: np.ndarray) -> np.ndarray:
        """Transform class indices back to disease names."""
        if not self.is_fitted_:
            raise ValueError("LabelEncoder must be fitted before inverse_transform")
        
        return np.array([self.idx_to_class_[int(idx)] for idx in y_idx])

    def fit_transform(self, y: Union[pd.Series, np.ndarray, List[str]]) -> np.ndarray:
        """Fit encoder and transform in one step."""
        return self.fit(y).transform(y)

    def save(self, path: Path):
        """Save label encoder to JSON."""
        data = {
            "classes": self.classes_,
            "mapping": self.class_to_idx_,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "DiseaseLabelEncoder":
        """Load label encoder from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        encoder = cls()
        encoder.classes_ = data["classes"]
        encoder.class_to_idx_ = {str(k): int(v) for k, v in data["mapping"].items()}
        encoder.idx_to_class_ = {int(v): str(k) for k, v in encoder.class_to_idx_.items()}
        encoder.is_fitted_ = True
        return encoder

    @property
    def n_classes(self) -> int:
        """Return number of unique classes."""
        return len(self.classes_)
