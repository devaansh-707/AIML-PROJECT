"""Model factory utilities for training pipeline construction."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency at runtime
    XGBClassifier = None

from src.features.build_features import FeaturePreprocessor


def _compute_scale_pos_weight(y: np.ndarray) -> float:
    positives = np.sum(y == 1)
    negatives = np.sum(y != 1)
    if positives == 0:
        return 1.0
    return float(negatives / positives)


def _build_logistic(cfg: Dict[str, Any], y: np.ndarray | None, is_multilabel: bool = False) -> LogisticRegression:
    """Build Logistic Regression classifier."""
    lr_cfg = cfg.get("model", {}).get("logistic", {})
    kwargs = {
        "C": lr_cfg.get("C", 1.0),
        "penalty": lr_cfg.get("penalty", "l2"),
        "solver": lr_cfg.get("solver", "lbfgs"),
        "max_iter": 5000,
        "random_state": cfg.get("model", {}).get("random_state", 42),
    }
    
    # For multiclass, use multinomial
    if not is_multilabel and y is not None and len(np.unique(y)) > 2:
        kwargs["multi_class"] = lr_cfg.get("multi_class", "multinomial")
    
    # Class weight for imbalance
    imbalance_cfg = cfg.get("imbalance", {})
    if imbalance_cfg.get("enable", True) and not is_multilabel:
        kwargs["class_weight"] = "balanced"
    
    lr = LogisticRegression(**kwargs)
    
    if is_multilabel:
        return MultiOutputClassifier(lr)
    return lr


def _build_xgboost(cfg: Dict[str, Any], y: np.ndarray | None, is_multilabel: bool = False) -> Any:
    """Build XGBoost classifier."""
    if XGBClassifier is None:
        raise ImportError("xgboost is not installed; install it or switch model.name to logreg")

    params = cfg.get("model", {}).get("xgboost", {}).copy()
    params.setdefault("random_state", cfg.get("model", {}).get("random_state", 42))
    params.setdefault("use_label_encoder", False)
    params.setdefault("eval_metric", "logloss")
    
    # Determine objective
    if is_multilabel:
        params["objective"] = "binary:logistic"
    elif y is not None:
        n_classes = len(np.unique(y))
        if n_classes == 2:
            params["objective"] = "binary:logistic"
        else:
            params["objective"] = "multi:softprob"
            params["num_class"] = n_classes
    else:
        params["objective"] = "binary:logistic"
    
    xgb = XGBClassifier(**params)
    
    if is_multilabel:
        return MultiOutputClassifier(xgb)
    return xgb


def build_training_pipeline(
    config: Dict[str, Any],
    y: np.ndarray | None = None,
    is_multilabel: bool = False,
) -> Pipeline:
    """Build complete training pipeline with feature engineering and model."""
    model_name = config.get("model", {}).get("name", "xgb").lower()
    
    if model_name in ["logistic", "logreg", "lr"]:
        estimator = _build_logistic(config, y, is_multilabel)
    elif model_name in ["xgboost", "xgb"]:
        estimator = _build_xgboost(config, y, is_multilabel)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    feature_pipe = FeaturePreprocessor(config)
    return Pipeline([("features", feature_pipe), ("model", estimator)])

