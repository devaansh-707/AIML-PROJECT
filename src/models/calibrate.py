"""Probability calibration utilities."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV


ScoreFunction = Callable[[np.ndarray, np.ndarray], float]


def calibrate_pipeline(
    pipeline,
    X,
    y,
    methods: Iterable[str],
    cv: int,
    scoring: ScoreFunction,
) -> Tuple[CalibratedClassifierCV, Dict[str, float]]:
    methods = list(dict.fromkeys(methods))  # preserve order, remove duplicates
    if not methods:
        raise ValueError("At least one calibration method must be provided")

    scored_models: Dict[str, Tuple[CalibratedClassifierCV, float]] = {}

    for method in methods:
        calibrator = CalibratedClassifierCV(
            estimator=clone(pipeline), method=method, cv=cv
        )
        calibrator.fit(X, y)
        proba = calibrator.predict_proba(X)
        score = float(scoring(y, proba))
        scored_models[method] = (calibrator, score)

    best_method = max(scored_models.items(), key=lambda item: item[1][1])[0]
    best_model = scored_models[best_method][0]
    calibration_scores = {method: score for method, (_, score) in scored_models.items()}
    calibration_scores["selected"] = best_method
    return best_model, calibration_scores

