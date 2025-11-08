"""Inference entrypoint using Hydra-driven config."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from hydra import main
from omegaconf import DictConfig

from utils import ensure_path


def run_inference(cfg: DictConfig):
    # Prefer new config layout under `paths.*`
    model_path = Path(cfg.paths.model)
    meta_path = Path(cfg.paths.meta)
    input_path = Path(cfg.paths.input)
    output_path = Path(cfg.paths.output)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found at {meta_path}")

    model = joblib.load(model_path)
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    df = pd.read_csv(input_path)
    target = meta.get("target")
    if target and target in df.columns:
        df = df.drop(columns=[target])

    required_cols = set(meta.get("schema_columns", []))
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for inference: {sorted(missing)}")

    # Optional user-provided threshold may live under `prediction.threshold`
    try:
        threshold = cfg.prediction.threshold
    except Exception:
        threshold = None
    if threshold is None:
        threshold = meta.get("threshold", 0.5)

    proba = model.predict_proba(df)

    if proba.ndim == 1 or proba.shape[1] == 1:
        proba = proba.reshape(-1, 1)

    if proba.shape[1] == 2:
        positive_scores = proba[:, 1]
        preds = (positive_scores >= threshold).astype(int)
        result = pd.DataFrame(
            {
                "prediction": preds,
                "probability": positive_scores,
            }
        )
    else:
        preds = np.argmax(proba, axis=1)
        columns = [f"class_{idx}_proba" for idx in range(proba.shape[1])]
        result = pd.DataFrame({"prediction": preds})
        result = pd.concat([result, pd.DataFrame(proba, columns=columns)], axis=1)

    ensure_path(output_path.parent)
    result.to_csv(output_path, index=False)
    return result


@main(config_path="configs", config_name="infer", version_base=None)
def run(cfg: DictConfig) -> None:
    run_inference(cfg)


if __name__ == "__main__":
    run()
