from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from predict import run_inference
from train import train_pipeline


def _build_config(path: Path, config_name: str) -> any:
    cfg = OmegaConf.load(path / "configs" / f"{config_name}.yaml")
    return cfg


def create_synthetic_dataset(n_samples: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    ages = rng.integers(20, 80, size=n_samples)
    bmi = rng.normal(28, 4, size=n_samples)
    heart_rate = rng.normal(75, 8, size=n_samples)
    gender = rng.choice(["M", "F"], size=n_samples)
    noise = rng.normal(0, 1, size=n_samples)
    outcome_prob = 1 / (1 + np.exp(-(0.04 * (ages - 50) + 0.1 * (bmi - 25) + noise)))
    outcome = (outcome_prob > 0.5).astype(int)

    df = pd.DataFrame(
        {
            "age": ages,
            "gender": gender,
            "bmi": bmi,
            "heart_rate": heart_rate,
            "discharge_notes": rng.choice(["none", "short", "long"], size=n_samples),
            "bill_amount": rng.normal(1000, 200, size=n_samples),
            "outcome": outcome,
        }
    )
    return df


def test_training_and_inference_end_to_end(tmp_path: Path):
    project_root = Path(__file__).resolve().parents[1]

    train_cfg = _build_config(project_root, "train")
    infer_cfg = _build_config(project_root, "infer")

    train_df = create_synthetic_dataset(200)
    test_df = create_synthetic_dataset(50).drop(columns=["outcome"])

    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    output_dir = tmp_path / "artifacts"

    train_cfg.paths.train_csv = str(train_csv)
    train_cfg.paths.output_dir = str(output_dir)
    train_cfg.calibration.apply = False
    train_cfg.shap.sample_size = 50

    results = train_pipeline(train_cfg)

    model_path = Path(results["model_path"])
    meta_path = Path(results["meta_path"])
    metrics_path = Path(results["metrics_path"])

    assert model_path.exists()
    assert meta_path.exists()
    assert metrics_path.exists()

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta.get("target") == "outcome"

    infer_cfg.model.path = str(model_path)
    infer_cfg.model.meta = str(meta_path)
    infer_cfg.prediction.input_csv = str(test_csv)
    infer_cfg.prediction.output_csv = str(tmp_path / "predictions.csv")

    predictions = run_inference(infer_cfg)

    assert Path(infer_cfg.prediction.output_csv).exists()
    assert "prediction" in predictions.columns
    assert len(predictions) == len(test_df)

