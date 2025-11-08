"""Optuna hyperparameter tuning entrypoint."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import numpy as np
import optuna
import pandas as pd
from hydra import main
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.eval.metrics import compute_metrics
from src.models.build_model import build_training_pipeline
from schemas.tabular_schema import validate_training_dataframe


METRIC_KEY_MAP = {
    "roc_auc": "roc_auc",
    "pr_auc": "pr_auc",
    "macro_f1": "f1",
}


def _load_data(cfg: DictConfig) -> pd.DataFrame:
    path = Path(cfg.paths.train_csv)
    if not path.exists():
        raise FileNotFoundError(f"Training data not found at {path}")
    return pd.read_csv(path)


def _suggest_params(trial: optuna.Trial, cfg: DictConfig) -> Dict[str, Any]:
    model_name = cfg.model.name.lower()
    params: Dict[str, Any] = {}

    if model_name in {"xgboost", "xgb"}:
        params.update(
            {
                "model.xgboost.n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "model.xgboost.learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "model.xgboost.max_depth": trial.suggest_int("max_depth", 3, 10),
                "model.xgboost.subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "model.xgboost.colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.5, 1.0
                ),
                "model.xgboost.gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "model.xgboost.min_child_weight": trial.suggest_float(
                    "min_child_weight", 1.0, 10.0
                ),
                "model.xgboost.reg_lambda": trial.suggest_float(
                    "reg_lambda", 0.1, 10.0, log=True
                ),
                "model.xgboost.reg_alpha": trial.suggest_float(
                    "reg_alpha", 1e-8, 1.0, log=True
                ),
            }
        )
    elif model_name in {"logistic", "logreg", "lr"}:
        params.update(
            {
                "model.logistic.C": trial.suggest_float("C", 1e-3, 10.0, log=True)
            }
        )
    else:
        raise ValueError(f"Optuna search is not configured for model: {model_name}")

    return params


def _evaluate(cfg: DictConfig, params: Dict[str, Any], df: pd.DataFrame) -> float:
    cfg_copy = OmegaConf.to_container(cfg, resolve=True)
    for key, value in params.items():
        section, field = key.split(".", 1)
        container = cfg_copy
        parts = key.split(".")
        for part in parts[:-1]:
            container = container[part]
        container[parts[-1]] = value

    # Support target from cfg.target.name
    target = cfg.target.name
    X = df.drop(columns=[target])
    y = df[target].values

    pipeline = build_training_pipeline(cfg_copy, y)
    skf = StratifiedKFold(
        n_splits=cfg.validation.folds, shuffle=True, random_state=cfg.model.random_state
    )
    proba = cross_val_predict(pipeline, X, y, cv=skf, method="predict_proba")
    if proba.ndim == 1 or proba.shape[1] == 1:
        pred = (proba >= 0.5).astype(int)
    else:
        pred = np.argmax(proba, axis=1)
    metrics = compute_metrics(y, pred, proba)
    metric_key = METRIC_KEY_MAP.get(cfg.evaluation.metric, cfg.evaluation.metric)
    return metrics.get(metric_key, -np.inf)


@main(config_path="../../configs", config_name="train", version_base=None)
def run(cfg: DictConfig) -> None:
    # Use `tuning.enable` per train.yaml
    if not cfg.tuning.enable:
        raise RuntimeError("Enable tuning.enable in config to run the search")

    df = _load_data(cfg)
    validate_training_dataframe(df, cfg)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, cfg)
        return _evaluate(cfg, params, df)

    study = optuna.create_study(direction="maximize")
    # Map timeout from minutes to seconds if available
    timeout_minutes = cfg.tuning.get("timeout_minutes", None)
    timeout_seconds = None if timeout_minutes is None else int(timeout_minutes) * 60
    study.optimize(objective, n_trials=cfg.tuning.n_trials, timeout=timeout_seconds)

    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "optuna_study.json").open("w", encoding="utf-8") as f:
        json.dump(study.best_trial.params, f, indent=2)

    df_trials = study.trials_dataframe()
    df_trials.to_csv(output_dir / "optuna_trials.csv", index=False)


if __name__ == "__main__":
    run()

