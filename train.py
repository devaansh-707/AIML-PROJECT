"""Training entrypoint orchestrating the full ML workflow."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from hydra import main
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from schemas.tabular_schema import validate_training_dataframe
from src.eval.metrics import (
    compute_metrics,
    confusion_matrix_to_dict,
    save_confusion_matrix,
    save_metrics,
)
from src.models.build_model import build_training_pipeline
from src.models.calibrate import calibrate_pipeline
from utils import (
    best_threshold_for_f1,
    ensure_path,
    set_global_seed,
)


log = logging.getLogger(__name__)


def _get_scoring_function(metric_name: str):
    metric_name = metric_name.lower()
    if metric_name == "roc_auc":

        def scorer(y_true, proba):
            proba = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba
            return roc_auc_score(y_true, proba)

        return scorer
    if metric_name in {"macro_f1", "f1"}:

        def scorer(y_true, proba):
            pred = np.argmax(proba, axis=1) if proba.ndim == 2 else (proba >= 0.5).astype(int)
            metrics = compute_metrics(y_true, pred, proba)
            return metrics["f1"]

        return scorer
    if metric_name == "pr_auc":

        def scorer(y_true, proba):
            from sklearn.metrics import average_precision_score

            proba = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba
            return average_precision_score(y_true, proba)

        return scorer

    raise ValueError(f"Unsupported scoring metric for calibration: {metric_name}")


def train_pipeline(cfg: DictConfig) -> dict:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    set_global_seed(cfg.model.random_state)

    output_dir = Path(cfg.paths.output_dir)
    ensure_path(output_dir)

    df = pd.read_csv(cfg.paths.train_csv)
    schema = validate_training_dataframe(df, cfg_dict)

    target = cfg.target.name
    y = df[target].values
    X = df.drop(columns=[target])

    pipeline = build_training_pipeline(cfg_dict, y)

    skf = StratifiedKFold(
        n_splits=cfg.validation.folds,
        shuffle=True,
        random_state=cfg.model.random_state,
    )

    proba = cross_val_predict(pipeline, X, y, cv=skf, method="predict_proba")
    if proba.ndim == 1 or proba.shape[1] == 1:
        pred = (proba >= 0.5).astype(int)
    else:
        pred = np.argmax(proba, axis=1)

    metrics = compute_metrics(y, pred, proba)

    threshold_info = {}
    if len(np.unique(y)) == 2:
        best_thr, best_f1 = best_threshold_for_f1(y, proba[:, 1])
        threshold_info = {"threshold": float(best_thr), "f1": float(best_f1)}
        pred = (proba[:, 1] >= best_thr).astype(int)

    metrics.update(threshold_info)

    confusion = confusion_matrix_to_dict(y, pred)

    save_metrics(metrics, output_dir / "cv_report.json")
    save_confusion_matrix(confusion, output_dir / "confusion_matrix.json")

    pipeline.fit(X, y)
    final_model = pipeline
    calibration_scores = {}

    if cfg.calibration.apply and len(np.unique(y)) == 2:
        scoring_fn = _get_scoring_function(cfg.evaluation.metric)
        final_model, calibration_scores = calibrate_pipeline(
            pipeline, X, y, cfg.calibration.methods, cfg.calibration.cv, scoring_fn
        )

    model_path = output_dir / "model.pkl"
    meta_path = output_dir / "meta.json"
    base_model_path = output_dir / "pipeline_base.pkl"
    joblib.dump(pipeline, base_model_path)
    joblib.dump(final_model, model_path)

    feature_processor = pipeline.named_steps["features"]
    feature_names = list(feature_processor.get_feature_names_out())

    # Feature importance
    model = pipeline.named_steps["model"]
    importances = []
    if hasattr(model, "feature_importances_"):
        importances = list(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coefs = model.coef_
        if coefs.ndim == 2:
            coefs = np.mean(np.abs(coefs), axis=0)
        importances = list(np.abs(coefs))

    if importances:
        fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        fi_df = fi_df.sort_values("importance", ascending=False)
        fi_df.to_csv(output_dir / "feature_importance.csv", index=False)

    # SHAP explainability (skip if compatibility issues)
    try:
        sample_size = min(cfg.shap.sample_size, len(X))
        max_display = cfg.shap.max_display
        sample_df = X.sample(sample_size, random_state=cfg.model.random_state)
        transformed = feature_processor.transform(sample_df)
        
        # Use appropriate explainer based on model type
        if hasattr(model, 'get_booster'):  # XGBoost
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model, transformed, feature_names=feature_names)
        
        shap_values = explainer(transformed)

        shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
        shap_df.insert(0, "index", sample_df.index)
        shap_df.to_csv(output_dir / "shap_values.csv", index=False)

        shap_abs = np.abs(shap_values.values)
        shap_importance = shap_abs.mean(axis=0)
        shap_importance_df = pd.DataFrame(
            {"feature": feature_names, "mean_abs_shap": shap_importance}
        ).sort_values("mean_abs_shap", ascending=False)
        shap_importance_df.to_csv(output_dir / "feature_importance_shap.csv", index=False)

        plt.figure(figsize=(12, 6))
        shap.summary_plot(
            shap_values,
            transformed,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary.png", dpi=200)
        plt.close()
        log.info("SHAP explainability analysis completed successfully")
    except Exception as e:
        log.warning(f"SHAP explainability skipped due to compatibility issue: {e}")

    meta = {
        "target": target,
        "model": cfg.model.name,
        "features": feature_names,
        "threshold": threshold_info.get("threshold"),
        "calibration": calibration_scores,
        "schema_columns": list(schema.columns.keys()),
        "base_model_path": str(base_model_path),
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log.info("Training complete. Artifacts saved to %s", output_dir)
    return {
        "model_path": str(model_path),
        "meta_path": str(meta_path),
        "metrics_path": str(output_dir / "cv_report.json"),
    }


@main(config_path="configs", config_name="train", version_base=None)
def run(cfg: DictConfig) -> None:
    train_pipeline(cfg)


if __name__ == "__main__":
    run()
