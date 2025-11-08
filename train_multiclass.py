"""Training entrypoint for multi-disease classification."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from hydra import main
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.eval.metrics_multiclass import (
    compute_multiclass_metrics,
    compute_multilabel_metrics,
    save_classification_report,
    save_confusion_matrix_multiclass,
    save_metrics,
)
from src.explain.shap_tools import (
    compute_shap_values,
    get_top_features_per_sample,
    save_shap_importance_csv,
    save_shap_summary,
)
from src.models.build_model import build_training_pipeline
from src.models.label_encoder import DiseaseLabelEncoder
from src.models.weights import compute_class_weights, compute_multilabel_weights
from utils import ensure_path, set_global_seed

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def detect_task_type(cfg: DictConfig, df: pd.DataFrame) -> tuple[str, bool]:
    """
    Detect whether this is multiclass, multilabel, or binary classification.
    
    Returns:
        (task_type, is_multilabel)
    """
    target_cfg = cfg.get("target", {})
    
    if target_cfg.get("multilabel", False):
        return "multilabel", True
    
    target_col = target_cfg.get("name", "prognosis")
    if target_col in df.columns:
        unique_vals = df[target_col].nunique()
        if unique_vals == 2:
            return "binary", False
        else:
            return "multiclass", False
    
    # Fallback
    return "multiclass", False


def train_multiclass(cfg: DictConfig):
    """Train multiclass disease classifier."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    set_global_seed(cfg.model.random_state)
    
    output_dir = Path(cfg.paths.output_dir)
    ensure_path(output_dir)
    
    # Load data
    df = pd.read_csv(cfg.paths.train_csv)
    target_col = cfg.target.name
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data. Available: {df.columns.tolist()}")
    
    # Separate features and target
    y_raw = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Encode disease names to integers
    label_encoder = DiseaseLabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    
    log.info(f"Training multiclass model for {label_encoder.n_classes} diseases: {label_encoder.classes_}")
    
    # Save label encoder
    label_encoder.save(Path(cfg.paths.le_out))
    
    # Build pipeline
    pipeline = build_training_pipeline(cfg_dict, y, is_multilabel=False)
    
    # Compute sample weights for imbalance
    sample_weights = None
    if cfg.imbalance.enable:
        sample_weights = compute_class_weights(y, cfg.imbalance.scheme)
        log.info(f"Computed sample weights with scheme: {cfg.imbalance.scheme}")
    
    # Cross-validation
    cv_strategy = StratifiedKFold(
        n_splits=cfg.cv.folds,
        shuffle=True,
        random_state=cfg.model.random_state,
    )
    
    log.info(f"Running {cfg.cv.folds}-fold cross-validation...")
    
    # Get CV predictions (note: sample_weights not supported in cross_val_predict)
    # We'll train final model with weights but CV without for simplicity
    y_proba = cross_val_predict(
        pipeline, X, y,
        cv=cv_strategy,
        method="predict_proba"
    )
    y_pred = np.argmax(y_proba, axis=1)
    
    # Compute metrics
    metrics = compute_multiclass_metrics(y, y_pred, y_proba, label_encoder.classes_)
    log.info(f"CV Metrics - Accuracy: {metrics['accuracy']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}")
    
    # Save metrics
    save_metrics(metrics, Path(cfg.paths.report_out))
    
    # Save confusion matrix
    cm_path = output_dir / "confusion_matrix.csv"
    save_confusion_matrix_multiclass(y, y_pred, label_encoder.classes_, cm_path)
    
    # Save classification report
    report_path = output_dir / "classification_report.json"
    save_classification_report(y, y_pred, label_encoder.classes_, report_path)
    
    # Train final model on full data
    log.info("Training final model on full dataset...")
    if sample_weights is not None:
        pipeline.fit(X, y, model__sample_weight=sample_weights)
    else:
        pipeline.fit(X, y)
    
    # Save model
    model_path = Path(cfg.paths.model_out)
    joblib.dump(pipeline, model_path)
    log.info(f"Model saved to {model_path}")
    
    # Feature importance
    model = pipeline.named_steps["model"]
    feature_processor = pipeline.named_steps["features"]
    feature_names = list(feature_processor.get_feature_names_out())
    
    importances = []
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coefs = model.coef_
        if coefs.ndim == 2:
            importances = np.mean(np.abs(coefs), axis=0)
        else:
            importances = np.abs(coefs)
    
    if len(importances) > 0:
        fi_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)
        fi_df.to_csv(cfg.paths.fi_out, index=False)
        log.info(f"Feature importance saved to {cfg.paths.fi_out}")
    
    # SHAP explainability
    try:
        log.info("Computing SHAP values...")
        X_transformed, shap_values, feat_names = compute_shap_values(
            pipeline, X,
            sample_size=cfg.explain.shap_sample,
            model_type="xgb" if cfg.model.name == "xgb" else "logreg"
        )
        
        if shap_values is not None:
            # Save SHAP summary plot
            shap_plot_path = output_dir / "shap_summary.png"
            save_shap_summary(shap_values, X_transformed, feat_names, shap_plot_path, cfg.explain.max_display)
            
            # Save SHAP importance CSV
            shap_importance_path = output_dir / "feature_importance_shap.csv"
            save_shap_importance_csv(shap_values, feat_names, shap_importance_path)
            
            # Save per-sample top features
            top_features_df = get_top_features_per_sample(
                shap_values, feat_names, top_k=cfg.explain.per_sample_top_features
            )
            top_features_df.to_csv(output_dir / "shap_per_sample_top.csv", index=False)
            
            log.info("SHAP explainability completed")
    except Exception as e:
        log.warning(f"SHAP computation skipped: {e}")
    
    # Save metadata
    meta = {
        "task": "multiclass",
        "target_column": target_col,
        "n_classes": label_encoder.n_classes,
        "classes": label_encoder.classes_,
        "model_type": cfg.model.name,
        "top_k": cfg.model.top_k,
        "features": feature_names,
        "cv_accuracy": float(metrics["accuracy"]),
        "cv_macro_f1": float(metrics["macro_f1"]),
    }
    
    with open(cfg.paths.meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    
    log.info(f"Training complete! Artifacts saved to {output_dir}")
    log.info(f"✓ Model: {model_path}")
    log.info(f"✓ Label Encoder: {cfg.paths.le_out}")
    log.info(f"✓ Metadata: {cfg.paths.meta_out}")
    
    return meta


@main(config_path="configs", config_name="train", version_base=None)
def run(cfg: DictConfig) -> None:
    """Main training entrypoint."""
    train_multiclass(cfg)


if __name__ == "__main__":
    run()
