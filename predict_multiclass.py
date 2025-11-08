"""Prediction script for multi-disease classification - outputs disease names."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from hydra import main
from omegaconf import DictConfig

from src.models.label_encoder import DiseaseLabelEncoder

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def predict_multiclass_topk(
    pipeline,
    label_encoder: DiseaseLabelEncoder,
    X: pd.DataFrame,
    top_k: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate predictions with top-K disease probabilities.
    
    Returns:
        (predictions_df, proba_df)
        - predictions_df: Contains prediction + top-k diseases with probabilities
        - proba_df: All disease probabilities (columns = disease names)
    """
    # Get probabilities
    y_proba = pipeline.predict_proba(X)
    y_pred_idx = np.argmax(y_proba, axis=1)
    
    # Convert indices to disease names
    y_pred_names = label_encoder.inverse_transform(y_pred_idx)
    
    # Build predictions dataframe with top-k
    pred_data = {"prediction": y_pred_names}
    
    for i in range(top_k):
        top_k_indices = np.argsort(y_proba, axis=1)[:, -(i+1)]
        top_k_probas = y_proba[np.arange(len(y_proba)), top_k_indices]
        top_k_names = label_encoder.inverse_transform(top_k_indices)
        
        pred_data[f"top{i+1}_disease"] = top_k_names
        pred_data[f"top{i+1}_probability"] = top_k_probas
    
    predictions_df = pd.DataFrame(pred_data)
    
    # Build probabilities dataframe (all diseases)
    proba_df = pd.DataFrame(y_proba, columns=label_encoder.classes_)
    
    return predictions_df, proba_df


def run_inference(cfg: DictConfig):
    """Run inference and save predictions."""
    # Load artifacts
    log.info("Loading model artifacts...")
    pipeline = joblib.load(cfg.paths.model)
    label_encoder = DiseaseLabelEncoder.load(Path(cfg.paths.label_encoder))
    
    with open(cfg.paths.meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    log.info(f"Loaded model for {len(label_encoder.classes_)} diseases")
    log.info(f"Task: {meta.get('task', 'multiclass')}")
    
    # Load input data
    log.info(f"Loading input data from {cfg.paths.input}")
    X = pd.read_csv(cfg.paths.input)
    
    # Remove target column if present
    target_col = meta.get("target_column")
    if target_col and target_col in X.columns:
        X = X.drop(columns=[target_col])
    
    log.info(f"Generating predictions for {len(X)} samples...")
    
    # Get top-k predictions
    top_k = cfg.inference.get("top_k", 3)
    predictions_df, proba_df = predict_multiclass_topk(
        pipeline, label_encoder, X, top_k=top_k
    )
    
    # Save predictions
    output_path = Path(cfg.paths.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)
    log.info(f"✓ Predictions saved to {output_path}")
    
    # Save full probabilities
    proba_path = Path(cfg.paths.proba_output)
    proba_df.to_csv(proba_path, index=False)
    log.info(f"✓ Probabilities saved to {proba_path}")
    
    # Show summary
    top_diseases = predictions_df["prediction"].value_counts().head(10)
    log.info("\nTop predicted diseases:")
    for disease, count in top_diseases.items():
        log.info(f"  {disease}: {count} patients ({count/len(X)*100:.1f}%)")
    
    return predictions_df


@main(config_path="configs", config_name="infer", version_base=None)
def run(cfg: DictConfig):
    """Main inference entrypoint."""
    run_inference(cfg)


if __name__ == "__main__":
    run()
