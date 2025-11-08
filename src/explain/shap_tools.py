"""SHAP explainability utilities for disease prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def create_shap_explainer(model, X_sample: np.ndarray, model_type: str = "xgb"):
    """
    Create appropriate SHAP explainer based on model type.
    
    Args:
        model: Trained model
        X_sample: Sample data for background
        model_type: 'xgb' or 'logreg'
    
    Returns:
        SHAP explainer object
    """
    if model_type == "xgb" or hasattr(model, "get_booster"):
        try:
            return shap.TreeExplainer(model)
        except Exception:
            # Fallback to kernel explainer
            return shap.KernelExplainer(model.predict_proba, X_sample[:100])
    else:
        # For logistic regression or other models
        return shap.KernelExplainer(model.predict_proba, X_sample[:100])


def compute_shap_values(
    pipeline,
    X: pd.DataFrame,
    sample_size: int = 2000,
    model_type: str = "xgb",
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Compute SHAP values for feature importance.
    
    Args:
        pipeline: Trained sklearn pipeline with 'features' and 'model' steps
        X: Input dataframe
        sample_size: Number of samples to use
        model_type: Model type for explainer selection
    
    Returns:
        Tuple of (transformed_data, shap_values, feature_names)
    """
    feature_processor = pipeline.named_steps["features"]
    model = pipeline.named_steps["model"]
    feature_names = list(feature_processor.get_feature_names_out())
    
    # Sample data
    if len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=42)
    else:
        X_sample = X
    
    # Transform
    X_transformed = feature_processor.transform(X_sample)
    
    # Create explainer
    try:
        explainer = create_shap_explainer(model, X_transformed, model_type)
        shap_values = explainer(X_transformed)
        return X_transformed, shap_values, feature_names
    except Exception as e:
        print(f"Warning: SHAP computation failed: {e}")
        return X_transformed, None, feature_names


def save_shap_summary(
    shap_values,
    X_transformed: np.ndarray,
    feature_names: list,
    output_path: Path,
    max_display: int = 20,
):
    """Save SHAP summary plot."""
    try:
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X_transformed,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not save SHAP summary plot: {e}")


def save_shap_importance_csv(
    shap_values,
    feature_names: list,
    output_path: Path,
):
    """Save SHAP-based feature importance as CSV."""
    try:
        shap_abs = np.abs(shap_values.values)
        shap_importance = shap_abs.mean(axis=0)
        
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": shap_importance,
        }).sort_values("mean_abs_shap", ascending=False)
        
        importance_df.to_csv(output_path, index=False)
    except Exception as e:
        print(f"Warning: Could not save SHAP importance CSV: {e}")


def get_top_features_per_sample(
    shap_values,
    feature_names: list,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Get top contributing features for each sample.
    
    Returns:
        DataFrame with columns: sample_idx, feature, shap_value
    """
    results = []
    
    for idx, sample_shap in enumerate(shap_values.values):
        # Get indices of top-k absolute SHAP values
        top_indices = np.argsort(np.abs(sample_shap))[-top_k:][::-1]
        
        for feat_idx in top_indices:
            results.append({
                "sample_idx": idx,
                "feature": feature_names[feat_idx],
                "shap_value": float(sample_shap[feat_idx]),
                "abs_shap": float(np.abs(sample_shap[feat_idx])),
            })
    
    return pd.DataFrame(results)
