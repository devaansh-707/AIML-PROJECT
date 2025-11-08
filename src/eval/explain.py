"""Standalone SHAP explainability utility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def generate_explanations(
    data_path: Path,
    output_dir: Path,
    base_model_path: Path,
    sample_size: int = 200,
    max_display: int = 20,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    base_pipeline = joblib.load(base_model_path)
    features = base_pipeline.named_steps["features"]
    estimator = base_pipeline.named_steps["model"]
    feature_names = features.get_feature_names_out()

    df = pd.read_csv(data_path)
    if sample_size < len(df):
        df_sample = df.sample(sample_size, random_state=42)
    else:
        df_sample = df

    transformed = features.transform(df_sample)
    explainer = shap.Explainer(estimator, transformed, feature_names=feature_names)
    shap_values = explainer(transformed)

    shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
    shap_df.insert(0, "index", df_sample.index)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SHAP-based explanations")
    parser.add_argument("--data", type=Path, required=True, help="Path to CSV data")
    parser.add_argument(
        "--meta",
        type=Path,
        default=Path("outputs/meta.json"),
        help="Path to meta.json with base pipeline reference",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Directory where explanation artifacts are saved",
    )
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--max-display", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.meta.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    base_model_path = meta.get("base_model_path")
    if base_model_path is None:
        raise ValueError("base_model_path missing from meta.json; retrain the model to regenerate")

    generate_explanations(
        data_path=args.data,
        output_dir=args.output,
        base_model_path=Path(base_model_path),
        sample_size=args.sample_size,
        max_display=args.max_display,
    )


if __name__ == "__main__":
    main()

