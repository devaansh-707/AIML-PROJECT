"""Streamlit application for interactive disease risk prediction and explainability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st


st.set_page_config(page_title="Disease Prediction Studio", layout="wide")


@st.cache_resource(show_spinner=False)
def load_artifacts(model_path: Path, meta_path: Path):
    model = joblib.load(model_path)
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    base_model_path = meta.get("base_model_path")
    if base_model_path:
        base_model_path = Path(base_model_path)
    if base_model_path and base_model_path.exists():
        base_model = joblib.load(base_model_path)
    else:
        base_model = model

    return model, base_model, meta


def compute_predictions(model, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    proba = model.predict_proba(df)
    if proba.ndim == 1 or proba.shape[1] == 1:
        proba = proba.reshape(-1, 1)

    if proba.shape[1] == 2:
        scores = proba[:, 1]
        preds = (scores >= threshold).astype(int)
        
        # Add user-friendly labels
        risk_labels = []
        for pred, prob in zip(preds, scores):
            if pred == 1:
                if prob >= 0.8:
                    risk_labels.append("🔴 HIGH RISK")
                elif prob >= 0.6:
                    risk_labels.append("🟡 MODERATE RISK")
                else:
                    risk_labels.append("🟢 POSITIVE")
            else:
                risk_labels.append("✅ NEGATIVE")
        
        result = pd.DataFrame(
            {
                "prediction": preds,
                "risk_level": risk_labels,
                "probability": scores,
            }
        )
    else:
        preds = np.argmax(proba, axis=1)
        proba_cols = [f"class_{idx}_proba" for idx in range(proba.shape[1])]
        result = pd.DataFrame({"prediction": preds})
        result = pd.concat([result, pd.DataFrame(proba, columns=proba_cols)], axis=1)
    return result


def compute_shap(base_pipeline, df: pd.DataFrame, sample_size: int = 200):
    features = base_pipeline.named_steps["features"]
    estimator = base_pipeline.named_steps["model"]
    feature_names = features.get_feature_names_out()

    if len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
    else:
        df_sample = df

    transformed = features.transform(df_sample)
    explainer = shap.Explainer(estimator, transformed, feature_names=feature_names)
    shap_values = explainer(transformed)
    return df_sample, shap_values, feature_names


def main():
    st.title("🩺 Disease Prediction Studio")
    st.write(
        "Upload patient records, generate predictions, and explore model explainability via SHAP "
        "global and per-patient insights."
    )

    default_model_path = Path("outputs/model.pkl")
    default_meta_path = Path("outputs/meta.json")

    with st.sidebar:
        st.header("Configuration")
        model_path = st.text_input("Model path", value=str(default_model_path))
        meta_path = st.text_input("Meta path", value=str(default_meta_path))
        threshold_override = st.number_input("Decision threshold", value=0.5, min_value=0.0, max_value=1.0, step=0.01)

    try:
        model, base_model, meta = load_artifacts(Path(model_path), Path(meta_path))
    except FileNotFoundError:
        st.error("Model or meta files not found. Train the model first via `make train`.")
        return
    except Exception as exc:  # pragma: no cover - user environment dependent
        st.error(f"Failed to load artifacts: {exc}")
        return

    target = meta.get("target")
    stored_threshold = meta.get("threshold", 0.5)
    threshold = threshold_override if threshold_override is not None else stored_threshold
    
    # Show required columns info
    required_cols = meta.get("schema_columns", [])
    st.info(f"📋 Required columns: {', '.join([c for c in required_cols if c != target])}")
    
    # Provide test data download if available
    test_data_path = Path("data/test.csv")
    if test_data_path.exists():
        with open(test_data_path, "rb") as f:
            st.download_button(
                "⬇️ Download Sample Test Data",
                data=f.read(),
                file_name="test_data_sample.csv",
                mime="text/csv",
                help="Use this file as a template for your data"
            )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if target and target in df.columns:
            df = df.drop(columns=[target])

        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
        # Validate required columns
        required_cols = meta.get("schema_columns", [])
        missing_cols = [col for col in required_cols if col not in df.columns and col != target]
        if missing_cols:
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            st.info(f"📋 Your CSV must have these columns: {', '.join(required_cols)}")
            st.info(f"✅ Use the test data at: data/test.csv")
            return

        if st.button("Run Predictions"):
            try:
                predictions = compute_predictions(model, df, threshold)
                
                # Show summary statistics
                if "prediction" in predictions.columns:
                    total = len(predictions)
                    positive = (predictions["prediction"] == 1).sum()
                    negative = (predictions["prediction"] == 0).sum()
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Patients", total)
                    col2.metric("🔴 Disease Detected", positive)
                    col3.metric("✅ Negative", negative)
                    
                    if positive > 0:
                        st.warning(f"⚠️ {positive} patient(s) show signs of disease - recommend further evaluation")
                    else:
                        st.success("✅ All patients appear healthy based on the model")
                
                st.subheader("Detailed Predictions")
                st.dataframe(predictions)

                st.download_button(
                    "Download predictions",
                    data=predictions.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.info("Make sure your CSV has all required columns with correct data types.")
                return

            st.subheader("Explainability")
            try:
                df_sample, shap_values, feature_names = compute_shap(base_model, df)
            except Exception as e:
                st.warning(f"⚠️ SHAP explainability unavailable: {str(e)}")
                return

            mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
            importance_df = (
                pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
                .sort_values("mean_abs_shap", ascending=False)
                .reset_index(drop=True)
            )

            st.markdown("**Global Feature Importance** (SHAP mean |value|)")
            st.bar_chart(importance_df.set_index("feature"))

            st.markdown("**Per-Row SHAP Explanation**")
            row_options = list(df_sample.index)
            selected_row = st.selectbox("Select row index", options=row_options)
            row_idx = df_sample.index.get_loc(selected_row)

            row_shap = shap_values.values[row_idx]
            row_df = (
                pd.DataFrame({"feature": feature_names, "shap_value": row_shap})
                .assign(abs_value=lambda d: np.abs(d["shap_value"]))
                .sort_values("abs_value", ascending=False)
            )

            st.write(f"Top contributions for row {selected_row}:")
            st.dataframe(row_df.head(20)[["feature", "shap_value"]])

            st.markdown("**SHAP Summary Plot**")
            shap.summary_plot(shap_values, show=False, plot_type="bar")
            fig = plt.gcf()
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("Awaiting CSV upload to start predictions.")


if __name__ == "__main__":
    main()

