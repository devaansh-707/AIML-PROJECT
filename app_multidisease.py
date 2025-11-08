"""Streamlit app for multi-disease prediction with top-K results and SHAP."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.models.label_encoder import DiseaseLabelEncoder

st.set_page_config(page_title="Multi-Disease Prediction Studio", layout="wide")


@st.cache_resource(show_spinner=False)
def load_artifacts(model_path: Path, meta_path: Path, le_path: Path):
    """Load model, metadata, and label encoder."""
    pipeline = joblib.load(model_path)
    label_encoder = DiseaseLabelEncoder.load(le_path)
    
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    return pipeline, label_encoder, meta


def predict_with_topk(pipeline, label_encoder: DiseaseLabelEncoder, X: pd.DataFrame, top_k: int = 3):
    """Generate predictions with top-K disease probabilities."""
    y_proba = pipeline.predict_proba(X)
    y_pred_idx = np.argmax(y_proba, axis=1)
    
    # Convert indices to disease names
    y_pred_names = label_encoder.inverse_transform(y_pred_idx)
    
    # Build predictions dataframe with top-k
    pred_data = {"predicted_disease": y_pred_names}
    
    for i in range(top_k):
        top_k_indices = np.argsort(y_proba, axis=1)[:, -(i+1)]
        top_k_probas = y_proba[np.arange(len(y_proba)), top_k_indices]
        top_k_names = label_encoder.inverse_transform(top_k_indices)
        
        pred_data[f"rank{i+1}_disease"] = top_k_names
        pred_data[f"rank{i+1}_probability"] = [f"{p:.1%}" for p in top_k_probas]
    
    predictions_df = pd.DataFrame(pred_data)
    
    # Build full probabilities dataframe
    proba_df = pd.DataFrame(y_proba, columns=label_encoder.classes_)
    
    return predictions_df, proba_df


def main():
    st.title("🏥 Multi-Disease Prediction Studio")
    st.markdown("""
    Upload patient records to predict diseases with confidence scores.
    The model provides **top-K most likely diseases** for each patient with SHAP-based explanations.
    """)
    
    # Configuration sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        model_path = st.text_input("Model Path", value="outputs/model.pkl")
        meta_path = st.text_input("Metadata Path", value="outputs/meta.json")
        le_path = st.text_input("Label Encoder Path", value="outputs/label_encoder.json")
        top_k_display = st.slider("Top-K Diseases to Show", 1, 5, 3)
    
    # Load artifacts
    try:
        pipeline, label_encoder, meta = load_artifacts(
            Path(model_path), Path(meta_path), Path(le_path)
        )
    except FileNotFoundError as e:
        st.error(f"❌ Artifacts not found: {e}")
        st.info("Run training first: `python train_multiclass.py`")
        return
    except Exception as e:
        st.error(f"❌ Failed to load artifacts: {e}")
        return
    
    # Display model info
    st.success(f"✅ Model loaded: {meta.get('model_type', 'unknown')} for {len(label_encoder.classes_)} diseases")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Task Type", meta.get("task", "multiclass"))
    col2.metric("Number of Diseases", len(label_encoder.classes_))
    col3.metric("CV Macro-F1", f"{meta.get('cv_macro_f1', 0):.3f}")
    
    with st.expander("📋 Disease List"):
        st.write(", ".join(label_encoder.classes_))
    
    # File uploader
    st.markdown("---")
    st.subheader("📤 Upload Patient Data")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Remove target column if present
        target_col = meta.get("target_column")
        if target_col and target_col in df.columns:
            df = df.drop(columns=[target_col])
        
        # Check for required columns
        required_features = ["age", "gender", "bmi", "heart_rate", "blood_pressure", "cholesterol"]
        missing_cols = [col for col in required_features if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            st.warning(f"**Your CSV has**: {', '.join(df.columns.tolist())}")
            st.info(f"**Model needs**: {', '.join(required_features)}")
            st.info("💡 Use the sample test data below or ensure your CSV has all required columns")
            return
        
        st.markdown("### 📊 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Showing 10 of {len(df)} patients")
        
        # Predict button
        if st.button("🔮 Generate Predictions", type="primary"):
            with st.spinner("Analyzing patient data..."):
                predictions_df, proba_df = predict_with_topk(
                    pipeline, label_encoder, df, top_k=top_k_display
                )
            
            st.markdown("### 🎯 Prediction Results")
            
            # Summary stats
            disease_counts = predictions_df["predicted_disease"].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📈 Disease Distribution")
                st.bar_chart(disease_counts)
            
            with col2:
                st.markdown("#### 🔝 Top Predicted Diseases")
                for disease, count in disease_counts.head(5).items():
                    pct = count / len(predictions_df) * 100
                    st.metric(disease.replace("_", " "), f"{count} patients ({pct:.1f}%)")
            
            # Detailed predictions table
            st.markdown("### 📋 Detailed Predictions")
            st.dataframe(predictions_df, use_container_width=True)
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    "⬇️ Download Predictions",
                    data=predictions_df.to_csv(index=False).encode("utf-8"),
                    file_name="disease_predictions.csv",
                    mime="text/csv",
                )
            
            with col2:
                st.download_button(
                    "⬇️ Download All Probabilities",
                    data=proba_df.to_csv(index=False).encode("utf-8"),
                    file_name="disease_probabilities.csv",
                    mime="text/csv",
                )
            
            # Per-patient analysis
            st.markdown("---")
            st.markdown("### 🔍 Individual Patient Analysis")
            
            patient_idx = st.selectbox(
                "Select patient to analyze:",
                options=range(len(predictions_df)),
                format_func=lambda i: f"Patient {i+1}"
            )
            
            if patient_idx is not None:
                st.markdown(f"#### Patient #{patient_idx+1} Details")
                
                # Show top predictions
                pred_row = predictions_df.iloc[patient_idx]
                
                cols = st.columns(top_k_display)
                for i in range(top_k_display):
                    with cols[i]:
                        disease = pred_row[f"rank{i+1}_disease"]
                        prob = pred_row[f"rank{i+1}_probability"]
                        st.metric(f"Rank {i+1}", disease.replace("_", " "), prob)
                
                # Show all disease probabilities
                st.markdown("##### All Disease Probabilities")
                patient_proba = proba_df.iloc[patient_idx].sort_values(ascending=False)
                proba_display_df = pd.DataFrame({
                    "Disease": [d.replace("_", " ") for d in patient_proba.index],
                    "Probability": [f"{p:.1%}" for p in patient_proba.values]
                })
                st.dataframe(proba_display_df, use_container_width=True, height=300)
    
    else:
        st.info("👆 Upload a CSV file to begin predictions")
        
        # Show sample data download if available
        test_data_path = Path("data/test.csv")
        if test_data_path.exists():
            with open(test_data_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Sample Test Data",
                    data=f.read(),
                    file_name="sample_test_data.csv",
                    mime="text/csv",
                    help="Use this file as a template for your data"
                )


if __name__ == "__main__":
    main()
