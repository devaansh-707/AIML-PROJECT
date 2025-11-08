# Multi-Disease Prediction System

Production-grade multi-disease classifier that predicts exact disease names (e.g., Diabetes, Hypertension, Heart_Disease) with confidence scores and SHAP explanations.

## 🎯 Features

- **Multi-Class Classification**: One patient → one disease prediction
- **Top-K Predictions**: Shows top 3 most likely diseases with probabilities
- **Disease Name Output**: All predictions use human-readable disease names (not numeric IDs)
- **Imbalance Handling**: Automatic sample weighting based on class frequency
- **SHAP Explainability**: Global and per-sample feature importance
- **Streamlit App**: Interactive UI for batch predictions and analysis
- **Full Pipeline**: Feature engineering → training → prediction → explainability

## 📁 Project Structure

```
├── configs/
│   ├── train.yaml          # Training configuration
│   └── infer.yaml          # Inference configuration
├── src/
│   ├── models/
│   │   ├── label_encoder.py    # Disease name ↔ index mapping
│   │   ├── weights.py          # Imbalance weighting
│   │   └── build_model.py      # Model factory (XGB/LogReg)
│   ├── features/
│   │   └── build_features.py   # Feature engineering
│   ├── eval/
│   │   └── metrics_multiclass.py   # Metrics for multiclass
│   └── explain/
│       └── shap_tools.py       # SHAP utilities
├── train_multiclass.py     # Training script
├── predict_multiclass.py   # Prediction script
├── app_multidisease.py     # Streamlit app
├── outputs/
│   ├── model.pkl           # Trained model
│   ├── label_encoder.json  # Disease name mappings
│   ├── meta.json           # Model metadata
│   ├── cv_report.json      # Cross-validation metrics
│   ├── confusion_matrix.csv    # Confusion matrix
│   ├── classification_report.json  # Per-class metrics
│   ├── feature_importance.csv      # Feature importances
│   └── shap_summary.png    # SHAP summary plot
└── Makefile                # Quick commands
```

## 🚀 Quick Start

### 1. Setup

```bash
# Install dependencies
make setup
# or
python -m pip install -r requirements.txt
```

### 2. Prepare Data

Your training CSV must have:
- **Feature columns**: Patient data (age, bmi, symptoms, etc.)
- **Target column**: Disease names (e.g., "Diabetes", "Hypertension")

Example:
```csv
age,gender,bmi,blood_pressure,cholesterol,prognosis
45,M,28.5,130,220,Diabetes
67,F,31.2,150,240,Hypertension
52,M,26.8,125,195,Heart_Disease
```

Update `configs/train.yaml` to set your target column name:
```yaml
target:
  name: prognosis  # Your disease column name
```

### 3. Train Model

```bash
make train
# or
python train_multiclass.py
```

**Outputs**:
- `outputs/model.pkl` - Trained model
- `outputs/label_encoder.json` - Disease mappings
- `outputs/meta.json` - Metadata
- `outputs/cv_report.json` - CV metrics
- `outputs/confusion_matrix.csv` - Confusion matrix
- `outputs/shap_summary.png` - SHAP plot

### 4. Generate Predictions

```bash
make infer
# or
python predict_multiclass.py
```

**Outputs**:
- `outputs/submission.csv` - Predictions with top-K diseases
- `outputs/proba.csv` - All disease probabilities

**Prediction Format**:
```csv
prediction,top1_disease,top1_probability,top2_disease,top2_probability,top3_disease,top3_probability
Diabetes,Diabetes,0.87,Hypertension,0.08,Heart_Disease,0.03
Hypertension,Hypertension,0.92,Diabetes,0.05,Arthritis,0.02
```

### 5. Launch Web App

```bash
make app
# or
python -m streamlit run app_multidisease.py
```

**Features**:
- Upload CSV → get predictions
- View top-K diseases per patient
- See disease distribution
- Analyze individual patients
- Download results

## ⚙️ Configuration

### Training Config (`configs/train.yaml`)

```yaml
target:
  name: prognosis           # Target column with disease names
  multilabel: false         # Set true for multi-label (multiple diseases per patient)

model:
  name: xgb                 # xgb | logreg
  top_k: 3                  # Number of top predictions to show

imbalance:
  enable: true
  scheme: inverse_freq      # inverse_freq | balanced | none

cv:
  folds: 5
  stratify: true

explain:
  shap_sample: 2000         # Number of samples for SHAP
```

### Inference Config (`configs/infer.yaml`)

```yaml
paths:
  model: outputs/model.pkl
  label_encoder: outputs/label_encoder.json
  input: data/test.csv
  output: outputs/submission.csv
  proba_output: outputs/proba.csv

inference:
  top_k: 3                  # Number of top predictions
```

## 📊 Model Support

### Multiclass (Default)
- One disease per patient
- XGBoost with softmax
- Logistic Regression with multinomial

### Class Imbalance
- Automatic sample weighting
- Inverse frequency or balanced schemes
- Applied during training

### Evaluation Metrics
- **Macro-F1**: Average F1 across all diseases
- **Accuracy**: Overall correctness
- **Per-class F1/Precision/Recall**: Individual disease performance
- **ROC-AUC (OVR)**: One-vs-rest AUC
- **Confusion Matrix**: Disease confusion patterns

## 🔍 Explainability

### SHAP Summary Plot
Global feature importance across all predictions.

### Per-Sample Top Features
Most important features for each patient's prediction.

### Feature Importance CSV
Ranked features by importance or SHAP values.

## 📈 Example Output

### Training Log
```
INFO - Training multiclass model for 5 diseases: ['Arthritis', 'Asthma', 'Diabetes', 'Heart_Disease', 'Hypertension']
INFO - Computed sample weights with scheme: inverse_freq
INFO - Running 5-fold cross-validation...
INFO - CV Metrics - Accuracy: 0.8420, Macro-F1: 0.8315
INFO - Training final model on full dataset...
INFO - Model saved to outputs/model.pkl
INFO - SHAP explainability completed
INFO - Training complete!
```

### Prediction Log
```
INFO - Loaded model for 5 diseases
INFO - Task: multiclass
INFO - Generating predictions for 200 samples...
INFO - Predictions saved to outputs/submission.csv
INFO - Probabilities saved to outputs/proba.csv
INFO - Top predicted diseases:
INFO -   Arthritis: 69 patients (34.5%)
INFO -   Heart_Disease: 44 patients (22.0%)
INFO -   Diabetes: 38 patients (19.0%)
```

## 🎯 Acceptance Criteria ✅

- ✅ Training saves: `model.pkl`, `label_encoder.json`, `meta.json`, `cv_report.json`, feature importance, SHAP plot
- ✅ `predict_multiclass.py` produces `submission.csv` with disease names and `proba.csv` with all disease columns
- ✅ Streamlit app shows top-k predictions + probabilities without errors
- ✅ All outputs use disease NAMES not numeric IDs
- ✅ Imbalance handling with sample weights
- ✅ SHAP explainability integrated
- ✅ Makefile for easy commands

## 🐛 Troubleshooting

### Missing Required Columns
Update `configs/train.yaml` with your column names:
```yaml
dataset:
  required_columns: []  # Empty = auto-detect
```

### Imbalance Issues
Adjust weighting scheme:
```yaml
imbalance:
  enable: true
  scheme: balanced  # or inverse_freq
```

### SHAP Memory Issues
Reduce sample size:
```yaml
explain:
  shap_sample: 1000  # Lower for large datasets
```

## 📝 Notes

- **Multilabel support**: Set `target.multilabel: true` in config (uses OneVsRestClassifier)
- **Custom models**: Extend `src/models/build_model.py`
- **Custom features**: Modify `src/features/build_features.py`

## 🚀 Production Deployment

1. Train model with production data
2. Save artifacts (`model.pkl`, `label_encoder.json`, `meta.json`)
3. Deploy Streamlit app or use `predict_multiclass.py` for batch inference
4. Monitor predictions and retrain periodically

---

**Happy predicting! 🏥**
