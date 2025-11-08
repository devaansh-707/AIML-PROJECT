# Disease Prediction Production Stack

State-of-the-art pipeline for tabular disease prediction with Hydra configs, feature engineering, Optuna tuning, explainable AI, and a Streamlit front-end. Built for reliability, experimentation, and hackathon-speed shipping.

## Project Structure
```
configs/            # Hydra configs for training + inference
schemas/            # Pandera validation schemas
src/                # Modular feature, model, tuning, eval code
tests/              # End-to-end regression tests
outputs/            # Default artifact location (gitignored)
app.py              # Streamlit experience
train.py / predict.py
Makefile            # One-command developer ergonomics
```

## 1. Setup
```bash
python -m venv .venv
. .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pre-commit install
```

Or run `make setup` after activating your virtualenv.

## 2. Training
Default paths expect `data/train.csv`. Configure everything through Hydra:
```bash
make train
# or
python train.py paths.train_csv=data/train.csv paths.output_dir=outputs model.name=xgboost
```

Artifacts saved to `outputs/` include:
- `model.pkl` – calibrated pipeline for inference
- `pipeline_base.pkl` – pre-calibration pipeline (used for SHAP)
- `meta.json` – metadata (target, threshold, schema)
- `cv_report.json` + `confusion_matrix.json`
- `feature_importance.csv`, `feature_importance_shap.csv`
- `shap_values.csv`, `shap_summary.png`

### Feature Engineering
- Leakage defence via pattern drop (`*discharge*`, `*bill*`)
- Median imputation + scaling
- Categorical OHE
- Missing-count feature
- Age binning
- Optional interaction features

Toggle features through `configs/train.yaml` (`features.*`).

### Model Stack
- XGBoost (default) and Logistic Regression
- Class imbalance handling (auto-calculated `scale_pos_weight` / class weights)
- Probability calibration (Platt + Isotonic)
- Threshold selection for F1 (binary)
- Macro-F1 for multiclass

### Evaluation
`src/eval/metrics.py` reports ROC-AUC, PR-AUC, Accuracy, Precision, Recall, F1, Confusion Matrix. Metrics saved as JSON for dashboards or CI gating.

## 3. Inference
Point to trained artifacts (defaults in `configs/infer.yaml`):
```bash
make infer
# or override
python predict.py model.path=outputs/model.pkl prediction.input_csv=data/test.csv
```

Outputs `outputs/submission.csv` with predictions and probabilities. Threshold can be overridden by config or CLI override `prediction.threshold=0.6`.

## 4. Hyperparameter Tuning
Optuna search with config-driven spaces:
```bash
python -m src.tuning.optuna_search optuna.enable=true optuna.n_trials=50
```
Study results stored under `outputs/` (`optuna_trials.csv`, `optuna_study.json`).

## 5. Explainability
During training and via `make explain` (uses `src/eval/explain.py`):
- `feature_importance_shap.csv` – global ranking (mean |SHAP|)
- `shap_summary.png` – beeswarm summary
- `shap_values.csv` – per-sample attributions (with row index)

Display or regenerate explanations for any CSV:
```bash
make explain                                     # defaults to data/test.csv
python -m src.eval.explain --data path/to.csv --output new_outputs/
```

## 6. Streamlit App
Launch the interactive UI:
```bash
make app
# or
streamlit run app.py
```
Features:
- Upload CSV and preview data
- Generate predictions + download batch
- Visualize global SHAP importance
- Inspect per-row SHAP contributions without leaving the browser

Provide custom artifact paths in the sidebar (`outputs/model.pkl`, `outputs/meta.json`).

## 7. Config System Overview
- Training config: `configs/train.yaml`
  - `paths.*` – data + artifact directories
  - `dataset.*` – target column, leak patterns, required columns
  - `features.*` – feature toggles (missing count, age bins, interactions)
  - `model.*` – estimator choice + hyperparams
  - `calibration.*` – methods + CV folds
  - `optuna.*` – search toggles
  - `shap.*` – sampling controls
- Inference config: `configs/infer.yaml`
  - `model.*` – artifact locations
  - `prediction.*` – input/output paths, thresholds

Hydra keeps runs reproducible; override any field via dot notation (`dataset.target=new_target`).

## 8. Metrics Cheatsheet
- **ROC-AUC** – ranking quality (binary, macro for multiclass)
- **PR-AUC** – focus on positives, useful for rare diseases
- **Accuracy** – overall correctness
- **Precision/Recall** – error trade-offs
- **F1 (macro)** – balances precision/recall across classes
- **Confusion Matrix** – saved as JSON for integrations

## 9. Testing & CI
- `make test` runs the synthetic E2E regression test (`tests/test_end_to_end.py`)
- GitHub Actions (`.github/workflows/ci.yml`) executes lint + tests
- Pre-commit hooks enforce Black, isort, Flake8, and hygiene checks

## 10. Docker
```bash
docker build -t disease-pred-app .
docker run -p 8501:8501 disease-pred-app
```
Container launches the Streamlit application.

## 11. Troubleshooting
- **Hydra working directory confusion**: configs pin `hydra.run.dir=.`; overrides persist in-place
- **xgboost missing**: reinstall dependencies (`make setup`) or switch `model.name=logistic`
- **Pandera validation failure**: check required columns (`dataset.required_columns`)
- **SHAP memory pressure**: lower `shap.sample_size` or use `make explain --sample-size 100`

## 12. Additional Resources
- Environment variables template: `.env.example`
- Default artifacts live under `outputs/` (gitignored). Clean with `rm -rf outputs` between experiments.

Happy building! Ship rigorous models with production-ready tooling and delightful UX.
