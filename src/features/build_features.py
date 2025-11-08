"""Feature preprocessing pipeline with leakage defense and engineering."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FeaturePreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible feature preprocessor.
    
    - Drops leakage columns matching patterns
    - Imputes missing values (median for numeric, mode for categorical)
    - Scales numerical features
    - One-hot encodes categorical features
    - Optional: missing count feature, age binning, interaction features
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.numerical_cols_ = None
        self.categorical_cols_ = None
        self.feature_names_ = None
        self.num_impute_values_ = {}
        self.cat_impute_values_ = {}
        self.scaler_ = None
        self.encoder_ = None
        self.age_bins_ = None

    def fit(self, X: pd.DataFrame, y=None):
        """Learn preprocessing parameters from training data."""
        X = X.copy()
        
        # Drop leakage columns
        drop_patterns = self.config.get("dataset", {}).get("drop_patterns", [])
        for pattern in drop_patterns:
            pattern_str = pattern.replace("*", "")
            cols_to_drop = [c for c in X.columns if pattern_str.lower() in c.lower()]
            X = X.drop(columns=cols_to_drop, errors="ignore")
        
        # Identify numerical and categorical columns
        self.numerical_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols_ = X.select_dtypes(include=["object", "category"]).columns.tolist()
        
        # Handle user overrides
        cat_override = self.config.get("dataset", {}).get("categorical_overrides", [])
        num_override = self.config.get("dataset", {}).get("numerical_overrides", [])
        
        for col in cat_override:
            if col in self.numerical_cols_:
                self.numerical_cols_.remove(col)
            if col not in self.categorical_cols_ and col in X.columns:
                self.categorical_cols_.append(col)
        
        for col in num_override:
            if col in self.categorical_cols_:
                self.categorical_cols_.remove(col)
            if col not in self.numerical_cols_ and col in X.columns:
                self.numerical_cols_.append(col)
        
        # Learn imputation values
        for col in self.numerical_cols_:
            self.num_impute_values_[col] = X[col].median()
        
        for col in self.categorical_cols_:
            mode_val = X[col].mode()
            self.cat_impute_values_[col] = mode_val[0] if len(mode_val) > 0 else "missing"
        
        # Fit scaler on numerical features
        X_num_imputed = X[self.numerical_cols_].copy()
        for col in self.numerical_cols_:
            X_num_imputed[col].fillna(self.num_impute_values_[col], inplace=True)
        
        scaler_type = self.config.get("features", {}).get("scaler", "standard")
        if scaler_type == "standard":
            self.scaler_ = StandardScaler()
            self.scaler_.fit(X_num_imputed)
        
        # Fit one-hot encoder on categorical features
        X_cat_imputed = X[self.categorical_cols_].copy()
        for col in self.categorical_cols_:
            X_cat_imputed[col].fillna(self.cat_impute_values_[col], inplace=True)
        
        if len(self.categorical_cols_) > 0:
            self.encoder_ = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            self.encoder_.fit(X_cat_imputed)
        
        # Determine age bins if enabled
        if self.config.get("features", {}).get("enable_age_binning", False):
            if "age" in self.numerical_cols_:
                age_data = X["age"].dropna()
                self.age_bins_ = [0, 30, 45, 60, 120]
        
        # Build feature names
        self._build_feature_names()
        
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform input data using learned preprocessing parameters."""
        X = X.copy()
        
        # Drop leakage columns
        drop_patterns = self.config.get("dataset", {}).get("drop_patterns", [])
        for pattern in drop_patterns:
            pattern_str = pattern.replace("*", "")
            cols_to_drop = [c for c in X.columns if pattern_str.lower() in c.lower()]
            X = X.drop(columns=cols_to_drop, errors="ignore")
        
        features_list = []
        
        # Missing count feature
        if self.config.get("features", {}).get("enable_missing_count", False):
            missing_count = X.isnull().sum(axis=1).values.reshape(-1, 1)
            features_list.append(missing_count)
        
        # Numerical features (impute + scale)
        X_num = X[self.numerical_cols_].copy()
        for col in self.numerical_cols_:
            X_num[col].fillna(self.num_impute_values_[col], inplace=True)
        
        if self.scaler_ is not None:
            X_num_scaled = self.scaler_.transform(X_num)
        else:
            X_num_scaled = X_num.values
        
        features_list.append(X_num_scaled)
        
        # Age binning
        if self.config.get("features", {}).get("enable_age_binning", False) and self.age_bins_:
            if "age" in self.numerical_cols_:
                age_idx = self.numerical_cols_.index("age")
                age_vals = X["age"].fillna(self.num_impute_values_.get("age", 40)).values
                age_binned = np.digitize(age_vals, bins=self.age_bins_).reshape(-1, 1)
                features_list.append(age_binned)
        
        # Categorical features (impute + one-hot encode)
        if len(self.categorical_cols_) > 0:
            X_cat = X[self.categorical_cols_].copy()
            for col in self.categorical_cols_:
                X_cat[col].fillna(self.cat_impute_values_[col], inplace=True)
            
            if self.encoder_ is not None:
                X_cat_encoded = self.encoder_.transform(X_cat)
                features_list.append(X_cat_encoded)
        
        # Interaction features (if enabled)
        if self.config.get("features", {}).get("enable_interactions", False):
            interaction_pairs = self.config.get("features", {}).get("interaction_pairs", [])
            for pair in interaction_pairs:
                if len(pair) == 2 and pair[0] in self.numerical_cols_ and pair[1] in self.numerical_cols_:
                    idx_0 = self.numerical_cols_.index(pair[0])
                    idx_1 = self.numerical_cols_.index(pair[1])
                    interaction = (X_num_scaled[:, idx_0] * X_num_scaled[:, idx_1]).reshape(-1, 1)
                    features_list.append(interaction)
        
        # Concatenate all features
        X_transformed = np.hstack(features_list)
        return X_transformed

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Return feature names after transformation."""
        return self.feature_names_

    def _build_feature_names(self):
        """Build the list of output feature names."""
        names = []
        
        # Missing count
        if self.config.get("features", {}).get("enable_missing_count", False):
            names.append("missing_count")
        
        # Numerical features
        names.extend(self.numerical_cols_)
        
        # Age binning
        if self.config.get("features", {}).get("enable_age_binning", False) and self.age_bins_:
            if "age" in self.numerical_cols_:
                names.append("age_bin")
        
        # Categorical one-hot encoded features
        if self.encoder_ is not None:
            cat_feature_names = self.encoder_.get_feature_names_out(self.categorical_cols_)
            names.extend(cat_feature_names)
        
        # Interaction features
        if self.config.get("features", {}).get("enable_interactions", False):
            interaction_pairs = self.config.get("features", {}).get("interaction_pairs", [])
            for pair in interaction_pairs:
                if len(pair) == 2:
                    names.append(f"{pair[0]}_x_{pair[1]}")
        
        self.feature_names_ = names
