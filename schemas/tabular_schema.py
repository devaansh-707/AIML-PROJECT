"""Pandera schema definitions and validation helpers for tabular data."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema


# Column-specific range checks; extend as needed for project-specific features.
COLUMN_CHECKS: Dict[str, List[Check]] = {
    "age": [Check.ge(0), Check.le(120)],
    "bmi": [Check.ge(0), Check.le(100)],
    "heart_rate": [Check.ge(20), Check.le(220)],
}


def _infer_pandera_type(series: pd.Series) -> pa.DataType:
    if pd.api.types.is_bool_dtype(series):
        return pa.Bool()
    if pd.api.types.is_integer_dtype(series):
        return pa.Int64()
    if pd.api.types.is_float_dtype(series):
        return pa.Float64()
    return pa.String()


def _nullable(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return bool(series.isna().any())
    return True


def build_schema(
    df: pd.DataFrame,
    target: str,
    required_columns: Optional[Iterable[str]] = None,
    categorical_overrides: Optional[Iterable[str]] = None,
    numerical_overrides: Optional[Iterable[str]] = None,
) -> DataFrameSchema:
    """Create a pandera schema from a dataframe and config metadata."""

    required_columns = set(required_columns or [])
    categorical_overrides = set(categorical_overrides or [])
    numerical_overrides = set(numerical_overrides or [])

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    columns: Dict[str, Column] = {}

    for col in df.columns:
        if col == target:
            continue

        series = df[col]
        if col in categorical_overrides:
            dtype = pa.String()
        elif col in numerical_overrides:
            dtype = pa.Float64()
        else:
            dtype = _infer_pandera_type(series)

        checks = COLUMN_CHECKS.get(col, [])
        columns[col] = Column(dtype, nullable=_nullable(series), checks=checks)

    return DataFrameSchema(columns)


def validate_training_dataframe(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> DataFrameSchema:
    dataset_cfg = config.get("dataset", {})
    # Support both legacy `dataset.target` and newer `target.name`
    target_name = dataset_cfg.get("target")
    if target_name is None:
        target_name = (config.get("target") or {}).get("name")
    schema = build_schema(
        df,
        target=target_name,
        required_columns=dataset_cfg.get("required_columns"),
        categorical_overrides=dataset_cfg.get("categorical_overrides"),
        numerical_overrides=dataset_cfg.get("numerical_overrides"),
    )

    schema.validate(df.drop(columns=[target_name], errors="ignore"))
    return schema


def validate_inference_dataframe(
    df: pd.DataFrame,
    schema: DataFrameSchema,
) -> None:
    schema.validate(df)

