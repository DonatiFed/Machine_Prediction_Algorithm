# src/preprocess.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------------------
# Columns to drop
# ---------------------------------------------------------------------
# keep engineered time features (sale_year, sale_month) and drop the raw datetime/text-like fields.
DROP_COLS = [
    "Sales date",
    "Model Description",
    "Secondary Description",
]


# ---------------------------------------------------------------------
# Returned artifacts
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class PreprocessArtifacts:
    preprocessor: ColumnTransformer
    numeric_cols: List[str]
    categorical_cols: List[str]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def drop_leakage_and_text(X: pd.DataFrame) -> pd.DataFrame:
    """Drop raw date/text columns consistently across train/val/test."""
    return X.drop(columns=[c for c in DROP_COLS if c in X.columns], errors="ignore")


# ---------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------
def build_preprocessor(
    X: pd.DataFrame,
    min_category_freq: int = 100,
) -> PreprocessArtifacts:
    
    X = drop_leakage_and_text(X)

    numeric_selector = make_column_selector(dtype_include=np.number)
    categorical_selector = make_column_selector(dtype_exclude=np.number)

    numeric_cols = list(numeric_selector(X))
    categorical_cols = list(categorical_selector(X))

    numeric_pipe = Pipeline(
        steps=[
            # add_indicator=True appends binary flags like "missingindicator_<col>"
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=min_category_freq,
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return PreprocessArtifacts(preprocessor, numeric_cols, categorical_cols)


# ---------------------------------------------------------------------
# Debug utility (optional, for local inspection)
# ---------------------------------------------------------------------
def debug_preprocessor(artifacts: PreprocessArtifacts, X: pd.DataFrame) -> None:
    """Quick sanity check: column counts, transformed shape, sample feature names."""
    print("\n=== Preprocessing Debug ===")
    print("Numeric columns:", len(artifacts.numeric_cols))
    print("Categorical columns:", len(artifacts.categorical_cols))

    Xt = artifacts.preprocessor.fit_transform(drop_leakage_and_text(X))
    print("Transformed shape:", Xt.shape)

    
    try:
        names = artifacts.preprocessor.get_feature_names_out()
        print("\nSample transformed feature names:")
        for n in names[:20]:
            print(" -", n)
    except Exception as e:
        print("\nCould not extract feature names:", e)

    if Xt.shape[1] > 5000:
        print("⚠️ High feature count — consider increasing min_frequency")


# ---------------------------------------------------------------------
# Standalone run (debug only)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from data import load_and_prepare

    DATA_PATH = "BIT_AI_assignment_data.csv"  

    splits, _ = load_and_prepare(DATA_PATH)

    artifacts = build_preprocessor(splits.X_train)
    debug_preprocessor(artifacts, splits.X_train)
