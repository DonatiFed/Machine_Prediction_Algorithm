# src/data.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ID_COLS = ["Sales ID", "Machine ID", "Model ID"]
TARGET_COL = "Sales Price"
DATE_COL = "Sales date"


@dataclass(frozen=True)
class DataSplit:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def load_raw_csv(path: str) -> pd.DataFrame:
    """Load raw CSV. Keep it simple and robust."""
    df = pd.read_csv(path, low_memory=False)
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal cleaning:
    - normalize missing tokens
    - parse target and date
    - create basic time features
    - drop pure identifiers
    """
    df = df.copy()

    # Normalize common missing tokens
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].replace(
                {"None or Unspecified": np.nan, "": np.nan, " ": np.nan}
            )

    # Target to numeric + filter invalid
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[TARGET_COL])
    df = df[df[TARGET_COL] > 0]

    # Parse date
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df["sale_year"] = df[DATE_COL].dt.year
        df["sale_month"] = df[DATE_COL].dt.month

    # Machine age feature (safe)
    if "Year Made" in df.columns and "sale_year" in df.columns:
        df["Year Made"] = pd.to_numeric(df["Year Made"], errors="coerce")
        df["machine_age"] = df["sale_year"] - df["Year Made"]
        df.loc[df["machine_age"] < 0, "machine_age"] = np.nan

    # Drop pure IDs if present
    drop_cols = [c for c in ID_COLS if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> DataSplit:
    """
    Split into train/val/test.
    val_size is fraction of full dataset (not of train).
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # First split off test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Then split train/val from remaining
    val_fraction_of_trainval = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_fraction_of_trainval,
        random_state=seed,
    )

    return DataSplit(X_train, X_val, X_test, y_train, y_val, y_test)


def load_and_prepare(path: str, seed: int = 42) -> Tuple[DataSplit, pd.DataFrame]:
    """Convenience function used by main.py."""
    raw = load_raw_csv(path)
    clean = basic_clean(raw)
    splits = split_data(clean, seed=seed)
    return splits, clean

def summarize_dataframe(df: pd.DataFrame) -> None:
    print("\n=== Data Summary ===")
    print("Shape:", df.shape)
    print("\nTarget describe:")
    print(df["Sales Price"].describe())
    print("\nTop missing columns:")
    print(df.isna().mean().sort_values(ascending=False).head(10))

    
# src/data.py
if __name__ == "__main__":
    PATH = "BIT_AI_assignment_data.csv"  

    splits, df = load_and_prepare(PATH)

    print("Clean dataset shape:", df.shape)
    print("Train shape:", splits.X_train.shape)
    print("Val shape:", splits.X_val.shape)
    print("Test shape:", splits.X_test.shape)

    print("\nTarget stats:")
    print(splits.y_train.describe())

    print("\nMissing target:", splits.y_train.isna().sum())