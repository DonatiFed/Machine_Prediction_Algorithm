# src/data.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ID_COLS = ["Sales ID", "Machine ID", "Model ID"]
TARGET_COL = "Sales Price"
DATE_COL = "Sales date"

# Columns that are truly numeric (or numeric IDs)
NUMERIC_LIKE_COLS = [
    "Year Made",
    "MachineHours CurrentMeter",
    "Auctioneer ID",
]


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
    return pd.read_csv(path, low_memory=False)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal, safe cleaning:
    - normalize missing tokens
    - parse target and date
    - create basic time features
    - create machine_age
    - coerce truly numeric columns
    - drop pure identifiers
    """
    df = df.copy()
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

    # Normalize common missing tokens (only for object columns)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].replace(
                {"None or Unspecified": np.nan, "": np.nan, " ": np.nan}
            )

    # Target to numeric + filter invalid
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[TARGET_COL])
    df = df[df[TARGET_COL] > 0]

    # Parse date + safe time features
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df["sale_year"] = df[DATE_COL].dt.year
        df["sale_month"] = df[DATE_COL].dt.month

    # Coerce truly numeric-like columns
    for c in NUMERIC_LIKE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Machine age feature (sale_year - Year Made)
    if "sale_year" in df.columns and "Year Made" in df.columns:
        df["machine_age"] = df["sale_year"] - df["Year Made"]
        df.loc[df["machine_age"] < 0, "machine_age"] = np.nan

    if "Auctioneer ID" in df.columns:
        df["Auctioneer ID"] = df["Auctioneer ID"].astype("Int64").astype(str)
        df.loc[df["Auctioneer ID"] == "<NA>", "Auctioneer ID"] = np.nan


    # Drop pure IDs
    df = df.drop(columns=[c for c in ID_COLS if c in df.columns], errors="ignore")

    return df


def analyze_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary DataFrame sorted by missingness percentage."""
    missing_count = df.isnull().sum()
    missing_stats = pd.DataFrame(
        {
            "column": missing_count.index,
            "missing_count": missing_count.values,
            "missing_percent": (missing_count.values / len(df) * 100).round(2),
            "dtype": df.dtypes.values,
        }
    )
    missing_stats = (
        missing_stats[missing_stats["missing_count"] > 0]
        .sort_values("missing_percent", ascending=False)
        .reset_index(drop=True)
    )
    return missing_stats


def print_missing_data_report(df: pd.DataFrame) -> None:
    """
    Print an informative missing data report.

    High missingness does NOT imply dropping the column.
    We keep features and handle missingness during preprocessing.
    """
    print("\n" + "=" * 70)
    print("MISSING DATA ANALYSIS")
    print("=" * 70)

    missing_report = analyze_missing_data(df)

    print(f"\nTotal rows: {len(df):,}")
    print(f"Columns: {df.shape[1]}")
    print(f"Columns with missing data: {len(missing_report)}/{df.shape[1]}")

    if len(missing_report) == 0:
        print("\n✓ No missing data found")
        print("=" * 70 + "\n")
        return

    print("\nMissing Data Summary (top 25):\n")
    print(missing_report.head(25).to_string(index=False))

    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)
    print(
        """
• High missingness features may still carry signal via presence/absence
• Categorical features: impute with 'Unknown'
• Numeric features: median imputation + optional missingness indicators
"""
    )

    print("\n" + "-" * 70)
    print("IMPUTATION STRATEGY (handled in preprocess.py)")
    print("-" * 70)
    print(
        """
Numeric:
  • Median imputation (robust to outliers)
  • Missingness indicators (optional, recommended)

Categorical:
  • Fill missing with 'Unknown'
"""
    )
    print("=" * 70 + "\n")


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> DataSplit:
    """Split into train / validation / test."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    val_fraction = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_fraction,
        random_state=seed,
    )

    return DataSplit(X_train, X_val, X_test, y_train, y_val, y_test)


def load_and_prepare(path: str, seed: int = 42) -> Tuple[DataSplit, pd.DataFrame]:
    """Main entry used by main.py."""
    raw = load_raw_csv(path)
    clean = basic_clean(raw)
    splits = split_data(clean, seed=seed)
    return splits, clean


if __name__ == "__main__":
    PATH = "BIT_AI_assignment_data.csv"  # adjust if needed

    splits, df = load_and_prepare(PATH)

    print("Clean dataset shape:", df.shape)
    print("Train shape:", splits.X_train.shape)
    print("Val shape:", splits.X_val.shape)
    print("Test shape:", splits.X_test.shape)

    print("\nTarget stats:")
    print(splits.y_train.describe())

    print("\nMissing target:", splits.y_train.isna().sum())

    print_missing_data_report(splits.X_train)
