# src/model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


@dataclass(frozen=True)
class FitResult:
    name: str
    pipeline: Any
    metrics: Dict[str, float]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """MAE, RMSE, R2 on original euro scale (sklearn-version safe)."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def train_baseline(preprocessor, X_train, y_train, X_val, y_val) -> FitResult:
    """Median baseline."""
    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", DummyRegressor(strategy="median")),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_val)
    metrics = regression_metrics(y_val, pred)
    return FitResult(name="DummyRegressor(median)", pipeline=pipe, metrics=metrics)


def train_lgbm_log_target(preprocessor, X_train, y_train, X_val, y_val) -> FitResult:
    """
    LightGBM trained on log1p(y).
    Predictions converted back to euros for metrics.
    """
    if not HAS_LGBM:
        raise RuntimeError("LightGBM not installed. Run: pip install lightgbm")

    y_train_log = np.log1p(y_train)

    model = LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        force_row_wise=True,
        verbose=-1,
    )

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", model),
    ])

    pipe.fit(X_train, y_train_log)

    pred_log = pipe.predict(X_val)
    pred = np.expm1(pred_log)

    metrics = regression_metrics(y_val, pred)
    return FitResult(name="LightGBM(log1p)", pipeline=pipe, metrics=metrics)


def predict_euros_from_log_model(pipe, X) -> np.ndarray:
    """Predict euros from a log-trained pipeline."""
    pred_log = pipe.predict(X)
    return np.expm1(pred_log)


# ---------------------------
# Self-test runner
# ---------------------------
if __name__ == "__main__":
    from data import load_and_prepare
    from preprocess import build_preprocessor, drop_leakage_and_text

    DATA_PATH = "BIT_AI_assignment_data.csv"  # adjust if needed

    splits, _ = load_and_prepare(DATA_PATH)

    artifacts = build_preprocessor(splits.X_train)
    preprocessor = artifacts.preprocessor

    X_train = drop_leakage_and_text(splits.X_train)
    X_val = drop_leakage_and_text(splits.X_val)
    X_test = drop_leakage_and_text(splits.X_test)

    print("Training baseline...")
    baseline = train_baseline(preprocessor, X_train, splits.y_train, X_val, splits.y_val)
    print("VAL baseline metrics:", baseline.metrics)

    if HAS_LGBM:
        print("\nTraining LightGBM...")
        lgbm = train_lgbm_log_target(preprocessor, X_train, splits.y_train, X_val, splits.y_val)
        print("VAL LightGBM metrics:", lgbm.metrics)

        y_test_pred = predict_euros_from_log_model(lgbm.pipeline, X_test)
        test_metrics = regression_metrics(splits.y_test.values, y_test_pred)
        print("\nTEST LightGBM metrics:", test_metrics)
    else:
        print("\nLightGBM not installed. Skipping.")
