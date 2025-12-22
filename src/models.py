# src/model.py
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


@dataclass(frozen=True)
class ModelResult:
    model_name: str
    metrics: Dict[str, float]
    model: Any


def _eval_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def train_baseline(preprocessor, X_train, y_train, X_val, y_val) -> ModelResult:
    # Baseline on raw target (no log)
    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", DummyRegressor(strategy="median")),
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    metrics = _eval_regression(y_val, preds)
    return ModelResult("DummyRegressor(median)", metrics, pipe)


def train_lgbm_log_target(preprocessor, X_train, y_train, X_val, y_val) -> ModelResult:
    if not HAS_LGBM:
        raise RuntimeError("LightGBM not installed. Run: pip install lightgbm")

    # Train on log1p target to handle right-skewed prices
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    model = LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", model),
    ])

    pipe.fit(X_train, y_train_log)

    pred_log = pipe.predict(X_val)
    pred = np.expm1(pred_log)  # back to euros
    metrics = _eval_regression(y_val, pred)

    return ModelResult("LightGBM(log1p)", metrics, pipe)
