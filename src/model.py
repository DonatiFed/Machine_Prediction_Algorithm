# src/model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from lightgbm import LGBMRegressor


@dataclass(frozen=True)
class FitResult:
    name: str
    pipeline: Any
    metrics: Dict[str, float]


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def train_baseline(preprocessor, X_train, y_train, X_val, y_val) -> FitResult:
    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", DummyRegressor(strategy="median")),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_val)
    return FitResult("DummyRegressor(median)", pipe, regression_metrics(y_val, pred))


def train_lgbm_log_target(preprocessor, X_train, y_train, X_val, y_val) -> FitResult:
    y_train_log = np.log1p(y_train)

    model = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
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
    pred = np.expm1(pred_log)

    return FitResult("LightGBM(log1p)", pipe, regression_metrics(y_val, pred))


def predict_euros_from_log_model(pipe, X) -> np.ndarray:
    pred_log = pipe.predict(X)
    return np.expm1(pred_log)
