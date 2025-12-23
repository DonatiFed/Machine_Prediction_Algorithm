# src/evaluate.py
from __future__ import annotations
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


import json
import os
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def plot_target_log_hist(y: np.ndarray, out_path: str, bins: int = 60) -> None:
    """Histogram of log1p(target) to show right-skew + stabilization."""
    plt.figure()
    plt.hist(np.log1p(y), bins=bins)
    plt.title("Target distribution: log1p(Sales Price)")
    plt.xlabel("log1p(price)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_pred_vs_actual(
    y_true: np.ndarray, y_pred: np.ndarray, out_path: str, max_points: int = 25000
) -> None:
    """Scatter plot predicted vs actual (optionally subsampled for speed/size)."""
    n = len(y_true)
    if n > max_points:
        idx = np.random.RandomState(42).choice(n, size=max_points, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    plt.figure()
    plt.scatter(y_true, y_pred, s=3, alpha=0.25)
    lim = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([0, lim], [0, lim])
    plt.title("Predicted vs Actual")
    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_residuals(
    y_true: np.ndarray, y_pred: np.ndarray, out_path: str, bins: int = 60
) -> None:
    """Histogram of residuals (Actual - Predicted)."""
    residuals = y_true - y_pred
    plt.figure()
    plt.hist(residuals, bins=bins)
    plt.title("Residual distribution (Actual - Predicted)")
    plt.xlabel("residual")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_feature_importance_lgbm(
    pipe, out_path: str, top_n: int = 20
) -> None:
    """
    Horizontal bar chart of top-N feature importances.
    Works when pipe is a sklearn Pipeline with:
      - "prep" ColumnTransformer supporting get_feature_names_out()
      - "model" LightGBM model with feature_importances_
    """
    model = pipe.named_steps.get("model", None)
    prep = pipe.named_steps.get("prep", None)
    if model is None or prep is None:
        return
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    try:
        names = prep.get_feature_names_out()
    except Exception:
        names = np.array([f"f{i}" for i in range(len(importances))], dtype=object)

    idx = np.argsort(importances)[::-1][:top_n]
    labels = names[idx]
    vals = importances[idx]

    plt.figure()
    y_pos = np.arange(len(idx))
    plt.barh(y_pos, vals)
    plt.yticks(y_pos, labels)
    plt.gca().invert_yaxis()
    plt.title(f"Top {top_n} Feature Importances (LightGBM)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def evaluate_and_save(
    model,
    preprocessor,
    X_val,
    y_val,
    X_test,
    y_test,
    output_dir: str,
) -> None:
    """
    Evaluate a fitted model (FitResult) on val/test, save metrics + plots to output_dir.

    Assumes:
      - model.pipeline is a sklearn Pipeline with steps ("prep", "model")
      - model is trained on log1p(target) (LightGBM(log1p) in your case)
    """
    ensure_dir(output_dir)

    # --- VAL ---
    y_val_pred = np.expm1(model.pipeline.predict(X_val))
    val_metrics = regression_metrics(y_val.values, y_val_pred)
    save_json(val_metrics, os.path.join(output_dir, "val_metrics.json"))

    # --- TEST ---
    y_test_pred = np.expm1(model.pipeline.predict(X_test))
    test_metrics = regression_metrics(y_test.values, y_test_pred)
    save_json(test_metrics, os.path.join(output_dir, "test_metrics.json"))

    # --- PLOTS ---
    plot_pred_vs_actual(
        y_test.values,
        y_test_pred,
        os.path.join(output_dir, "pred_vs_actual_test.png"),
    )
    plot_residuals(
        y_test.values,
        y_test_pred,
        os.path.join(output_dir, "residuals_test.png"),
    )
    plot_feature_importance_lgbm(
        model.pipeline,
        os.path.join(output_dir, "feature_importance_top20.png"),
        top_n=20,
    )

    print("\nVAL metrics:", val_metrics)
    print("TEST metrics:", test_metrics)

# ---------------------------
# Self-test runner
# ---------------------------
if __name__ == "__main__":
    """
    Quick test: trains LightGBM via model.py and writes plots/metrics to outputs_eval_test/.
    Run:
      /Users/.../bin/python src/evaluate.py
    """
    from data import load_and_prepare
    from preprocess import build_preprocessor, drop_leakage_and_text
    from model import (
        train_lgbm_log_target,
        predict_euros_from_log_model,
        regression_metrics,
    )

    OUT_DIR = "outputs_eval_test"
    DATA_PATH = "BIT_AI_assignment_data.csv"  # adjust if needed

    ensure_dir(OUT_DIR)

    splits, _ = load_and_prepare(DATA_PATH)

    # Target distribution plot (train only)
    plot_target_log_hist(
        splits.y_train.values,
        os.path.join(OUT_DIR, "target_log_hist.png"),
    )

    # Build preprocessor (train only)
    artifacts = build_preprocessor(splits.X_train)
    preprocessor = artifacts.preprocessor

    X_train = drop_leakage_and_text(splits.X_train)
    X_val = drop_leakage_and_text(splits.X_val)
    X_test = drop_leakage_and_text(splits.X_test)

    # Train LGBM
    print("Training LightGBM for evaluation test...")
    fit = train_lgbm_log_target(preprocessor, X_train, splits.y_train, X_val, splits.y_val)
    print("VAL metrics:", fit.metrics)

    # Test predictions + metrics
    y_test_pred = predict_euros_from_log_model(fit.pipeline, X_test)
    test_metrics = regression_metrics(splits.y_test.values, y_test_pred)
    print("TEST metrics:", test_metrics)

    # Save metrics
    save_json(fit.metrics, os.path.join(OUT_DIR, "val_metrics_lgbm.json"))
    save_json(test_metrics, os.path.join(OUT_DIR, "test_metrics_lgbm.json"))

    # Plots
    plot_pred_vs_actual(
        splits.y_test.values,
        y_test_pred,
        os.path.join(OUT_DIR, "pred_vs_actual_test.png"),
    )
    plot_residuals(
        splits.y_test.values,
        y_test_pred,
        os.path.join(OUT_DIR, "residuals_test.png"),
    )
    plot_feature_importance_lgbm(
        fit.pipeline,
        os.path.join(OUT_DIR, "feature_importance_top20.png"),
        top_n=20,
    )

    print(f"Saved evaluation artifacts to: {OUT_DIR}")
