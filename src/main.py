# src/main.py
from __future__ import annotations

import json
from pathlib import Path
import warnings
import joblib

import numpy as np

from data import load_and_prepare
from preprocess import build_preprocessor, drop_leakage_and_text
from model import train_lgbm_log_target, train_baseline
from evaluate import evaluate_and_save

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
)

DATA_PATH = "BIT_AI_assignment_data.csv"
OUTPUT_DIR = "outputs"
MODEL_DIR = "artifacts"
SEED = 42


class MachinePricePredictor:
    """
    Small wrapper around the end-to-end pipeline:
      - preprocessing (imputation + OHE)
      - LightGBM regression trained on log1p(target)
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.preprocessor = None
        self.model_fit = None  # FitResult from model.py
        self.pipeline = None   # sklearn Pipeline (prep + model)

    def fit(self, X_train, y_train, X_val, y_val) -> None:
        """Fit preprocessor + model."""
        artifacts = build_preprocessor(X_train)
        self.preprocessor = artifacts.preprocessor

        self.model_fit = train_lgbm_log_target(
            self.preprocessor,
            X_train,
            y_train,
            X_val,
            y_val,
            random_state=self.random_state,
        )

        # Keep a direct handle to the fitted sklearn pipeline for save/load/predict.
        self.pipeline = self.model_fit.pipeline

    def predict(self, X) -> np.ndarray:
        """
        Predict prices in euros.

        Note: the underlying model is trained on log1p(price),
        so we invert that transformation here.
        """
        if self.pipeline is None:
            raise RuntimeError("Model not trained or loaded.")
        pred_log = self.pipeline.predict(X)
        return np.expm1(pred_log)

    def evaluate(self, X_val, y_val, X_test, y_test, output_dir: str) -> None:
        """Run evaluation and save metrics/plots."""
        if self.model_fit is None:
            raise RuntimeError("Model not trained (evaluation expects FitResult).")
        evaluate_and_save(
            model=self.model_fit,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            output_dir=output_dir,
        )

    def save(self, path: str) -> None:
        """Persist the fitted pipeline to disk."""
        if self.pipeline is None:
            raise RuntimeError("Nothing to save: model not trained.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "random_state": self.random_state,
            },
            path,
        )

    @staticmethod
    def load(path: str) -> "MachinePricePredictor":
        """Load a trained predictor from disk."""
        data = joblib.load(path)
        predictor = MachinePricePredictor(random_state=data["random_state"])
        predictor.pipeline = data["pipeline"]
        return predictor


def main() -> None:
    print("=== Machine Price Prediction Pipeline ===")

    # 1) Load and clean data
    print("\n[1/6] Loading and cleaning data...")
    splits, _ = load_and_prepare(DATA_PATH, seed=SEED)

    X_train = drop_leakage_and_text(splits.X_train)
    X_val = drop_leakage_and_text(splits.X_val)
    X_test = drop_leakage_and_text(splits.X_test)

    # 2) Build preprocessing (train-only) for baseline comparison
    print("\n[2/6] Building preprocessing pipeline...")
    artifacts = build_preprocessor(X_train)
    preprocessor = artifacts.preprocessor

    # 3) Baseline (business baseline)
    print("\n[3/6] Training baseline (median predictor)...")
    baseline_fit = train_baseline(
        preprocessor,
        X_train,
        splits.y_train,
        X_val,
        splits.y_val,
    )
    print("VAL baseline metrics:", baseline_fit.metrics)

    # 4) Train model
    print("\n[4/6] Training LightGBM model...")
    predictor = MachinePricePredictor(random_state=SEED)
    predictor.fit(X_train, splits.y_train, X_val, splits.y_val)

    # 5) Evaluate
    print("\n[5/6] Evaluating model...")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Save baseline metrics for a clean business comparison
    baseline_path = Path(OUTPUT_DIR) / "val_metrics_baseline.json"
    baseline_path.write_text(
        json.dumps(baseline_fit.metrics, indent=2),
        encoding="utf-8",
    )

    predictor.evaluate(
        X_val,
        splits.y_val,
        X_test,
        splits.y_test,
        output_dir=OUTPUT_DIR,
    )

    # Print baseline vs model improvement (easy “value add” line for the report/pitch)
    try:
        baseline_mae = float(baseline_fit.metrics["MAE"])
        model_val_mae = float(predictor.model_fit.metrics["MAE"])
        abs_gain = baseline_mae - model_val_mae
        rel_gain = abs_gain / baseline_mae if baseline_mae > 0 else 0.0
        print(
            f"\nBaseline vs Model (VAL MAE): {baseline_mae:,.0f} -> {model_val_mae:,.0f} "
            f"(improvement: {abs_gain:,.0f}, {rel_gain:.1%})"
        )
    except Exception:
        pass

    # 6) Save trained model
    print("\n[6/6] Saving model artifact...")
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    model_path = f"{MODEL_DIR}/machine_price_model.joblib"
    predictor.save(model_path)

    print("\nPipeline completed successfully.")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
