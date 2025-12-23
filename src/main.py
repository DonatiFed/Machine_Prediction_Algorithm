# src/main.py
from __future__ import annotations

import warnings
from pathlib import Path

from data import load_and_prepare
from evaluate import evaluate_and_save
from model import train_lgbm_log_target
from preprocess import build_preprocessor, drop_leakage_and_text


# ---------------------------------------------------------------------
# Configuration (kept minimal on purpose for this assignment)
# ---------------------------------------------------------------------
DATA_PATH = "BIT_AI_assignment_data.csv"
OUTPUT_DIR = "outputs"
SEED = 42

# LightGBM + sklearn sometimes emits this harmless warning depending on input types.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
)


def main() -> None:
    print("=== Machine Price Prediction Pipeline ===")

    # ------------------------------------------------------------------
    # 1. Load and clean data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading and cleaning data...")
    splits, _ = load_and_prepare(DATA_PATH, seed=SEED)

    # Apply the exact same drop policy to every split (no surprises downstream)
    X_train = drop_leakage_and_text(splits.X_train)
    X_val = drop_leakage_and_text(splits.X_val)
    X_test = drop_leakage_and_text(splits.X_test)

    # ------------------------------------------------------------------
    # 2. Build preprocessing pipeline (fit on train only)
    # ------------------------------------------------------------------
    print("\n[2/5] Building preprocessing pipeline...")
    artifacts = build_preprocessor(X_train)

    # ------------------------------------------------------------------
    # 3. Train model (LightGBM on log1p(target))
    # ------------------------------------------------------------------
    print("\n[3/5] Training LightGBM model...")
    model = train_lgbm_log_target(
        artifacts.preprocessor,
        X_train,
        splits.y_train,
        X_val,
        splits.y_val,
        random_state=SEED,
    )

    # ------------------------------------------------------------------
    # 4. Evaluate model and save artifacts
    # ------------------------------------------------------------------
    print("\n[4/5] Evaluating model...")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    evaluate_and_save(
        model=model,
        preprocessor=artifacts.preprocessor,
        X_val=X_val,
        y_val=splits.y_val,
        X_test=X_test,
        y_test=splits.y_test,
        output_dir=OUTPUT_DIR,
    )

    # ------------------------------------------------------------------
    # 5. Done
    # ------------------------------------------------------------------
    print("\n[5/5] Pipeline completed successfully.")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
