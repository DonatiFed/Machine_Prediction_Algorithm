# Machine Price Prediction – Bit AI 

This repository contains an end-to-end machine learning pipeline to predict the **Sales Price** of industrial machines using historical auction data.

The solution focuses on robust preprocessing, explicit handling of missing data, and a modern tree-based regression model suitable for large, sparse tabular datasets.

---

## Problem Description

The goal is to predict the **Sales Price** of a machine based on:

- Machine specifications and configuration
- Usage-related attributes
- Temporal information derived from the sale date

The dataset exhibits several real-world challenges:

- Large scale (400k+ rows)
- High-cardinality categorical features
- Many features with high missingness
- Strong right-skew in the target distribution

---

## Modeling Approach

### High-Level Strategy

- Use a gradient boosting tree model (LightGBM) optimized for tabular data
- Apply a log transformation to the target variable to stabilize variance
- Treat missing values as informative signals rather than noise
- Build a fully modular and reproducible pipeline

---

## Data Processing

### Data Cleaning (`data.py`)

The following steps are applied:

- Normalize common missing tokens (e.g. `"None or Unspecified"`)
- Remove unnamed index columns
- Convert the target column to numeric and drop invalid rows
- Parse sale dates and create:
  - `sale_year`
  - `sale_month`
  - `machine_age`
- Enforce correct numeric vs categorical data types
- Drop pure identifier columns (`Sales ID`, `Machine ID`, `Model ID`)
- Split the dataset into Train / Validation / Test sets (70% / 15% / 15%) using a fixed random seed

---

## Feature Engineering and Encoding

### Preprocessing Pipeline (`preprocess.py`)

All feature processing is handled through a single `ColumnTransformer`.

#### Numeric Features

- Median imputation
- Missingness indicators (`add_indicator=True`)

This allows the model to distinguish between:
- A genuine numeric value
- A value that was originally missing

#### Categorical Features

- Missing values filled with `"Unknown"`
- One-hot encoding
- Rare categories grouped using `min_frequency`

This approach keeps the feature space manageable while preserving predictive signal.

---

## Models

### Baseline Model

- `DummyRegressor(strategy="median")`
- Used as a sanity check and performance floor

### Final Model (`model.py`)

- LightGBM Regressor
- Trained on `log1p(Sales Price)`
- Leaf-wise tree growth strategy
- Parallel training with controlled regularization

Predictions are transformed back to the original euro scale for evaluation.

---

## Evaluation

### Metrics (`evaluate.py`)

Model performance is evaluated on both validation and test sets using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² score

### Generated Artifacts

The following outputs are automatically saved to the `outputs/` directory:

- Validation and test metrics (JSON)
- Predicted vs actual scatter plot
- Residual distribution histogram
- Top-20 feature importance plot
- Log-transformed target distribution histogram

---

## Results Summary

| Metric | Validation | Test |
|------|-----------|------|
| MAE  | ~4.8k € | ~4.8k € |
| RMSE | ~7.7k € | ~7.6k € |
| R²   | ~0.89 | ~0.89 |

The model generalizes well and shows minimal overfitting.

---

## How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```
### Required packages:

pandas

numpy

scikit-learn

lightgbm

matplotlib

### Run the Full Pipeline
```bash
python src/main.py
```
This will:

- Load and clean the data
- Build the preprocessing pipeline
- Train the LightGBM model
- Evaluate on validation and test sets
- Save all metrics and plots

### Project Structure
```
.
├── src/
│   ├── data.py        # Data loading and cleaning
│   ├── preprocess.py  # Feature engineering and encoding
│   ├── model.py       # Model training
│   ├── evaluate.py    # Metrics and plots
│   └── main.py        # End-to-end pipeline
├── outputs/           # Saved metrics and figures
├── README.md
```
### Notes and Future Improvements
- The pipeline is easily extensible to other models (XGBoost, CatBoost, linear baselines)
- Hyperparameter tuning and cross-validation could further improve performance
- Feature importance analysis could be extended using SHAP values

