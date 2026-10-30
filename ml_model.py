from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
"""
ml_model.py
-----------
Machine Learning pipeline for historical temperature prediction.
Implements leak-free feature engineering, chronological train/val/test split,
previous-day persistence baseline, and Random Forest regression evaluation.
"""

import os
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def create_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Construct time-series regression features with strict leakage prevention.
    All rolling features and lags use past observations only (shifted by at least 1 day).
    Because the input DataFrame is reindexed to a continuous daily calendar,
    shift(k) guarantees alignment with exactly t-k calendar days.
    """
    featured = df.copy()

    # Target is TAVG at day t
    # Lags (strictly t-1, t-2, t-3, t-7 calendar days)
    featured["tavg_lag_1"] = featured["TAVG"].shift(1)
    featured["tavg_lag_2"] = featured["TAVG"].shift(2)
    featured["tavg_lag_3"] = featured["TAVG"].shift(3)
    featured["tavg_lag_7"] = featured["TAVG"].shift(7)

    # Rolling statistics: computed ONLY on past data (shifted by 1)
    past_series = featured["TAVG"].shift(1)
    featured["tavg_roll_mean_7"] = past_series.rolling(window=7, min_periods=5).mean()
    featured["tavg_roll_std_7"] = past_series.rolling(window=7, min_periods=5).std()
    featured["tavg_roll_mean_30"] = past_series.rolling(window=30, min_periods=20).mean()

    # Calendar and cyclical encodings
    featured["day_of_year"] = featured["DATE"].dt.dayofyear.astype(float)
    featured["month"] = featured["DATE"].dt.month.astype(float)
    featured["day_of_week"] = (featured["DATE"].dt.weekday + 1).astype(float)
    featured["quarter"] = featured["DATE"].dt.quarter.astype(float)

    # Cyclical sin/cos transformation of day of year
    featured["sin_doy"] = np.sin(2 * np.pi * featured["day_of_year"] / 365.25)
    featured["cos_doy"] = np.cos(2 * np.pi * featured["day_of_year"] / 365.25)

    feature_cols = [
        "tavg_lag_1",
        "tavg_lag_2",
        "tavg_lag_3",
        "tavg_lag_7",
        "tavg_roll_mean_7",
        "tavg_roll_std_7",
        "tavg_roll_mean_30",
        "month",
        "day_of_week",
        "sin_doy",
        "cos_doy"
    ]

    for col in feature_cols:
        featured[col] = featured[col].astype(np.float64)

    return featured, feature_cols


def chronological_split(
    featured_df: pd.DataFrame,
    train_end_year: int = 2015,
    val_end_year: int = 2018
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset chronologically to mimic real-world forecasting.
    No random shuffling is ever performed.
    """
    clean = featured_df.dropna(subset=["TAVG", "tavg_lag_1", "tavg_roll_mean_7", "tavg_roll_mean_30"]).copy()

    train_df = clean[clean["DATE"].dt.year <= train_end_year].copy()
    val_df = clean[(clean["DATE"].dt.year > train_end_year) & (clean["DATE"].dt.year <= val_end_year)].copy()
    test_df = clean[clean["DATE"].dt.year > val_end_year].copy()

    return train_df, val_df, test_df


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate standard regression evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {
        "MAE": round(float(mae), 3),
        "RMSE": round(float(rmse), 3),
        "R2": round(float(r2), 4),
        "MAPE": round(float(mape), 2)
    }


def train_and_evaluate_models(
    df: pd.DataFrame,
    train_end_year: int = 2015,
    val_end_year: int = 2018
) -> Dict[str, Any]:
    """
    Train Random Forest and baseline models, evaluating strictly on out-of-sample test data.
    """
    featured_df, feature_cols = create_features(df)
    train_df, val_df, test_df = chronological_split(featured_df, train_end_year, val_end_year)

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Insufficient data to perform chronological train/test split.")

    X_train = np.ascontiguousarray(train_df[feature_cols].values, dtype=np.float64)
    y_train = np.ascontiguousarray(train_df["TAVG"].values, dtype=np.float64)
    X_test = np.ascontiguousarray(test_df[feature_cols].values, dtype=np.float64)
    y_test = np.ascontiguousarray(test_df["TAVG"].values, dtype=np.float64)

    # 1. Baseline: Previous-Day Persistence (TAVG_t = TAVG_{t-1})
    baseline_pred = test_df["tavg_lag_1"].values
    baseline_metrics = compute_metrics(y_test, baseline_pred)

    # 2. Linear / Ridge Regression with StandardScaler pipeline
    ridge_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=10.0, solver="lsqr"))
    ])
    ridge_pipeline.fit(X_train, y_train)
    ridge_pred = ridge_pipeline.predict(X_test)
    ridge_metrics = compute_metrics(y_test, ridge_pred)

    # 3. Random Forest Regressor
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_metrics = compute_metrics(y_test, rf_pred)

    # Feature importances
    importances = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    # Prepare detailed test prediction DataFrame
    predictions_df = pd.DataFrame({
        "DATE": test_df["DATE"].dt.strftime("%Y-%m-%d"),
        "ACTUAL_TAVG": np.round(y_test, 2),
        "PRED_RF": np.round(rf_pred, 2),
        "PRED_BASELINE": np.round(baseline_pred, 2),
        "PRED_RIDGE": np.round(ridge_pred, 2),
        "RESIDUAL_RF": np.round(y_test - rf_pred, 2),
        "RESIDUAL_BASELINE": np.round(y_test - baseline_pred, 2),
        "ABS_ERROR_RF": np.round(np.abs(y_test - rf_pred), 2),
    })

    # Summary metrics table
    metrics_summary = pd.DataFrame([
        {"Model": "Persistence Baseline (t-1)", **baseline_metrics},
        {"Model": "Ridge Linear Regression", **ridge_metrics},
        {"Model": "Random Forest Regressor", **rf_metrics}
    ])

    return {
        "rf_model": rf,
        "feature_cols": feature_cols,
        "importances": importances,
        "test_df": test_df,
        "predictions_df": predictions_df,
        "metrics_summary": metrics_summary,
        "train_period": f"{train_df['DATE'].min().strftime('%Y-%m-%d')} to {train_df['DATE'].max().strftime('%Y-%m-%d')}",
        "val_period": f"{val_df['DATE'].min().strftime('%Y-%m-%d')} to {val_df['DATE'].max().strftime('%Y-%m-%d')}",
        "test_period": f"{test_df['DATE'].min().strftime('%Y-%m-%d')} to {test_df['DATE'].max().strftime('%Y-%m-%d')}",
        "train_count": len(train_df),
        "val_count": len(val_df),
        "test_count": len(test_df),
    }


def predict_next_day(
    rf_model: RandomForestRegressor,
    feature_cols: List[str],
    tavg_lag_1: float,
    tavg_lag_2: float,
    tavg_lag_3: float,
    tavg_lag_7: float,
    tavg_roll_mean_7: float,
    tavg_roll_std_7: float,
    tavg_roll_mean_30: float,
    target_date: pd.Timestamp
) -> float:
    """Generate next-day temperature prediction given specific past observations."""
    doy = float(target_date.dayofyear)
    input_data = {
        "tavg_lag_1": [float(tavg_lag_1)],
        "tavg_lag_2": [float(tavg_lag_2)],
        "tavg_lag_3": [float(tavg_lag_3)],
        "tavg_lag_7": [float(tavg_lag_7)],
        "tavg_roll_mean_7": [float(tavg_roll_mean_7)],
        "tavg_roll_std_7": [float(tavg_roll_std_7)],
        "tavg_roll_mean_30": [float(tavg_roll_mean_30)],
        "month": [float(target_date.month)],
        "day_of_week": [float(target_date.weekday() + 1)],
        "sin_doy": [float(np.sin(2 * np.pi * doy / 365.25))],
        "cos_doy": [float(np.cos(2 * np.pi * doy / 365.25))],
    }
    input_df = pd.DataFrame(input_data)[feature_cols]
    input_arr = np.ascontiguousarray(input_df.values, dtype=np.float64)
    pred = rf_model.predict(input_arr)[0]
    return float(round(pred, 2))


def export_ml_powerbi_files(ml_results: Dict[str, Any], output_dir: str = "powerbi_exports") -> Dict[str, str]:
    """Export ML test predictions and metrics tables for Power BI."""
    os.makedirs(output_dir, exist_ok=True)
    generated = {}

    pred_path = os.path.join(output_dir, "model_test_predictions.csv")
    ml_results["predictions_df"].to_csv(pred_path, index=False)
    generated["test_predictions"] = pred_path

    metrics_path = os.path.join(output_dir, "model_evaluation_metrics.csv")
    ml_results["metrics_summary"].to_csv(metrics_path, index=False)
    generated["evaluation_metrics"] = metrics_path

    return generated
