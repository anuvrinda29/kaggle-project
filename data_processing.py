"""
data_processing.py
------------------
Data loading, schema normalization, calendar alignment, validation,
and feature engineering for Mumbai Weather Intelligence.
"""

from __future__ import annotations
import os
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np

# Column aliases for robust auto-detection on custom CSV uploads
COLUMN_ALIASES = {
    "DATE": ["date", "date_time", "datetime", "timestamp", "dt", "time"],
    "TAVG": ["tavg", "temperature", "temp", "avg_temp", "t_avg", "mean_temp", "temp_avg", "temperature_average"],
    "TMAX": ["tmax", "max_temp", "t_max", "maximum_temperature", "temp_max", "t_high"],
    "TMIN": ["tmin", "min_temp", "t_min", "minimum_temperature", "temp_min", "t_low"],
    "CITY": ["city", "station", "location", "station_name", "place"]
}


def get_season(month: int) -> str:
    """Return meteorological season for a given calendar month (1-12) according to IMD."""
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Post-Monsoon"


def detect_and_map_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """
    Detect column headers using case-insensitive and normalized alias matching.
    Returns:
        renamed_df: DataFrame with normalized uppercase column names
        detected_map: Dictionary mapping original column to normalized column
        warnings: List of detection warning strings
    """
    detected_map = {}
    warnings = []
    normalized_cols = {col.strip().lower().replace(" ", "_"): col for col in df.columns}

    for canonical, aliases in COLUMN_ALIASES.items():
        matched_original = None
        if canonical.lower() in normalized_cols:
            matched_original = normalized_cols[canonical.lower()]
        else:
            for alias in aliases:
                if alias in normalized_cols:
                    matched_original = normalized_cols[alias]
                    break
        
        if matched_original:
            detected_map[matched_original] = canonical

    renamed_df = df.rename(columns=detected_map)

    if "DATE" not in renamed_df.columns:
        warnings.append("No 'DATE' column detected automatically. Please map the Date column manually.")
    if "TAVG" not in renamed_df.columns:
        warnings.append("No 'TAVG' (Average Temperature) column detected. Please map it manually.")
    if "TMAX" not in renamed_df.columns:
        warnings.append("TMAX (Maximum Temperature) column was not found. TMAX analytics will be disabled.")
    if "TMIN" not in renamed_df.columns:
        warnings.append("TMIN (Minimum Temperature) column was not found. TMIN analytics will be disabled.")

    return renamed_df, detected_map, warnings


def load_weather_dataset(file_source: Any = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load dataset from default CSV path or user-uploaded file buffer.
    Returns:
        df: Cleaned and structured pandas DataFrame with explicit daily calendar
        report: Dictionary containing data quality and pipeline audit details
    """
    report = {
        "source": "Default Bundled Dataset",
        "raw_rows": 0,
        "valid_rows": 0,
        "rejected_rows": 0,
        "duplicate_dates": 0,
        "missing_calendar_dates": 0,
        "missing_values": {},
        "has_tmax": False,
        "has_tmin": False,
        "date_min": None,
        "date_max": None,
        "warnings": [],
        "station_name": "Bombay Santacruz, IN"
    }

    if file_source is None:
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "mumbai_temperature_preprocessed.csv"),
            os.path.join(os.path.dirname(__file__), "data", "mumbai_temperature_preprocessed.csv"),
            "mumbai_temperature_preprocessed.csv",
            "data/mumbai_temperature_preprocessed.csv"
        ]
        target_path = None
        for p in possible_paths:
            if os.path.exists(p):
                target_path = p
                break

        if not target_path:
            raise FileNotFoundError("Default Mumbai weather dataset 'mumbai_temperature_preprocessed.csv' could not be found.")

        raw_df = pd.read_csv(target_path)
        report["source"] = f"Bundled Dataset ({os.path.basename(target_path)})"
    elif isinstance(file_source, (str, os.PathLike)):
        raw_df = pd.read_csv(file_source)
        report["source"] = os.path.basename(file_source)
    else:
        raw_df = pd.read_csv(file_source)
        report["source"] = getattr(file_source, "name", "Uploaded CSV")

    report["raw_rows"] = len(raw_df)

    df, col_map, detection_warnings = detect_and_map_columns(raw_df)
    report["warnings"].extend(detection_warnings)

    if "DATE" not in df.columns:
        raise ValueError("The dataset does not contain an identifiable 'DATE' column.")
    if "TAVG" not in df.columns:
        raise ValueError("The dataset does not contain an identifiable 'TAVG' (Average Temperature) column.")

    # Parse DATE safely
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    invalid_dates = df["DATE"].isna()
    if invalid_dates.sum() > 0:
        report["warnings"].append(f"Dropped {invalid_dates.sum()} rows with unparseable dates.")
        df = df[~invalid_dates].copy()

    # Chronological sort
    df = df.sort_values("DATE").reset_index(drop=True)

    # Check for duplicate dates
    dup_mask = df.duplicated(subset=["DATE"], keep="first")
    report["duplicate_dates"] = int(dup_mask.sum())
    if report["duplicate_dates"] > 0:
        report["warnings"].append(f"Found {report['duplicate_dates']} duplicate dates. Retaining first occurrence.")
        df = df[~dup_mask].copy().reset_index(drop=True)

    # Numeric conversion for temperature columns
    for temp_col in ["TAVG", "TMAX", "TMIN"]:
        if temp_col in df.columns:
            df[temp_col] = pd.to_numeric(df[temp_col], errors="coerce")
            anomalous = (df[temp_col] < 0) | (df[temp_col] > 55)
            if anomalous.sum() > 0:
                report["warnings"].append(f"Flagged {anomalous.sum()} impossible temperature values (<0°C or >55°C) in {temp_col}.")
                df.loc[anomalous, temp_col] = np.nan

    report["has_tmax"] = bool("TMAX" in df.columns and df["TMAX"].notna().sum() > 0)
    report["has_tmin"] = bool("TMIN" in df.columns and df["TMIN"].notna().sum() > 0)

    valid_tavg = df["TAVG"].notna()
    rejected = int((~valid_tavg).sum())
    report["rejected_rows"] = rejected
    if rejected > 0:
        report["warnings"].append(f"Dropped {rejected} rows with missing TAVG.")
        df = df[valid_tavg].copy().reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No valid temperature observations remained after preprocessing.")

    report["valid_rows"] = len(df)
    report["date_min"] = df["DATE"].min().strftime("%Y-%m-%d")
    report["date_max"] = df["DATE"].max().strftime("%Y-%m-%d")

    # Calendar continuity analysis: construct regular daily sequence
    full_calendar = pd.date_range(start=df["DATE"].min(), end=df["DATE"].max(), freq="D")
    expected_days = len(full_calendar)
    report["missing_calendar_dates"] = int(expected_days - len(df))

    # Reindex to full daily calendar to explicitly expose missing days in the time series
    df_indexed = df.set_index("DATE").reindex(full_calendar)
    df_indexed.index.name = "DATE"
    df = df_indexed.reset_index()

    # Re-derive standard time features
    df["YEAR"] = df["DATE"].dt.year
    df["MONTH_NUMBER"] = df["DATE"].dt.month
    df["MONTH_NAME"] = df["DATE"].dt.strftime("%B")
    df["DAY"] = df["DATE"].dt.day
    df["DAY_OF_YEAR"] = df["DATE"].dt.dayofyear
    df["DAY_OF_WEEK"] = df["DATE"].dt.weekday + 1 # 1=Monday, 7=Sunday
    df["WEEKDAY_NAME"] = df["DATE"].dt.strftime("%A")
    df["QUARTER"] = df["DATE"].dt.quarter
    df["SEASON"] = df["MONTH_NUMBER"].apply(get_season)

    # City / station fallback
    if "CITY" not in df.columns or df["CITY"].isna().all():
        df["CITY"] = report["station_name"]
    else:
        df["CITY"] = df["CITY"].ffill().bfill().fillna(report["station_name"])
        report["station_name"] = str(df["CITY"].iloc[0])

    # Diurnal Temperature Range (DTR) = TMAX - TMIN where both exist
    if report["has_tmax"] and report["has_tmin"]:
        df["TEMP_RANGE"] = df["TMAX"] - df["TMIN"]
        invalid_range = df["TEMP_RANGE"] < 0
        if invalid_range.sum() > 0:
            df.loc[invalid_range, ["TMAX", "TMIN", "TEMP_RANGE"]] = np.nan
            report["warnings"].append(f"Invalidated {invalid_range.sum()} records where TMIN exceeded TMAX.")
    else:
        df["TEMP_RANGE"] = np.nan

    # Missing value statistics
    for col in ["TAVG", "TMAX", "TMIN", "TEMP_RANGE"]:
        if col in df.columns:
            missing_cnt = int(df[col].isna().sum())
            report["missing_values"][col] = {
                "missing_count": missing_cnt,
                "total_rows": len(df),
                "completeness_pct": round((1 - missing_cnt / len(df)) * 100, 2)
            }

    return df, report


def compute_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 12-month climatological summary table across all observed years."""
    observed = df.dropna(subset=["TAVG"])
    monthly = observed.groupby(["MONTH_NUMBER", "MONTH_NAME"]).agg(
        avg_temp=("TAVG", "mean"),
        min_temp=("TAVG", "min"),
        max_temp=("TAVG", "max"),
        std_temp=("TAVG", "std"),
        observation_count=("TAVG", "count")
    ).reset_index()

    if "TMAX" in observed.columns and observed["TMAX"].notna().sum() > 0:
        tmax_summary = observed.groupby("MONTH_NUMBER")["TMAX"].agg(mean_tmax="mean", abs_max="max")
        monthly = monthly.merge(tmax_summary, on="MONTH_NUMBER", how="left")

    if "TMIN" in observed.columns and observed["TMIN"].notna().sum() > 0:
        tmin_summary = observed.groupby("MONTH_NUMBER")["TMIN"].agg(mean_tmin="mean", abs_min="min")
        monthly = monthly.merge(tmin_summary, on="MONTH_NUMBER", how="left")

    return monthly.sort_values("MONTH_NUMBER").reset_index(drop=True)


def compute_yearly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute annual historical summary, temperature anomaly, and year-over-year deltas."""
    observed = df.dropna(subset=["TAVG"])
    yearly = observed.groupby("YEAR").agg(
        avg_temp=("TAVG", "mean"),
        min_temp=("TAVG", "min"),
        max_temp=("TAVG", "max"),
        std_temp=("TAVG", "std"),
        days_recorded=("TAVG", "count")
    ).reset_index()

    yearly["yoy_change"] = yearly["avg_temp"].diff()
    overall_mean = observed["TAVG"].mean()
    yearly["temp_anomaly"] = yearly["avg_temp"] - overall_mean

    return yearly.sort_values("YEAR").reset_index(drop=True)


def compute_seasonal_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate seasonal metrics according to Indian Meteorological definitions."""
    observed = df.dropna(subset=["TAVG"])
    season_order = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]
    seasonal = observed.groupby("SEASON").agg(
        avg_temp=("TAVG", "mean"),
        min_temp=("TAVG", "min"),
        max_temp=("TAVG", "max"),
        std_temp=("TAVG", "std"),
        total_observations=("TAVG", "count")
    ).reset_index()

    seasonal["SEASON"] = pd.Categorical(seasonal["SEASON"], categories=season_order, ordered=True)
    return seasonal.sort_values("SEASON").reset_index(drop=True)


def detect_outliers(df: pd.DataFrame, col: str = "TAVG", factor: float = 1.5) -> pd.DataFrame:
    """
    Detect statistical outliers using Tukey's Interquartile Range (IQR) rule:
    Lower bound = Q1 - factor * IQR
    Upper bound = Q3 + factor * IQR
    """
    valid = df.dropna(subset=[col]).copy()
    q1 = valid[col].quantile(0.25)
    q3 = valid[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr

    outliers = valid[(valid[col] < lower_bound) | (valid[col] > upper_bound)].copy()
    outliers["outlier_type"] = np.where(outliers[col] > upper_bound, "High Outlier (Heatwave)", "Low Outlier (Cold Snap)")
    outliers["bound_reference"] = np.where(outliers[col] > upper_bound, upper_bound, lower_bound)
    return outliers.sort_values("DATE").reset_index(drop=True)


def generate_powerbi_exports(df: pd.DataFrame, output_dir: str = "powerbi_exports") -> Dict[str, str]:
    """Export clean, structured CSV files formatted specifically for Power BI Desktop ingestion."""
    os.makedirs(output_dir, exist_ok=True)
    generated = {}

    clean_daily_path = os.path.join(output_dir, "mumbai_weather_cleaned.csv")
    df.to_csv(clean_daily_path, index=False)
    generated["cleaned_daily"] = clean_daily_path

    monthly_path = os.path.join(output_dir, "mumbai_monthly_summary.csv")
    monthly_df = compute_monthly_summary(df)
    monthly_df.to_csv(monthly_path, index=False)
    generated["monthly_summary"] = monthly_path

    yearly_path = os.path.join(output_dir, "mumbai_yearly_summary.csv")
    yearly_df = compute_yearly_summary(df)
    yearly_df.to_csv(yearly_path, index=False)
    generated["yearly_summary"] = yearly_path

    seasonal_path = os.path.join(output_dir, "mumbai_seasonal_kpi.csv")
    seasonal_df = compute_seasonal_summary(df)
    seasonal_df.to_csv(seasonal_path, index=False)
    generated["seasonal_kpi"] = seasonal_path

    return generated
