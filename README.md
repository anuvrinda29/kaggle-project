# Mumbai Weather Intelligence
### Historical Weather Analytics and Machine Learning-Based Temperature Prediction
**College Data Visualization (DV) Mini Project — 10 Marks**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50%2B-FF4B4B.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-7.1%2B-3F4F75.svg)](https://plotly.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6%2B-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Academic%20Use-green.svg)]()

---

## 📌 Project Links & Submission Placeholders

- **GitHub Repository:** `[Insert your GitHub repository URL here]`
- **Live Deployed Web Application:** `[Insert your Streamlit Community Cloud URL here]`
- **Power BI Published Dashboard:** `[Insert your published Power BI web link here]` *(See `POWER_BI_GUIDE.md`)*

---

## 📖 Executive Summary

**Mumbai Weather Intelligence** is an end-to-end meteorological data analytics and machine learning web application built for an academic Data Visualization (DV) mini project. The project analyzes **21 years (2000–2020)** of daily weather observations from the **Bombay Santacruz station (Station ID: 43003)** comprising **7,646 valid records**.

The platform features:
1. **Automated Data Quality & Preprocessing:** Continuous calendar alignment (`pd.date_range`), duplicate detection, physical bounds verification, and graceful degradation for missing variables.
2. **Interactive Climatological Visualizations:** High-contrast Dark Navy UI built with Plotly, providing multi-year time-series trends, rolling averages, climatological monthly profiles, and annual anomalies.
3. **Advanced Exploratory Analytics:** Year-by-month thermal matrices (heatmaps), seasonal dispersion box plots, bivariate correlation analysis, and Tukey IQR outlier detection.
4. **Chronological Machine Learning Engine:** Strict leak-free time-series regression comparing a **Random Forest Regressor** against a **Previous-Day Persistence Baseline** on an unseen held-out test set (2019–2020).
5. **Power BI Data Pipeline:** Automated generation of clean CSV export tables accompanied by a step-by-step report implementation guide and DAX formulas.

---

## 🏗️ Architecture Overview

![System Architecture](docs/architecture_diagram.svg)

```text
DV PROJECT/
├── .streamlit/
│   └── config.toml                  # Dark navy/charcoal styling and Streamlit UI configuration
├── data/
│   └── mumbai_temperature_preprocessed.csv  # Bundled primary historical dataset (7,646 records)
├── powerbi_exports/
│   ├── mumbai_weather_cleaned.csv   # Cleaned daily weather observations
│   ├── mumbai_monthly_summary.csv   # Aggregated monthly averages, min, max
│   ├── mumbai_yearly_summary.csv    # Annual trend summaries and year-over-year deltas
│   ├── model_test_predictions.csv   # Out-of-sample actual vs. predicted values & residuals
│   ├── model_evaluation_metrics.csv # Quantitative metrics (MAE, RMSE, R², MAPE)
│   └── mumbai_seasonal_kpi.csv      # Seasonal aggregations
├── docs/
│   └── architecture_diagram.svg     # Clean vector architecture diagram for methodology page
├── tests/
│   ├── test_data_processing.py      # Unit tests for ingestion, validation, and feature generation
│   └── test_ml_model.py             # Unit tests for time-series splitting, feature leakage, and model metrics
├── app.py                           # Main Streamlit application entry point
├── data_processing.py               # Data loading, validation, calendar alignment, preprocessing
├── visualizations.py                # Plotly interactive chart builders with dark navy theme
├── ml_model.py                      # Chronological time series ML training, baselining, and prediction
├── requirements.txt                 # Exact project dependencies
├── pytest.ini                       # Automated test suite configuration
├── DATA_SOURCE.md                   # Dataset provenance, schema dictionary, and data quality notes
├── POWER_BI_GUIDE.md                # 10-mark Power BI report guide, DAX formulas & visual layouts
└── .gitignore                       # Git ignore file
```

---

## 📊 Dataset & Schema Overview

- **Station:** Bombay Santacruz (Mumbai, India)
- **Period:** 2000-01-01 to 2020-12-27 (7,667 calendar days; 7,646 recorded; 21 calendar days missing).
- **Primary Columns:**
  - `DATE`: ISO 8601 date.
  - `TAVG`: Daily average temperature in °C (**100% complete**, range: 17.7°C to 33.7°C, mean: 27.92°C).
  - `TMAX`: Daily maximum temperature in °C (**60.8% complete**, range: 22.3°C to 41.3°C, mean: 32.54°C).
  - `TMIN`: Daily minimum temperature in °C (**75.0% complete**, range: 8.5°C to 29.7°C, mean: 22.47°C).
  - `SEASON`: Meteorological season categorized according to Indian Meteorological Department (IMD) standards (*Winter, Summer, Monsoon, Post-Monsoon*).
- **Data Source & Provenance:** Historical surface records from NOAA GHCN-Daily / IMD archives.
  > ⚠️ **Source URL Status — [FLAGGED FOR PROJECT OWNER]:** An exact direct download URL was not included in the provided project files. The project owner should insert their specific course portal or repository URL in [`DATA_SOURCE.md`](DATA_SOURCE.md) before final submission.

---

## 🤖 Machine Learning Methodology & Results

### 1. Leak-Free Protocol
- **Continuous Calendar Guarantee:** Dataset is reindexed to daily frequency (`freq='D'`). All lag features (`tavg_lag_1`, `tavg_lag_2`, `tavg_lag_7`) and rolling statistics (`tavg_roll_mean_7`, `tavg_roll_mean_30`) use strictly past observations ($t-1, t-2, \dots$).
- **No Data Leakage:** Current day $t$ observations are never present in predictive features.
- **Chronological Split:**
  - **Training Set:** 2000-01-21 to 2015-12-31 (5,783 observations)
  - **Validation Set:** 2016-01-01 to 2018-12-31 (1,096 observations)
  - **Held-Out Test Set:** 2019-01-01 to 2020-12-27 (727 observations)

### 2. Quantitative Benchmark Results (Held-Out Test Set)

| Model Name | MAE (°C) | RMSE (°C) | $R^2$ Score | MAPE (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Persistence Baseline ($t-1$)** | 0.658 | 0.883 | 0.7901 | 2.37% |
| **Ridge Linear Regression** | 0.626 | 0.836 | 0.8117 | 2.26% |
| **Random Forest Regressor** | **0.638** | **0.838** | **0.8111** | **2.29%** |

*Findings:* Daily temperature exhibits strong natural persistence ($t-1$ explains ~79% of variance). Ridge Linear Regression slightly outperforms Random Forest, and both models achieve an observed **~5.1% reduction in RMSE** over the baseline (lowering RMSE from 0.883°C to 0.836°C–0.838°C), capturing seasonal transitions and explaining **81.1% of the temperature variance** on unseen test data.

---

## 🚀 Local Installation & Execution

### Prerequisites
- Python 3.9 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/mumbai-weather-intelligence.git
cd mumbai-weather-intelligence
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Automated Unit Tests
```bash
pytest tests/ -v
```
*(All 11 tests should pass in ~2 seconds).*

### 5. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your web browser and navigate to: **`http://localhost:8501`**

---

## 📈 Power BI Integration (College Deliverable)

The project includes pre-generated export files in `powerbi_exports/`:
1. `mumbai_weather_cleaned.csv`
2. `mumbai_monthly_summary.csv`
3. `mumbai_yearly_summary.csv`
4. `mumbai_seasonal_kpi.csv`
5. `model_test_predictions.csv`
6. `model_evaluation_metrics.csv`

For full instructions on data modeling, relationship configuration, suggested chart layouts, and exact DAX measures, consult **[`POWER_BI_GUIDE.md`](POWER_BI_GUIDE.md)**.

---

## ☁️ Deployment Guide (Streamlit Community Cloud)

1. Push this repository to your GitHub account:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Mumbai Weather Intelligence Application"
   git branch -M main
   git remote add origin https://github.com/your-username/mumbai-weather-intelligence.git
   git push -u origin main
   ```
2. Log into [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New app**.
4. Select your repository, set the branch to `main`, and specify the entry point as `app.py`.
5. Click **Deploy!**
6. Once deployed, copy your public URL (e.g., `https://mumbai-weather-intelligence.streamlit.app/`) and paste it into the placeholder at the top of this `README.md`.

---

## ⚠️ Academic Disclaimer

This project was developed strictly for academic evaluation for a college Data Visualization (DV) mini project (10 marks). It provides historical retrospective analyses and statistical regressions based on surface station data. It is **not** an official meteorological forecasting service and should not be used for emergency weather warnings or maritime navigation.
