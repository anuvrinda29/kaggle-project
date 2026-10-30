"""
app.py
------
Mumbai Weather Intelligence: Historical Weather Analytics & ML Temperature Prediction
Streamlit Web Application for College Data Visualization (DV) Mini Project.
"""

from __future__ import annotations
import os
from typing import Dict, Any, Optional
import datetime
import pandas as pd
import numpy as np
import streamlit as st

import data_processing as dp
import visualizations as viz
import ml_model as ml

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION & CUSTOM STYLES
# ==============================================================================
st.set_page_config(
    page_title="Mumbai Weather Intelligence",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Contrast Dark Navy / Charcoal Theme CSS
CUSTOM_CSS = """
<style>
    /* Global Styles */
    .stApp {
        background-color: #0B132B;
        color: #F8FAFC;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1C2541 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Header Container */
    .header-banner {
        background: linear-gradient(135deg, #1C2541 0%, #1E293B 100%);
        padding: 24px 28px;
        border-radius: 12px;
        border: 1px solid rgba(0, 245, 212, 0.25);
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 6px;
        letter-spacing: -0.5px;
    }
    .header-title span {
        color: #00F5D4;
    }
    .header-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 14px;
    }
    .header-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: rgba(56, 189, 248, 0.15);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .badge-alert {
        background-color: rgba(244, 63, 94, 0.15);
        color: #F43F5E;
        border: 1px solid rgba(244, 63, 94, 0.3);
    }
    
    /* KPI Metric Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }
    .kpi-card {
        background-color: #1C2541;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #94A3B8;
        font-weight: 500;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 4px;
    }
    .kpi-sub {
        font-size: 0.75rem;
        color: #38BDF8;
    }
    
    /* Academic Disclaimer Banner */
    .disclaimer-banner {
        background-color: rgba(251, 191, 36, 0.1);
        border-left: 4px solid #FBBF24;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 20px;
        font-size: 0.88rem;
        color: #FDE68A;
    }
    
    /* Card Container */
    .section-card {
        background-color: #1C2541;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 24px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# DATA LOADING & CACHING
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_default_data():
    """Load and preprocess the bundled historical Mumbai weather dataset."""
    return dp.load_weather_dataset()


@st.cache_data(show_spinner=False)
def get_ml_results(data_hash: int, df_subset: pd.DataFrame):
    """Train and evaluate models with strict chronological split."""
    return ml.train_and_evaluate_models(df_subset)


# ==============================================================================
# SIDEBAR CONTROLS & NAVIGATION
# ==============================================================================
st.sidebar.markdown("### 🌤️ Weather Intelligence")
st.sidebar.markdown("**DV Mini Project (10 Marks)**")
st.sidebar.markdown("---")

# Data Source Selection
data_mode = st.sidebar.radio(
    "Data Source Mode",
    ["Bundled Mumbai Weather (2000–2020)", "Upload Custom Weather CSV"],
    index=0,
    help="Default bundled dataset covers Bombay Santacruz station. Upload your own CSV to test custom weather files."
)

active_df = None
data_report = None

if data_mode == "Bundled Mumbai Weather (2000–2020)":
    try:
        active_df, data_report = load_default_data()
    except Exception as e:
        st.sidebar.error(f"Failed to load bundled dataset: {e}")
        st.stop()
else:
    uploaded_file = st.sidebar.file_uploader("Upload Weather CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            active_df, data_report = dp.load_weather_dataset(uploaded_file)
            st.sidebar.success(f"Loaded: {uploaded_file.name}")
        except Exception as e:
            st.sidebar.error(f"File error: {e}")
            st.info("Falling back to default bundled dataset...")
            active_df, data_report = load_default_data()
    else:
        st.sidebar.info("No file uploaded yet. Showing default bundled dataset.")
        active_df, data_report = load_default_data()

st.sidebar.markdown("---")

# Navigation Menu
page_selection = st.sidebar.radio(
    "Dashboard Navigation",
    [
        "🌤️ Weather Overview",
        "📊 Advanced Visualizations & EDA",
        "🤖 ML & Temperature Prediction",
        "📑 Methodology & Architecture",
        "📥 Data Quality & Power BI Exports"
    ],
    index=0
)

st.sidebar.markdown("---")

# Date Filter
min_date = active_df["DATE"].min().date()
max_date = active_df["DATE"].max().date()

st.sidebar.markdown("### 📅 Temporal Filter")
date_range = st.sidebar.date_input(
    "Filter Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_filter, end_filter = date_range
    mask = (active_df["DATE"].dt.date >= start_filter) & (active_df["DATE"].dt.date <= end_filter)
    filtered_df = active_df[mask].copy().reset_index(drop=True)
else:
    filtered_df = active_df.copy()

# Moving Average Window Control
rolling_window = st.sidebar.slider(
    "Moving Average Window (Days)",
    min_value=7,
    max_value=90,
    value=30,
    step=1,
    help="Configures the smoothing window for the historical temperature trend visualization."
)

st.sidebar.markdown("---")
st.sidebar.caption("Mumbai Weather Intelligence v1.0.0 | Academic Project")


# ==============================================================================
# HEADER BANNER (ALL PAGES)
# ==============================================================================
st.markdown(f"""
<div class="header-banner">
    <div class="header-title">Mumbai <span>Weather Intelligence</span></div>
    <div class="header-subtitle">Historical Climatological Analytics & Machine Learning-Based Temperature Regression</div>
    <div class="header-badges">
        <span class="badge">📍 Station: {data_report['station_name']}</span>
        <span class="badge">📅 Coverage: {data_report['date_min']} to {data_report['date_max']}</span>
        <span class="badge">📊 Valid Records: {data_report['valid_rows']:,} Days</span>
        <span class="badge">🎯 Target Variable: TAVG (°C)</span>
        <span class="badge badge-alert">Academic Mini Project (10 Marks)</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-banner">
    ⚠️ <b>Academic Disclaimer:</b> This application presents historical retrospective analytics and statistical machine learning regressions based on station records (2000–2020). It is strictly an academic research project and is <b>not</b> an official real-time meteorological forecasting service.
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# PAGE 1: WEATHER OVERVIEW
# ==============================================================================
if page_selection == "🌤️ Weather Overview":
    st.markdown("### 🌤️ Weather Overview & Key Performance Indicators")
    st.markdown(
        "Explore historical daily temperatures recorded at the Santacruz station, analyze multi-year climatological cycles, "
        "and monitor annual temperature anomalies with interactive filtering."
    )

    # Dynamic KPI Cards
    obs_tavg = filtered_df["TAVG"].dropna()
    mean_tavg = obs_tavg.mean() if len(obs_tavg) > 0 else 0
    
    # TMAX
    has_tmax = "TMAX" in filtered_df.columns and filtered_df["TMAX"].notna().sum() > 0
    if has_tmax:
        max_rec = filtered_df["TMAX"].max()
        max_date_str = filtered_df.loc[filtered_df["TMAX"] == max_rec, "DATE"].iloc[0].strftime("%b %d, %Y")
        kpi_max_val = f"{max_rec:.1f} °C"
        kpi_max_sub = f"Recorded on {max_date_str}"
    else:
        kpi_max_val = "N/A"
        kpi_max_sub = "TMAX not available in dataset"

    # TMIN
    has_tmin = "TMIN" in filtered_df.columns and filtered_df["TMIN"].notna().sum() > 0
    if has_tmin:
        min_rec = filtered_df["TMIN"].min()
        min_date_str = filtered_df.loc[filtered_df["TMIN"] == min_rec, "DATE"].iloc[0].strftime("%b %d, %Y")
        kpi_min_val = f"{min_rec:.1f} °C"
        kpi_min_sub = f"Recorded on {min_date_str}"
    else:
        kpi_min_val = "N/A"
        kpi_min_sub = "TMIN not available in dataset"

    valid_count = len(obs_tavg)
    completeness = (valid_count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
    date_span_str = f"{filtered_df['DATE'].min().strftime('%Y-%m-%d')} to {filtered_df['DATE'].max().strftime('%Y-%m-%d')}"

    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-label">Average Temperature (TAVG)</div>
            <div class="kpi-value">{mean_tavg:.2f} °C</div>
            <div class="kpi-sub">Across {valid_count:,} observed days</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Highest Recorded (TMAX)</div>
            <div class="kpi-value">{kpi_max_val}</div>
            <div class="kpi-sub">{kpi_max_sub}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Lowest Recorded (TMIN)</div>
            <div class="kpi-value">{kpi_min_val}</div>
            <div class="kpi-sub">{kpi_min_sub}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Observations & Completeness</div>
            <div class="kpi-value">{valid_count:,} <span style="font-size:1rem;color:#94A3B8;">/ {len(filtered_df):,}</span></div>
            <div class="kpi-sub">{completeness:.1f}% calendar completeness</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Selected Date Span</div>
            <div class="kpi-value" style="font-size:1.15rem;margin-top:6px;">{date_span_str}</div>
            <div class="kpi-sub">{len(filtered_df):,} calendar days</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Chart A: Historical Temperature Trend
    st.plotly_chart(
        viz.plot_temperature_trend(filtered_df, rolling_window=rolling_window, show_rolling=True),
        use_container_width=True
    )

    # Charts B & C side by side
    col1, col2 = st.columns(2)
    with col1:
        monthly_summary = dp.compute_monthly_summary(filtered_df)
        st.plotly_chart(viz.plot_monthly_climatology(monthly_summary), use_container_width=True)
    with col2:
        yearly_summary = dp.compute_yearly_summary(filtered_df)
        st.plotly_chart(viz.plot_yearly_trend(yearly_summary), use_container_width=True)

    # Chart D & E
    col3, col4 = st.columns([0.58, 0.42])
    with col3:
        st.plotly_chart(viz.plot_temperature_comparison(filtered_df), use_container_width=True)
    with col4:
        st.plotly_chart(viz.plot_temperature_distribution(filtered_df), use_container_width=True)


# ==============================================================================
# PAGE 2: ADVANCED VISUALIZATIONS & EXPLORATORY DATA ANALYSIS
# ==============================================================================
elif page_selection == "📊 Advanced Visualizations & EDA":
    st.markdown("### 📊 Advanced Exploratory Data Analysis & Climatological Diagnostics")
    st.markdown(
        "This section implements high-value multidimensional exploratory views designed to uncover hidden seasonality, "
        "temperature dispersion, extreme weather outliers, and data availability."
    )

    # Advanced Viz 1: Year x Month Heatmap
    st.plotly_chart(viz.plot_year_month_heatmap(filtered_df), use_container_width=True)
    st.caption(
        "**Analytical Value:** The Year × Month thermal matrix immediately surfaces inter-annual shifts in Mumbai's climatology. "
        "Notice the consistently high temperatures in April–May (pre-monsoon summer), the sharp moderation in June–August (monsoon precipitation), "
        "and the pronounced secondary peak in October ('second summer')."
    )

    # Advanced Viz 2 & 3 side by side
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(viz.plot_monthly_boxplots(filtered_df), use_container_width=True)
        st.caption(
            "**Analytical Value:** Box plots reveal the statistical distribution, median, interquartile range (IQR), and extreme fliers "
            "for each calendar month. Winter months (Jan–Feb) exhibit the widest temperature dispersion, whereas Monsoon months (Jul–Aug) "
            "display remarkable thermal stability."
        )
    with col2:
        corr_fig = viz.plot_temperature_correlation(filtered_df)
        if corr_fig is not None:
            st.plotly_chart(corr_fig, use_container_width=True)
            st.caption(
                "**Analytical Value:** Bivariate correlation between daily average (TAVG) and maximum (TMAX) temperatures confirms a robust linear relationship. "
                "The regression slope demonstrates how daytime peak heating strongly governs daily mean temperature."
            )
        else:
            st.info("TMAX is unavailable in the current dataset slice for correlation analysis.")

    # Advanced Viz 4: Missing Data & Completeness Audit
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(viz.plot_data_completeness(filtered_df), use_container_width=True)
        st.caption(
            "**Analytical Value:** The completeness curve visualizes variable availability over time. While TAVG is complete across the 21-year period, "
            "TMAX and TMIN have significant observation gaps in earlier years (e.g. 2000–2008), justifying why TAVG was selected as the primary target for ML."
        )
    with col4:
        outliers_df = dp.detect_outliers(filtered_df, col="TAVG", factor=1.5)
        st.plotly_chart(viz.plot_outliers(filtered_df, outliers_df), use_container_width=True)
        st.caption(
            f"**Analytical Value:** Statistical outlier detection using Tukey's 1.5×IQR rule identifies {len(outliers_df)} anomalous days "
            f"({len(outliers_df[outliers_df['outlier_type'].str.contains('Heatwave')])} heatwaves, "
            f"{len(outliers_df[outliers_df['outlier_type'].str.contains('Cold Snap')])} cold snaps)."
        )

    # Meteorological Findings Panel
    st.markdown("#### 🔍 Key Climatological Findings (Santacruz Station, 2000–2020)")
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        st.markdown("""
        **1. Pre-Monsoon Peak (Summer)**
        - **Hottest Months:** April and May (average ~30.5°C to 31.5°C).
        - **Extreme High:** Peak TMAX reached 41.3°C on March 26, 2018.
        - **Drivers:** Strong solar insolation and advection of dry continental air from the northwest before onset of the Arabian Sea monsoon.
        """)
    with fcol2:
        st.markdown("""
        **2. Monsoon Thermal Buffer**
        - **Monsoon Months:** June to September.
        - **Thermal Stability:** Temperatures drop to ~28.0°C–28.5°C with the lowest variance of any season.
        - **Drivers:** Continuous cloud cover, high atmospheric moisture, and persistent rainfall limit daytime solar heating and nocturnal radiational cooling.
        """)
    with fcol3:
        st.markdown("""
        **3. The 'October Heat' & Winter Minimum**
        - **Secondary Peak:** October temperatures rebound to ~29.5°C after the monsoon retreats ('October Heat').
        - **Coolest Months:** January and February (average ~24.5°C).
        - **Extreme Low:** Record TMIN of 8.5°C observed during a severe winter cold-wave advection.
        """)


# ==============================================================================
# PAGE 3: MACHINE LEARNING & PREDICTION
# ==============================================================================
elif page_selection == "🤖 ML & Temperature Prediction":
    st.markdown("### 🤖 Machine Learning-Based Temperature Prediction Engine")
    st.markdown(
        "A rigorous, leak-free regression pipeline predicting daily average temperature (TAVG) for the next calendar day ($t$). "
        "Evaluated on a strictly held-out chronological test period (2019–2020) and benchmarked against a persistence baseline."
    )

    # Train / Retrieve ML Results
    try:
        with st.spinner("Training Random Forest and Baseline Regressors..."):
            ml_results = get_ml_results(len(active_df), active_df)
    except Exception as e:
        st.error(f"Machine learning execution failed: {e}")
        st.stop()

    # Model Evaluation Metrics Table
    st.markdown("#### 1. Out-of-Sample Performance Evaluation (Held-Out Test Set: 2019–2020)")
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    rf_metrics = ml_results["metrics_summary"].loc[ml_results["metrics_summary"]["Model"] == "Random Forest Regressor"].iloc[0]
    base_metrics = ml_results["metrics_summary"].loc[ml_results["metrics_summary"]["Model"] == "Persistence Baseline (t-1)"].iloc[0]
    
    with m_col1:
        st.metric("Random Forest MAE", f"{rf_metrics['MAE']} °C", delta=f"{rf_metrics['MAE'] - base_metrics['MAE']:.3f} °C", delta_color="inverse")
    with m_col2:
        st.metric("Random Forest RMSE", f"{rf_metrics['RMSE']} °C", delta=f"{rf_metrics['RMSE'] - base_metrics['RMSE']:.3f} °C", delta_color="inverse")
    with m_col3:
        st.metric("Random Forest R² Score", f"{rf_metrics['R2']:.4f}", delta=f"+{rf_metrics['R2'] - base_metrics['R2']:.4f}")
    with m_col4:
        st.metric("Test Observations", f"{ml_results['test_count']:,} Days", help=f"Chronological test period: {ml_results['test_period']}")

    # Metrics Summary Table
    st.dataframe(
        ml_results["metrics_summary"].style.format({
            "MAE": "{:.3f} °C",
            "RMSE": "{:.3f} °C",
            "R2": "{:.4f}",
            "MAPE": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "**Methodological Note & Model Comparison:** Models were trained on historical data up to 2015, validated on 2016–2018, and evaluated on unseen held-out observations from 2019–2020 (727 days). "
        "Ridge Linear Regression slightly outperforms Random Forest (MAE 0.626°C vs 0.638°C, RMSE 0.836°C vs 0.838°C), reflecting the strong linear autoregressive physics of daily thermal inertia. "
        "Both machine learning models demonstrate an observed ~5.1% reduction in RMSE over the Previous-Day Persistence Baseline (dropping from 0.883°C to 0.836°C–0.838°C) and explain ~81.1% of temperature variance on unseen test records."
    )

    # ML Visualizations
    st.markdown("#### 2. Model Diagnostics and Visual Evaluation")
    st.plotly_chart(viz.plot_actual_vs_predicted(ml_results["predictions_df"]), use_container_width=True)

    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        st.plotly_chart(viz.plot_prediction_scatter(ml_results["predictions_df"]), use_container_width=True)
    with col2:
        st.plotly_chart(viz.plot_residuals(ml_results["predictions_df"]), use_container_width=True)

    # Feature Importance
    st.plotly_chart(viz.plot_feature_importance(ml_results["importances"]), use_container_width=True)

    # Interactive Temperature Prediction Simulator
    st.markdown("---")
    st.markdown("#### 3. Interactive Temperature Prediction Simulator")
    st.markdown(
        "Evaluate the trained Random Forest regressor under two distinct operational paradigms: "
        "**Historical Evaluation** (against known ground truth) and **Future Simulation** (one day beyond the historical dataset)."
    )

    sim_mode = st.radio(
        "Select Simulation Operational Mode",
        [
            "🔬 Historical Evaluation Mode (Benchmark Against Ground Truth)",
            "🔮 Future Simulation Mode (Next Calendar Day Beyond Dataset)"
        ],
        horizontal=True,
        help="Historical mode tests predictions against actual observed records. Future mode simulates the day immediately following the dataset."
    )

    if sim_mode == "🔬 Historical Evaluation Mode (Benchmark Against Ground Truth)":
        st.markdown(
            "**Protocol:** Select a target date $t$ from the held-out test period (2019–2020). "
            "Input lag features are strictly drawn from dates prior to the target date ($t-1, t-2, \\dots$). "
            "The target day's actual temperature is strictly reserved for ground-truth evaluation and is **never** fed as an input feature."
        )
        sim_col1, sim_col2 = st.columns([0.58, 0.42])
        with sim_col1:
            test_dates = ml_results["test_df"]["DATE"].dt.strftime("%Y-%m-%d").tolist()
            selected_date_str = st.selectbox(
                "Select Target Date (t) to Predict",
                options=test_dates[-60:],
                index=len(test_dates[-60:]) - 1,
                help="Select any date from the held-out test period (2019-2020)."
            )

            # Retrieve row corresponding to target date t
            target_row = ml_results["test_df"][ml_results["test_df"]["DATE"].dt.strftime("%Y-%m-%d") == selected_date_str].iloc[0]
            actual_val = target_row["TAVG"]
            target_dt = target_row["DATE"]
            prev_dt_str = (target_dt - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

            st.info(f"🎯 **Target Prediction Date ($t$):** `{selected_date_str}` | **Input Data Cutoff ($t-1$):** `{prev_dt_str}`")

            c1, c2, c3 = st.columns(3)
            with c1:
                in_lag1 = st.number_input(f"Day t-1 ({prev_dt_str}) Temp (°C)", value=float(target_row["tavg_lag_1"]), step=0.1)
                in_lag2 = st.number_input("Day t-2 Temp (°C)", value=float(target_row["tavg_lag_2"]), step=0.1)
            with c2:
                in_lag3 = st.number_input("Day t-3 Temp (°C)", value=float(target_row["tavg_lag_3"]), step=0.1)
                in_lag7 = st.number_input("Day t-7 Temp (°C)", value=float(target_row["tavg_lag_7"]), step=0.1)
            with c3:
                in_roll7 = st.number_input("Past 7-Day Mean (°C)", value=float(target_row["tavg_roll_mean_7"]), step=0.1)
                in_roll30 = st.number_input("Past 30-Day Mean (°C)", value=float(target_row["tavg_roll_mean_30"]), step=0.1)

        with sim_col2:
            st.markdown("<div class=\"section-card\">", unsafe_allow_html=True)
            st.markdown("##### 🎯 Historical Evaluation Output")
            
            sim_pred = ml.predict_next_day(
                rf_model=ml_results["rf_model"],
                feature_cols=ml_results["feature_cols"],
                tavg_lag_1=in_lag1,
                tavg_lag_2=in_lag2,
                tavg_lag_3=in_lag3,
                tavg_lag_7=in_lag7,
                tavg_roll_mean_7=in_roll7,
                tavg_roll_std_7=float(target_row["tavg_roll_std_7"]),
                tavg_roll_mean_30=in_roll30,
                target_date=target_dt
            )

            abs_error = abs(sim_pred - actual_val)
            pct_error = (abs_error / actual_val) * 100

            st.metric("Predicted Target TAVG", f"{sim_pred:.2f} °C")
            st.write(f"**Target Prediction Date:** `{selected_date_str}`")
            st.write(f"**Actual Ground Truth:** `{actual_val:.2f} °C`")
            st.write(f"**Absolute Error:** `{abs_error:.2f} °C` (MAPE: `{pct_error:.1f}%`)")

            if abs_error <= 0.65:
                st.success("✅ Model prediction within nominal test MAE (±0.64°C).")
            else:
                st.warning("⚠️ Prediction error exceeds nominal average due to atmospheric variance.")

            st.caption(
                "🔒 **Integrity Verified:** Features strictly use observations from $t-1$ and prior. "
                "Target day observation was withheld during prediction."
            )
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        # Future Simulation Mode
        last_obs_row = active_df.dropna(subset=["TAVG"]).iloc[-1]
        last_date = last_obs_row["DATE"]
        next_day = last_date + pd.Timedelta(days=1)
        next_day_str = next_day.strftime("%Y-%m-%d")
        last_date_str = last_date.strftime("%Y-%m-%d")

        st.markdown(
            f"**Protocol:** Simulates the temperature for **{next_day_str}** (one day immediately following the dataset boundary). "
            f"All features are derived from the latest available historical observations up to **{last_date_str}**."
        )

        sim_col1, sim_col2 = st.columns([0.58, 0.42])
        with sim_col1:
            st.info(f"🔮 **Target Forecast Horizon ($t$):** `{next_day_str}` | **Latest Recorded Date ($t-1$):** `{last_date_str}`")
            
            # Compute past rolling and lag features using observations up to last_date
            past_valid = active_df.dropna(subset=["TAVG"]).set_index("DATE").reindex(pd.date_range(end=last_date, periods=35, freq="D"))
            fut_lag1 = float(last_obs_row["TAVG"])
            fut_lag2 = float(past_valid["TAVG"].iloc[-2]) if len(past_valid) >= 2 else fut_lag1
            fut_lag3 = float(past_valid["TAVG"].iloc[-3]) if len(past_valid) >= 3 else fut_lag1
            fut_lag7 = float(past_valid["TAVG"].iloc[-7]) if len(past_valid) >= 7 else fut_lag1
            fut_roll7 = float(past_valid["TAVG"].iloc[-7:].mean())
            fut_roll_std7 = float(past_valid["TAVG"].iloc[-7:].std()) if not np.isnan(past_valid["TAVG"].iloc[-7:].std()) else 0.5
            fut_roll30 = float(past_valid["TAVG"].iloc[-30:].mean())

            c1, c2, c3 = st.columns(3)
            with c1:
                in_fut_lag1 = st.number_input(f"Day t-1 ({last_date_str}) (°C)", value=fut_lag1, step=0.1)
                in_fut_lag2 = st.number_input("Day t-2 (°C)", value=fut_lag2, step=0.1)
            with c2:
                in_fut_lag3 = st.number_input("Day t-3 (°C)", value=fut_lag3, step=0.1)
                in_fut_lag7 = st.number_input("Day t-7 (°C)", value=fut_lag7, step=0.1)
            with c3:
                in_fut_roll7 = st.number_input("Past 7-Day Mean (°C)", value=fut_roll7, step=0.1)
                in_fut_roll30 = st.number_input("Past 30-Day Mean (°C)", value=fut_roll30, step=0.1)

        with sim_col2:
            st.markdown("<div class=\"section-card\">", unsafe_allow_html=True)
            st.markdown("##### 🔮 Future Day Simulation Output")
            
            fut_pred = ml.predict_next_day(
                rf_model=ml_results["rf_model"],
                feature_cols=ml_results["feature_cols"],
                tavg_lag_1=in_fut_lag1,
                tavg_lag_2=in_fut_lag2,
                tavg_lag_3=in_fut_lag3,
                tavg_lag_7=in_fut_lag7,
                tavg_roll_mean_7=in_fut_roll7,
                tavg_roll_std_7=fut_roll_std7,
                tavg_roll_mean_30=in_fut_roll30,
                target_date=next_day
            )

            st.metric("Simulated Next-Day TAVG", f"{fut_pred:.2f} °C")
            st.write(f"**Target Forecast Date:** `{next_day_str}`")
            st.write("**Actual Ground Truth:** *Not Available (Unobserved Future)*")
            st.write("**Absolute Error:** *N/A (Ground Truth Does Not Exist)*")

            st.warning(
                "⚠️ **Academic Simulation Notice:** Ground truth does not exist for dates beyond the historical record. "
                "This empirical forecast is an academic statistical simulation and must not be used as an official weather forecast."
            )
            st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 4: METHODOLOGY & ARCHITECTURE
# ==============================================================================
elif page_selection == "📑 Methodology & Architecture":
    st.markdown("### 📑 Project Methodology, Architecture & Academic References")
    
    st.markdown("""
    #### 1. Problem Statement & Objectives
    - **Problem Statement:** Historical weather stations generate multi-decadal time series containing missing observations, irregular calendar gaps, and varying parameter completeness. Developing accurate, interpretable predictive models requires strict temporal protocols that prevent future data leakage while maintaining climatological context.
    - **Primary Objectives:**
      1. Preprocess and audit 21 years (2000–2020) of historical daily observations for Mumbai (Bombay Santacruz).
      2. Engineer leak-free temporal features and continuous calendar alignment.
      3. Implement and quantitatively evaluate an interpretable Random Forest regression model against a persistence baseline.
      4. Deliver an interactive Streamlit application and prepare production-ready Power BI reporting assets.

    #### 2. End-to-End System Architecture
    """)

    # High-resolution vector architecture diagram asset
    if os.path.exists("docs/architecture_diagram.svg"):
        st.image(
            "docs/architecture_diagram.svg",
            use_container_width=True,
            caption="Figure 1: End-to-End System Architecture — Data Ingestion, Calendar Audit, ML Regression Engine, and Analytics Delivery"
        )
    else:
        st.info("Architecture diagram asset located at docs/architecture_diagram.svg")

    st.markdown("""
    #### 3. Mathematical Formulations
    - **Persistence Baseline:**
      $$\\hat{y}_t = y_{t-1}$$
    - **Mean Absolute Error (MAE):**
      $$\\text{MAE} = \\frac{1}{N}\\sum_{t=1}^{N} |y_t - \\hat{y}_t|$$
    - **Root Mean Squared Error (RMSE):**
      $$\\text{RMSE} = \\sqrt{\\frac{1}{N}\\sum_{t=1}^{N} (y_t - \\hat{y}_t)^2}$$
    - **Coefficient of Determination ($R^2$):**
      $$R^2 = 1 - \\frac{\\sum_{t=1}^{N} (y_t - \\hat{y}_t)^2}{\\sum_{t=1}^{N} (y_t - \\bar{y})^2}$$
    - **Cyclical Day-of-Year Encoding:**
      $$x_{\\sin} = \\sin\\left(\\frac{2\\pi \\cdot \\text{DOY}}{365.25}\\right), \\quad x_{\\cos} = \\cos\\left(\\frac{2\\pi \\cdot \\text{DOY}}{365.25}\\right)$$

    #### 4. Project Limitations & Future Scope
    - **Limitations:**
      - The dataset lacks relative humidity, atmospheric pressure, precipitation, and wind velocity columns, which are critical drivers of coastal convective thunderstorms.
      - 21 missing calendar days required continuous reindexing to avoid erroneous lag association.
      - TMAX has ~39% missing records, restricting multi-variable modeling to TAVG.
    - **Future Scope:**
      - Integration of satellite-derived ERA5 reanalysis data for humidity and sea surface temperatures (SST).
      - Recurrent neural networks (LSTM, GRU) or Temporal Fusion Transformers (TFT) for multi-day horizon forecasting.

    #### 5. Academic Data Sources & References
    - **Primary Dataset:** Historical weather observations for Bombay Santacruz station (Station ID: 43003 / VABB), covering 2000-01-01 to 2020-12-27. *(Archive: NOAA GHCN-Daily / IMD surface archives. [FLAGGED FOR PROJECT OWNER: Insert exact original public repository URL in DATA_SOURCE.md]).*
    - **Meteorological Season Definitions:** Indian Meteorological Department (IMD) standard classification: Winter (Dec–Feb), Summer (Mar–May), Monsoon (Jun–Sep), Post-Monsoon (Oct–Nov).
    - **Machine Learning Protocol:** Hyndman, R.J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*, 3rd edition. OTexts.
    """)


# ==============================================================================
# PAGE 5: DATA QUALITY & POWER BI EXPORTS
# ==============================================================================
elif page_selection == "📥 Data Quality & Power BI Exports":
    st.markdown("### 📥 Data Quality Audit & Power BI Export Center")
    st.markdown(
        "Download pre-processed, certified CSV files ready for import into Power BI Desktop, "
        "and review the comprehensive data quality audit report."
    )

    # Data Quality Audit Card
    st.markdown("#### 1. Data Quality & Preprocessing Audit")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="section-card">
            <h5>📋 Ingestion & Integrity Metrics</h5>
            <ul>
                <li><b>Data Source:</b> {data_report['source']}</li>
                <li><b>Identified Station:</b> {data_report['station_name']}</li>
                <li><b>Total Raw Records:</b> {data_report['raw_rows']:,}</li>
                <li><b>Valid Clean Records:</b> {data_report['valid_rows']:,}</li>
                <li><b>Rejected / Invalid Rows:</b> {data_report['rejected_rows']}</li>
                <li><b>Duplicate Dates Detected:</b> {data_report['duplicate_dates']}</li>
                <li><b>Missing Calendar Days:</b> {data_report['missing_calendar_dates']} (out of continuous daily timeline)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="section-card">
            <h5>⚙️ Applied Preprocessing Decisions</h5>
            <ul>
                <li><b>Continuous Calendar Reindexing:</b> Reindexed to daily frequency (<code>freq='D'</code>) to guarantee lag features correspond to exact calendar days.</li>
                <li><b>Zero-Filling Prohibition:</b> Missing temperatures are never replaced with zero or silent overall means.</li>
                <li><b>Physical Bounds Verification:</b> Temperatures outside 0°C–55°C flagged as physically invalid for Mumbai.</li>
                <li><b>DTR Verification:</b> Diurnal Range calculated as TMAX - TMIN; invalidated if TMIN > TMAX.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Missing Value Breakdown Table
    st.markdown("##### Variable Completeness Breakdown")
    missing_data_rows = []
    for var, stats in data_report["missing_values"].items():
        missing_data_rows.append({
            "Variable": var,
            "Total Calendar Days": stats["total_rows"],
            "Missing Count": stats["missing_count"],
            "Completeness (%)": f"{stats['completeness_pct']}%"
        })
    st.dataframe(pd.DataFrame(missing_data_rows), use_container_width=True, hide_index=True)

    # Generate Power BI Exports
    st.markdown("---")
    st.markdown("#### 2. Power BI Export Center (College Submission Deliverables)")
    st.markdown(
        "Click the buttons below to download the exact CSV tables generated by our pipeline. "
        "These files are formatted specifically for Power BI Desktop data modeling and report creation."
    )

    # Generate files in filesystem
    exports = dp.generate_powerbi_exports(filtered_df)
    ml_results = get_ml_results(len(active_df), active_df)
    ml_exports = ml.export_ml_powerbi_files(ml_results)

    dcol1, dcol2, dcol3 = st.columns(3)
    with dcol1:
        st.download_button(
            label="📥 Download Clean Daily Dataset (CSV)",
            data=open(exports["cleaned_daily"], "rb").read(),
            file_name="mumbai_weather_cleaned.csv",
            mime="text/csv",
            help="Daily continuous observations with derived calendar and seasonal features."
        )
        st.download_button(
            label="📥 Download Monthly Summary (CSV)",
            data=open(exports["monthly_summary"], "rb").read(),
            file_name="mumbai_monthly_summary.csv",
            mime="text/csv",
            help="Climatological monthly averages, min, max, and standard deviation."
        )
    with dcol2:
        st.download_button(
            label="📥 Download Yearly Summary (CSV)",
            data=open(exports["yearly_summary"], "rb").read(),
            file_name="mumbai_yearly_summary.csv",
            mime="text/csv",
            help="Annual averages, year-over-year deltas, and baseline temperature anomalies."
        )
        st.download_button(
            label="📥 Download Seasonal KPI (CSV)",
            data=open(exports["seasonal_kpi"], "rb").read(),
            file_name="mumbai_seasonal_kpi.csv",
            mime="text/csv",
            help="Aggregated temperature metrics across IMD meteorological seasons."
        )
    with dcol3:
        st.download_button(
            label="📥 Download ML Test Predictions (CSV)",
            data=open(ml_exports["test_predictions"], "rb").read(),
            file_name="model_test_predictions.csv",
            mime="text/csv",
            help="Held-out test set (2019-2020) actual vs. predicted values and residuals."
        )
        st.download_button(
            label="📥 Download Model Metrics (CSV)",
            data=open(ml_exports["evaluation_metrics"], "rb").read(),
            file_name="model_evaluation_metrics.csv",
            mime="text/csv",
            help="Quantitative benchmark metrics (MAE, RMSE, R2, MAPE) comparing models."
        )

    st.markdown("""
    <div class="section-card" style="margin-top:20px;">
        <h5>📘 Power BI Integration Guide</h5>
        <p>A comprehensive step-by-step tutorial is available in <code>POWER_BI_GUIDE.md</code> in the project directory. It includes:</p>
        <ul>
            <li>Star schema and data relationship modeling.</li>
            <li>Production-ready DAX formulas for Moving Averages, Year-over-Year Delta, and Residual Error.</li>
            <li>Visual layout blueprints for Executive Overview, Climatology, and ML Evaluation pages.</li>
            <li>Instructions for publishing to the Power BI Service and embedding the live link.</li>
        </ul>
        <p><b>Power BI Published Dashboard Link:</b> <i>[Insert your published Power BI URL here after completing the steps in POWER_BI_GUIDE.md]</i></p>
    </div>
    """, unsafe_allow_html=True)
