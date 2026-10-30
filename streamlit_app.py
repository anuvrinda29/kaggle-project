"""Mumbai Weather Intelligence: Streamlit app.
Run the pipeline first (python mumbai_weather_pipeline.py), then:  streamlit run app/streamlit_app.py
"""
import os

import pandas as pd
import streamlit as st

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")

st.set_page_config(page_title="Mumbai Weather Intelligence", layout="wide")


@st.cache_data
def load(name, **kwargs):
    return pd.read_csv(os.path.join(DATA, name), **kwargs)


try:
    clean = load("weather_clean.csv", parse_dates=["date"])
    monthly, yearly, seasonal = load("monthly_summary.csv"), load("yearly_summary.csv"), load("seasonal_summary.csv")
    metrics, preds = load("model_metrics.csv"), load("test_predictions.csv", parse_dates=["date"])
except FileNotFoundError:
    st.error("Processed files not found. Run `python mumbai_weather_pipeline.py` first.")
    st.stop()

st.title("Mumbai Weather Intelligence")
st.caption("Historical temperature patterns and model evaluation | Bombay Santacruz station | 2000-2020")

tab1, tab2, tab3 = st.tabs(["Weather Overview", "Advanced Analysis", "Model Evaluation"])

with tab1:
    y0, y1 = int(clean["year"].min()), int(clean["year"].max())
    years = st.slider("Year range", y0, y1, (y0, y1))
    sel = clean[(clean["year"] >= years[0]) & (clean["year"] <= years[1])]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean daily average (TAVG)", f"{sel['tavg'].mean():.2f} °C")
    c2.metric("Highest TMAX", f"{sel['tmax'].max():.2f} °C")
    c3.metric("Lowest TMIN", f"{sel['tmin'].min():.2f} °C")
    c4.metric("Valid TAVG records", f"{int(sel['tavg'].notna().sum()):,}")
    left, right = st.columns(2)
    left.subheader("Annual average temperature")
    left.line_chart(sel.groupby("year")["tavg"].mean())
    right.subheader("Monthly average temperature")
    right.bar_chart(sel.groupby("month")["tavg"].mean())

with tab2:
    left, right = st.columns(2)
    left.subheader("Annual anomaly vs long-term mean")
    left.bar_chart(yearly.set_index("year")["temp_anomaly"])
    right.subheader("Average temperature by season")
    right.bar_chart(seasonal.set_index("season")["avg_temp"])
    st.subheader("Monthly temperature profile")
    st.bar_chart(monthly.set_index("month")[["avg_temp", "max_temp", "min_temp"]])
    st.info("Descriptive patterns from one station: not proof of cause or city-wide effects.")

with tab3:
    st.subheader(f"Held-out benchmark ({len(preds):,} daily rows)")
    st.dataframe(metrics, hide_index=True, use_container_width=True)
    st.subheader("Actual vs predicted temperature")
    st.line_chart(preds.set_index("date")[["actual_tavg", "pred_baseline", "pred_rf", "pred_ridge"]])
    st.subheader("Random Forest residuals over time")
    st.line_chart(preds.set_index("date")["residual_rf"])
    st.caption("Rolling one-step-ahead experiment, not a multi-week operational forecast.")
