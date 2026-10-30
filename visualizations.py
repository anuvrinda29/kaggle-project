"""
visualizations.py
-----------------
Interactive Plotly charts styled with a polished Dark Navy / Charcoal palette
and high-contrast typography for Mumbai Weather Intelligence.
"""

from __future__ import annotations
from typing import Optional
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Dark Navy UI Theme Constants
THEME = {
    "bg_color": "#0B132B",
    "card_bg": "#1C2541",
    "surface_bg": "#1E293B",
    "text_primary": "#F8FAFC",
    "text_muted": "#94A3B8",
    "grid_color": "rgba(255, 255, 255, 0.08)",
    "accent_cyan": "#00F5D4",
    "accent_blue": "#38BDF8",
    "accent_indigo": "#818CF8",
    "accent_amber": "#FBBF24",
    "accent_coral": "#F43F5E",
    "accent_emerald": "#10B981"
}


def apply_theme_layout(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    """Apply uniform high-contrast dark navy layout styling to any Plotly figure."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "x": 0.01,
            "y": 0.96,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 16, "color": THEME["text_primary"]}
        },
        paper_bgcolor=THEME["card_bg"],
        plot_bgcolor="rgba(11, 19, 43, 0.5)",
        font={"family": "sans-serif", "color": THEME["text_primary"]},
        margin={"l": 50, "r": 30, "t": 60, "b": 40},
        height=height,
        hovermode="x unified",
        legend={
            "bgcolor": "rgba(28, 37, 65, 0.8)",
            "bordercolor": "rgba(255, 255, 255, 0.15)",
            "borderwidth": 1,
            "font": {"color": THEME["text_primary"]}
        }
    )
    fig.update_xaxes(
        gridcolor=THEME["grid_color"],
        zerolinecolor=THEME["grid_color"],
        linecolor="rgba(255, 255, 255, 0.2)",
        tickfont={"color": THEME["text_muted"]}
    )
    fig.update_yaxes(
        gridcolor=THEME["grid_color"],
        zerolinecolor=THEME["grid_color"],
        linecolor="rgba(255, 255, 255, 0.2)",
        tickfont={"color": THEME["text_muted"]}
    )
    return fig


# ==============================================================================
# PAGE 1: WEATHER OVERVIEW VISUALIZATIONS
# ==============================================================================

def plot_temperature_trend(
    df: pd.DataFrame,
    rolling_window: int = 30,
    show_rolling: bool = True
) -> go.Figure:
    """Historical daily temperature trend with customizable moving average overlay."""
    fig = go.Figure()

    # Observed daily TAVG points / line
    fig.add_trace(go.Scatter(
        x=df["DATE"],
        y=df["TAVG"],
        mode="lines",
        name="Daily Observed TAVG",
        line={"color": "rgba(56, 189, 248, 0.35)", "width": 1},
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Observed TAVG:</b> %{y:.1f} °C<extra></extra>"
    ))

    # Rolling average overlay
    if show_rolling and rolling_window > 1:
        rolling_series = df["TAVG"].rolling(window=rolling_window, min_periods=max(1, rolling_window // 2)).mean()
        fig.add_trace(go.Scatter(
            x=df["DATE"],
            y=rolling_series,
            mode="lines",
            name=f"{rolling_window}-Day Moving Average",
            line={"color": THEME["accent_cyan"], "width": 2.5},
            hovertemplate=f"<b>Date:</b> %{{x|%Y-%m-%d}}<br><b>{rolling_window}-Day Mean:</b> %{{y:.1f}} °C<extra></extra>"
        ))

    # Overall dataset mean reference line
    overall_mean = df["TAVG"].mean()
    fig.add_hline(
        y=overall_mean,
        line_dash="dot",
        line_color=THEME["accent_amber"],
        annotation_text=f"Historical Baseline Mean: {overall_mean:.1f}°C",
        annotation_position="bottom right",
        annotation_font_color=THEME["accent_amber"]
    )

    fig = apply_theme_layout(fig, f"Historical Temperature Trajectory with {rolling_window}-Day Moving Average")
    fig.update_xaxes(
        title_text="Date",
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=5, label="5Y", step="year", stepmode="backward"),
                dict(count=10, label="10Y", step="year", stepmode="backward"),
                dict(step="all", label="All")
            ]),
            bgcolor=THEME["surface_bg"],
            font=dict(color=THEME["text_primary"])
        )
    )
    fig.update_yaxes(title_text="Temperature (°C)")
    return fig


def plot_monthly_climatology(monthly_df: pd.DataFrame) -> go.Figure:
    """Climatological monthly average temperature bar chart with standard deviation error bars."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=monthly_df["MONTH_NAME"],
        y=monthly_df["avg_temp"],
        error_y=dict(type="data", array=monthly_df["std_temp"], visible=True, color=THEME["text_muted"]),
        marker=dict(
            color=monthly_df["avg_temp"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Avg °C", thickness=15, len=0.8)
        ),
        hovertemplate="<b>%{x}</b><br>Mean Temp: %{y:.2f} °C<br>Std Dev: ±%{error_y.array:.2f} °C<extra></extra>"
    ))

    fig = apply_theme_layout(fig, "Climatological Monthly Mean Temperature (°C) with Standard Deviation")
    fig.update_xaxes(title_text="Calendar Month")
    fig.update_yaxes(title_text="Mean Temperature (°C)", range=[monthly_df["avg_temp"].min() - 3, monthly_df["avg_temp"].max() + 3])
    return fig


def plot_yearly_trend(yearly_df: pd.DataFrame) -> go.Figure:
    """Annual average temperature with linear trendline and temperature anomalies."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Annual Average Temperature (°C)", "Temperature Anomaly Relative to Baseline Mean (°C)")
    )

    # Top plot: Annual mean line
    fig.add_trace(go.Scatter(
        x=yearly_df["YEAR"],
        y=yearly_df["avg_temp"],
        mode="lines+markers",
        name="Annual Mean",
        line=dict(color=THEME["accent_blue"], width=2.5),
        marker=dict(size=7, color=THEME["accent_cyan"]),
        hovertemplate="<b>Year:</b> %{x}<br><b>Mean Temp:</b> %{y:.2f} °C<extra></extra>"
    ), row=1, col=1)

    # Linear OLS trendline
    z = np.polyfit(yearly_df["YEAR"], yearly_df["avg_temp"], 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(
        x=yearly_df["YEAR"],
        y=p(yearly_df["YEAR"]),
        mode="lines",
        name=f"Linear Trend ({z[0]:+.3f}°C/yr)",
        line=dict(color=THEME["accent_coral"], dash="dash", width=2),
        hovertemplate="<b>Trendline:</b> %{y:.2f} °C<extra></extra>"
    ), row=1, col=1)

    # Bottom plot: Anomaly bars
    colors = np.where(yearly_df["temp_anomaly"] >= 0, THEME["accent_coral"], THEME["accent_cyan"])
    fig.add_trace(go.Bar(
        x=yearly_df["YEAR"],
        y=yearly_df["temp_anomaly"],
        name="Anomaly",
        marker_color=colors,
        hovertemplate="<b>Year:</b> %{x}<br><b>Anomaly:</b> %{y:+.2f} °C<extra></extra>"
    ), row=2, col=1)

    fig = apply_theme_layout(fig, "Yearly Historical Temperature Evolution and Annual Anomalies", height=500)
    fig.update_yaxes(title_text="Temp (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Anomaly (°C)", row=2, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=1, dtick=2)
    return fig


def plot_temperature_comparison(df: pd.DataFrame) -> go.Figure:
    """Multi-variable comparison: TAVG, TMAX, and TMIN time series."""
    fig = go.Figure()

    if "TMAX" in df.columns and df["TMAX"].notna().sum() > 0:
        fig.add_trace(go.Scatter(
            x=df["DATE"],
            y=df["TMAX"],
            mode="lines",
            name=f"TMAX (Max, n={int(df['TMAX'].notna().sum()):,})",
            line=dict(color=THEME["accent_coral"], width=1.2),
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>TMAX:</b> %{y:.1f} °C<extra></extra>"
        ))

    fig.add_trace(go.Scatter(
        x=df["DATE"],
        y=df["TAVG"],
        mode="lines",
        name=f"TAVG (Average, n={int(df['TAVG'].notna().sum()):,})",
        line=dict(color=THEME["accent_cyan"], width=1.5),
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>TAVG:</b> %{y:.1f} °C<extra></extra>"
    ))

    if "TMIN" in df.columns and df["TMIN"].notna().sum() > 0:
        fig.add_trace(go.Scatter(
            x=df["DATE"],
            y=df["TMIN"],
            mode="lines",
            name=f"TMIN (Min, n={int(df['TMIN'].notna().sum()):,})",
            line=dict(color=THEME["accent_indigo"], width=1.2),
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>TMIN:</b> %{y:.1f} °C<extra></extra>"
        ))

    fig = apply_theme_layout(fig, "Comparative Tri-Variable Temperature Timeline (TAVG, TMAX, TMIN)")
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Temperature (°C)")
    return fig


def plot_temperature_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of daily average temperatures with mean and median indicators."""
    valid_tavg = df["TAVG"].dropna()
    mean_val = valid_tavg.mean()
    median_val = valid_tavg.median()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=valid_tavg,
        nbinsx=35,
        name="Daily Observations",
        marker_color="rgba(0, 245, 212, 0.7)",
        marker_line=dict(color=THEME["accent_cyan"], width=1),
        hovertemplate="<b>Temp Range:</b> %{x} °C<br><b>Days:</b> %{y}<extra></extra>"
    ))

    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color=THEME["accent_amber"],
        line_width=2,
        annotation_text=f"Mean: {mean_val:.1f}°C",
        annotation_position="top left",
        annotation_font=dict(color=THEME["accent_amber"], size=12),
        annotation=dict(yshift=10, xshift=-14)
    )
    fig.add_vline(
        x=median_val,
        line_dash="dot",
        line_color=THEME["accent_coral"],
        line_width=2,
        annotation_text=f"Median: {median_val:.1f}°C",
        annotation_position="top right",
        annotation_font=dict(color=THEME["accent_coral"], size=12),
        annotation=dict(yshift=-28, xshift=14)
    )

    fig = apply_theme_layout(fig, "Daily Average Temperature Distribution Frequency", height=420)
    fig.update_xaxes(title_text="Average Temperature (°C)")
    fig.update_yaxes(title_text="Count of Observations (Days)")
    return fig


# ==============================================================================
# PAGE 2: ADVANCED VISUALIZATIONS & EXPLORATORY DATA ANALYSIS
# ==============================================================================

def plot_year_month_heatmap(df: pd.DataFrame) -> go.Figure:
    """Year-by-Month Temperature Heatmap (Advanced Viz 1)."""
    valid = df.dropna(subset=["TAVG"])
    pivot = valid.pivot_table(index="YEAR", columns="MONTH_NUMBER", values="TAVG", aggfunc="mean")
    
    # Month names for x-axis
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    x_labels = [month_names[m - 1] for m in pivot.columns]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=x_labels,
        y=pivot.index,
        colorscale="Turbo",
        colorbar=dict(title="Mean °C", thickness=15),
        hovertemplate="<b>Year:</b> %{y}<br><b>Month:</b> %{x}<br><b>Avg Temp:</b> %{z:.2f} °C<extra></extra>"
    ))

    fig = apply_theme_layout(fig, "Year × Month Thermal Matrix (Historical Climatological Heatmap)", height=480)
    fig.update_xaxes(title_text="Month of Year")
    fig.update_yaxes(title_text="Observation Year", dtick=2)
    return fig


def plot_monthly_boxplots(df: pd.DataFrame) -> go.Figure:
    """Monthly temperature dispersion and box plots (Advanced Viz 2)."""
    valid = df.dropna(subset=["TAVG"]).sort_values("MONTH_NUMBER")
    
    fig = px.box(
        valid,
        x="MONTH_NAME",
        y="TAVG",
        color="SEASON",
        color_discrete_map={
            "Winter": THEME["accent_blue"],
            "Summer": THEME["accent_amber"],
            "Monsoon": THEME["accent_emerald"],
            "Post-Monsoon": THEME["accent_indigo"]
        },
        category_orders={"MONTH_NAME": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]}
    )

    fig = apply_theme_layout(fig, "Monthly Temperature Variance (Box Plots)", height=440)
    fig.update_xaxes(title_text="Month of Year")
    fig.update_yaxes(title_text="Daily Average Temperature (°C)")
    return fig


def plot_temperature_correlation(df: pd.DataFrame) -> Optional[go.Figure]:
    """Scatter plot and correlation between TAVG and TMAX / TMIN with graceful statsmodels fallback."""
    valid = df.dropna(subset=["TAVG", "TMAX"]).copy()
    if len(valid) < 50:
        return None

    corr = valid["TAVG"].corr(valid["TMAX"])
    season_colors = {
        "Winter": THEME["accent_blue"],
        "Summer": THEME["accent_amber"],
        "Monsoon": THEME["accent_emerald"],
        "Post-Monsoon": THEME["accent_indigo"]
    }

    try:
        fig = px.scatter(
            valid,
            x="TAVG",
            y="TMAX",
            color="SEASON",
            trendline="ols",
            opacity=0.6,
            color_discrete_map=season_colors
        )
        title_text = f"Bivariate Correlation: TAVG vs. TMAX (Pearson r = {corr:.3f}, with OLS Trend)"
    except Exception:
        fig = px.scatter(
            valid,
            x="TAVG",
            y="TMAX",
            color="SEASON",
            opacity=0.6,
            color_discrete_map=season_colors
        )
        title_text = f"Bivariate Correlation: TAVG vs. TMAX (Pearson r = {corr:.3f})"

    fig = apply_theme_layout(fig, title_text, height=440)
    fig.update_xaxes(title_text="Daily Average Temperature TAVG (°C)")
    fig.update_yaxes(title_text="Daily Maximum Temperature TMAX (°C)")
    return fig


def plot_data_completeness(df: pd.DataFrame) -> go.Figure:
    """Visual completeness and missing observation percentage across the timeline."""
    annual_stats = []
    years = sorted(df["YEAR"].unique())

    for yr in years:
        sub = df[df["YEAR"] == yr]
        total_days = len(sub)
        tavg_pct = (sub["TAVG"].notna().sum() / total_days) * 100
        tmax_pct = (sub["TMAX"].notna().sum() / total_days) * 100 if "TMAX" in sub else 0
        tmin_pct = (sub["TMIN"].notna().sum() / total_days) * 100 if "TMIN" in sub else 0
        annual_stats.append({
            "Year": yr,
            "TAVG Completeness (%)": round(tavg_pct, 1),
            "TMAX Completeness (%)": round(tmax_pct, 1),
            "TMIN Completeness (%)": round(tmin_pct, 1)
        })

    comp_df = pd.DataFrame(annual_stats)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=comp_df["Year"], y=comp_df["TAVG Completeness (%)"],
        mode="lines+markers", name="TAVG Availability",
        line=dict(color=THEME["accent_cyan"], width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=comp_df["Year"], y=comp_df["TMAX Completeness (%)"],
        mode="lines+markers", name="TMAX Availability",
        line=dict(color=THEME["accent_coral"], width=1.5, dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=comp_df["Year"], y=comp_df["TMIN Completeness (%)"],
        mode="lines+markers", name="TMIN Availability",
        line=dict(color=THEME["accent_indigo"], width=1.5, dash="dash")
    ))

    fig = apply_theme_layout(fig, "Annual Variable Availability and Completeness Audit (2000–2020)", height=440)
    fig.update_xaxes(title_text="Year", dtick=2)
    fig.update_yaxes(title_text="Completeness (%)", range=[0, 105])
    return fig


def plot_outliers(df: pd.DataFrame, outliers_df: pd.DataFrame) -> go.Figure:
    """Outlier detection visualization using Tukey's IQR rule."""
    fig = go.Figure()

    # Main observations
    fig.add_trace(go.Scatter(
        x=df["DATE"],
        y=df["TAVG"],
        mode="lines",
        name="Normal Temperature Range",
        line=dict(color="rgba(148, 163, 184, 0.4)", width=1),
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>TAVG:</b> %{y:.1f} °C<extra></extra>"
    ))

    # Heatwave outliers
    heatwaves = outliers_df[outliers_df["outlier_type"] == "High Outlier (Heatwave)"]
    if len(heatwaves) > 0:
        fig.add_trace(go.Scatter(
            x=heatwaves["DATE"],
            y=heatwaves["TAVG"],
            mode="markers",
            name=f"High Outliers / Heatwaves (n={len(heatwaves)})",
            marker=dict(color=THEME["accent_coral"], size=8, symbol="triangle-up"),
            hovertemplate="<b>Heatwave:</b> %{x|%Y-%m-%d}<br><b>Temp:</b> %{y:.1f} °C<extra></extra>"
        ))

    # Cold snap outliers
    coldsnaps = outliers_df[outliers_df["outlier_type"] == "Low Outlier (Cold Snap)"]
    if len(coldsnaps) > 0:
        fig.add_trace(go.Scatter(
            x=coldsnaps["DATE"],
            y=coldsnaps["TAVG"],
            mode="markers",
            name=f"Low Outliers / Cold Snaps (n={len(coldsnaps)})",
            marker=dict(color=THEME["accent_cyan"], size=8, symbol="triangle-down"),
            hovertemplate="<b>Cold Snap:</b> %{x|%Y-%m-%d}<br><b>Temp:</b> %{y:.1f} °C<extra></extra>"
        ))

    fig = apply_theme_layout(fig, "Tukey IQR Statistical Outlier Identification (1.5 × IQR Threshold)", height=440)
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Temperature (°C)")
    return fig


# ==============================================================================
# PAGE 3: MACHINE LEARNING & PREDICTION VISUALIZATIONS
# ==============================================================================

def plot_actual_vs_predicted(predictions_df: pd.DataFrame) -> go.Figure:
    """Out-of-sample time-series comparison: Actual vs. RF Predicted vs. Baseline."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=predictions_df["DATE"],
        y=predictions_df["ACTUAL_TAVG"],
        mode="lines",
        name="Actual Ground Truth TAVG",
        line=dict(color=THEME["text_primary"], width=1.8),
        hovertemplate="<b>Date:</b> %{x}<br><b>Actual:</b> %{y:.1f} °C<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=predictions_df["DATE"],
        y=predictions_df["PRED_RF"],
        mode="lines",
        name="Random Forest Prediction",
        line=dict(color=THEME["accent_cyan"], width=1.8),
        hovertemplate="<b>Date:</b> %{x}<br><b>RF Pred:</b> %{y:.1f} °C<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=predictions_df["DATE"],
        y=predictions_df["PRED_BASELINE"],
        mode="lines",
        name="Persistence Baseline (t-1)",
        line=dict(color=THEME["accent_coral"], width=1.0, dash="dot"),
        hovertemplate="<b>Date:</b> %{x}<br><b>Baseline:</b> %{y:.1f} °C<extra></extra>"
    ))

    fig = apply_theme_layout(fig, "Held-Out Test Set: Actual vs. Predicted Temperature (Chronological Split)")
    fig.update_xaxes(title_text="Test Set Date", rangeslider_visible=True)
    fig.update_yaxes(title_text="Temperature (°C)")
    return fig


def plot_prediction_scatter(predictions_df: pd.DataFrame) -> go.Figure:
    """Parity scatter plot (Actual vs Predicted) with ideal 45-degree reference line."""
    min_val = min(predictions_df["ACTUAL_TAVG"].min(), predictions_df["PRED_RF"].min()) - 1
    max_val = max(predictions_df["ACTUAL_TAVG"].max(), predictions_df["PRED_RF"].max()) + 1

    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=predictions_df["ACTUAL_TAVG"],
        y=predictions_df["PRED_RF"],
        mode="markers",
        name="Predictions",
        marker=dict(color=THEME["accent_cyan"], size=6, opacity=0.6),
        hovertemplate="<b>Actual:</b> %{x:.1f} °C<br><b>Predicted:</b> %{y:.1f} °C<extra></extra>"
    ))

    # Perfect prediction line (y = x)
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        name="Perfect Agreement (y = x)",
        line=dict(color=THEME["accent_coral"], dash="dash", width=2)
    ))

    fig = apply_theme_layout(fig, "Prediction Parity: Actual vs. Random Forest Predicted TAVG (°C)")
    fig.update_xaxes(title_text="Actual Ground Truth Temperature (°C)", range=[min_val, max_val])
    fig.update_yaxes(title_text="Predicted Model Temperature (°C)", range=[min_val, max_val])
    return fig


def plot_residuals(predictions_df: pd.DataFrame) -> go.Figure:
    """Residual diagnostic subplots: Residuals vs Time and Residual Distribution."""
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.65, 0.35],
        subplot_titles=("Residual Sequence Over Test Period (y - ŷ)", "Residual Error Distribution")
    )

    # Residuals vs Time
    fig.add_trace(go.Scatter(
        x=predictions_df["DATE"],
        y=predictions_df["RESIDUAL_RF"],
        mode="markers",
        name="Residuals",
        marker=dict(color=THEME["accent_cyan"], size=5, opacity=0.6),
        hovertemplate="<b>Date:</b> %{x}<br><b>Residual:</b> %{y:.2f} °C<extra></extra>"
    ), row=1, col=1)

    fig.add_hline(y=0, line_dash="dash", line_color=THEME["accent_coral"], row=1, col=1)

    # Residual Histogram
    fig.add_trace(go.Histogram(
        x=predictions_df["RESIDUAL_RF"],
        nbinsx=30,
        name="Error Dist",
        marker_color="rgba(0, 245, 212, 0.7)",
        marker_line=dict(color=THEME["accent_cyan"], width=1)
    ), row=1, col=2)

    fig.add_vline(x=0, line_dash="dash", line_color=THEME["accent_coral"], row=1, col=2)

    fig = apply_theme_layout(fig, "Model Residual Diagnostics (Error Analysis)", height=480)
    fig.update_layout(
        title={
            "text": "<b>Model Residual Diagnostics (Error Analysis)</b>",
            "x": 0.01,
            "y": 0.98,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 16, "color": THEME["text_primary"]}
        },
        margin={"l": 50, "r": 30, "t": 85, "b": 45}
    )
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Residual Error (°C)", row=1, col=1)
    fig.update_xaxes(title_text="Error (°C)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    return fig


def plot_feature_importance(importances_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of Random Forest feature importances."""
    sorted_df = importances_df.sort_values("Importance", ascending=True)

    fig = go.Figure(go.Bar(
        x=sorted_df["Importance"],
        y=sorted_df["Feature"],
        orientation="h",
        marker=dict(
            color=sorted_df["Importance"],
            colorscale="Viridis",
            showscale=False
        ),
        hovertemplate="<b>%{y}:</b> %{x:.4f}<extra></extra>"
    ))

    fig = apply_theme_layout(fig, "Random Forest Feature Importance (MDI)")
    fig.update_xaxes(title_text="Relative Feature Importance")
    fig.update_yaxes(title_text="Predictive Feature")
    return fig
