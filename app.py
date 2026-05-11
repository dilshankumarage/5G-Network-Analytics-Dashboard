from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.alerts.rules import build_alerts
from src.analytics.anomaly_detection import detect_isolation_forest_anomalies, detect_statistical_anomalies
from src.analytics.kpi_metrics import (
    compute_kpi_summary,
    congestion_analysis,
    network_health_score,
    peak_hour_analysis,
    sla_qos_summary,
    tower_performance_ranking,
)
from src.analytics.prediction import forecast_latency, predict_congestion_risk
from src.data_generation.generator import GeneratorConfig, generate_and_save_dataset
from src.utils.config import DEFAULT_DATASET_PATH, MODELS_DIR
from src.visualizations.charts import (
    anomaly_scatter,
    congestion_heatmap,
    latency_over_time,
    packet_loss_trends,
    signal_strength_trends,
    throughput_trends,
    tower_comparison,
)


st.set_page_config(page_title="5G Network Analytics Dashboard", page_icon="📡", layout="wide")


def inject_css(theme: str) -> None:
    if theme == "Dark Ops Center":
        background = "linear-gradient(180deg, #08111f 0%, #0d1728 100%)"
        card = "#121d33"
        text = "#eef4ff"
        muted = "#9bb0d0"
    else:
        background = "linear-gradient(180deg, #f5f9ff 0%, #e9f0fb 100%)"
        card = "#ffffff"
        text = "#10213a"
        muted = "#51637a"

    st.markdown(
        f"""
        <style>
        .stApp {{ background: {background}; color: {text}; }}
        .metric-card {{
            background: {card};
            border-radius: 16px;
            padding: 1rem 1.1rem;
            border: 1px solid rgba(126, 149, 182, 0.18);
            box-shadow: 0 12px 28px rgba(6, 18, 40, 0.08);
        }}
        .metric-label {{ color: {muted}; font-size: 0.84rem; text-transform: uppercase; letter-spacing: 0.06em; }}
        .metric-value {{ color: {text}; font-size: 1.6rem; font-weight: 700; margin-top: 0.25rem; }}
        .metric-subtitle {{ color: {muted}; font-size: 0.82rem; }}
        .section-card {{ background: {card}; border-radius: 18px; padding: 1rem 1rem 0.4rem 1rem; border: 1px solid rgba(126, 149, 182, 0.16); }}
        .alert-critical {{ border-left: 6px solid #d9534f; }}
        .alert-high {{ border-left: 6px solid #ff9f1a; }}
        .alert-medium {{ border-left: 6px solid #f0c419; }}
        .hero-title {{ font-size: 2.2rem; font-weight: 800; margin-bottom: 0.25rem; }}
        .hero-subtitle {{ color: {muted}; margin-bottom: 1rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        generate_and_save_dataset(path, GeneratorConfig())
    data = pd.read_csv(path, parse_dates=["timestamp"])
    return data


def kpi_card(label: str, value: object, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_alert_table(alerts: pd.DataFrame) -> pd.DataFrame:
    if alerts.empty:
        return alerts
    display = alerts.copy()
    display["timestamp"] = pd.to_datetime(display["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
    display["value"] = display["value"].round(2) if pd.api.types.is_numeric_dtype(display["value"]) else display["value"]
    return display.sort_values(["severity", "timestamp"], ascending=[True, False])


def main() -> None:
    st.sidebar.title("Network Controls")
    theme = st.sidebar.radio("Theme", ["Dark Ops Center", "Light Ops Center"], index=0)
    inject_css(theme)

    st.sidebar.caption("Dataset and model utilities stay local on your machine.")
    regenerate = st.sidebar.button("Generate Fresh Dataset")

    if regenerate or not DEFAULT_DATASET_PATH.exists():
        generate_and_save_dataset(DEFAULT_DATASET_PATH, GeneratorConfig())

    data = load_data(DEFAULT_DATASET_PATH)

    st.sidebar.markdown("---")
    tower_options = ["All Towers"] + sorted(data["tower_id"].unique().tolist())
    selected_tower = st.sidebar.selectbox("Tower", tower_options)

    min_date = data["timestamp"].min().date()
    max_date = data["timestamp"].max().date()
    start_date, end_date = st.sidebar.date_input("Date Range", (min_date, max_date), min_value=min_date, max_value=max_date)

    status_options = ["All", "UP", "DEGRADED", "CONGESTED", "OUTAGE"]
    selected_status = st.sidebar.selectbox("Network Status", status_options)

    filtered = data.copy()
    filtered = filtered[(filtered["timestamp"].dt.date >= start_date) & (filtered["timestamp"].dt.date <= end_date)]
    if selected_tower != "All Towers":
        filtered = filtered[filtered["tower_id"] == selected_tower]
    if selected_status != "All":
        filtered = filtered[filtered["network_status"] == selected_status]

    st.markdown('<div class="hero-title">5G Network Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Synthetic telecom operations dashboard for KPI monitoring, alerts, anomaly detection, and predictive analytics.</div>',
        unsafe_allow_html=True,
    )

    summary = compute_kpi_summary(filtered)
    health_score = network_health_score(filtered)
    sla_summary = sla_qos_summary(filtered)

    latest_timestamp = filtered["timestamp"].max().strftime("%Y-%m-%d %H:%M") if not filtered.empty else "N/A"
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card("Network Health", f"{health_score:.1f}/100", f"Latest data point: {latest_timestamp}")
    with col2:
        kpi_card("Average Latency", f"{summary.get('avg_latency_ms', 0):.1f} ms", "Latency performance across selected data")
    with col3:
        kpi_card("SLA Compliance", f"{sla_summary.get('sla_compliance_percent', 0):.1f}%", "Latency and packet loss within thresholds")
    with col4:
        kpi_card("Avg QoS Score", f"{summary.get('avg_qos_score', 0):.1f}", "Composite service quality indicator")

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    alerts = build_alerts(filtered)
    anomaly_stat = detect_statistical_anomalies(filtered)
    anomaly_iforest = detect_isolation_forest_anomalies(filtered)

    tab_overview, tab_towers, tab_alerts, tab_anomalies, tab_predictions = st.tabs(
        ["Overview", "Tower Analysis", "Alerts", "Anomalies", "Predictions"]
    )

    with tab_overview:
        left, right = st.columns(2)
        with left:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.plotly_chart(latency_over_time(filtered, theme == "Dark Ops Center"), width="stretch")
            st.plotly_chart(throughput_trends(filtered, theme == "Dark Ops Center"), width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)
        with right:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.plotly_chart(signal_strength_trends(filtered, theme == "Dark Ops Center"), width="stretch")
            st.plotly_chart(packet_loss_trends(filtered, theme == "Dark Ops Center"), width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div style="height: 0.5rem;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.plotly_chart(congestion_heatmap(filtered, theme == "Dark Ops Center"), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_towers:
        ranking = tower_performance_ranking(filtered)
        congestion = congestion_analysis(filtered)
        peak_hours = peak_hour_analysis(filtered)

        left, right = st.columns(2)
        with left:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.plotly_chart(tower_comparison(filtered, theme == "Dark Ops Center"), width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)
        with right:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.dataframe(ranking.style.format({"avg_latency_ms": "{:.2f}", "avg_download_speed_mbps": "{:.2f}", "avg_qos_score": "{:.2f}"}), width="stretch", height=340)
            st.markdown("</div>", unsafe_allow_html=True)

        bottom_left, bottom_right = st.columns(2)
        with bottom_left:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.dataframe(congestion.style.format({"avg_congestion_level": "{:.3f}", "peak_congestion": "{:.3f}"}), width="stretch", height=300)
            st.markdown("</div>", unsafe_allow_html=True)
        with bottom_right:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.dataframe(peak_hours.style.format({"avg_latency_ms": "{:.2f}", "avg_active_users": "{:.0f}", "avg_download_speed_mbps": "{:.2f}", "avg_congestion_level": "{:.3f}"}), width="stretch", height=300)
            st.markdown("</div>", unsafe_allow_html=True)

    with tab_alerts:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        if alerts.empty:
            st.success("No active alerts for the selected filters.")
        else:
            formatted_alerts = format_alert_table(alerts)
            st.dataframe(formatted_alerts, width="stretch", height=280)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_anomalies:
        stat_view = anomaly_stat[anomaly_stat["is_anomaly"]].copy()
        iforest_view = anomaly_iforest[anomaly_iforest["iforest_anomaly"]].copy()

        left, right = st.columns(2)
        with left:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.plotly_chart(anomaly_scatter(anomaly_stat, "is_anomaly", theme == "Dark Ops Center", "Statistical Anomaly Detection"), width="stretch")
            st.dataframe(stat_view[["timestamp", "tower_id", "latency_ms", "packet_loss_percent", "active_users", "congestion_level", "anomaly_score"]] if not stat_view.empty else stat_view, width="stretch", height=240)
            st.markdown("</div>", unsafe_allow_html=True)
        with right:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.plotly_chart(anomaly_scatter(anomaly_iforest, "iforest_anomaly", theme == "Dark Ops Center", "Isolation Forest Anomaly Detection"), width="stretch")
            st.dataframe(iforest_view[["timestamp", "tower_id", "latency_ms", "packet_loss_percent", "active_users", "congestion_level", "iforest_score"]] if not iforest_view.empty else iforest_view, width="stretch", height=240)
            st.markdown("</div>", unsafe_allow_html=True)

    with tab_predictions:
        congestion_prediction = predict_congestion_risk(filtered)
        latency_forecast = forecast_latency(filtered)

        left, right = st.columns(2)
        with left:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            if congestion_prediction.get("available"):
                st.metric("Congestion Risk Probability", f"{congestion_prediction['congestion_probability'] * 100:.1f}%", delta=f"Model accuracy: {congestion_prediction['accuracy'] * 100:.1f}%")
                st.write("Top predictive features")
                st.dataframe(congestion_prediction["feature_importances"].rename("importance"), width="stretch")
            else:
                st.info(congestion_prediction.get("message", "Prediction unavailable."))
            st.markdown("</div>", unsafe_allow_html=True)
        with right:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            if latency_forecast.get("available"):
                st.metric("Forecast Next Latency", f"{latency_forecast['predicted_next_latency_ms']:.1f} ms", delta=f"MAE: {latency_forecast['mae']:.1f} ms")
            else:
                st.info(latency_forecast.get("message", "Forecast unavailable."))
            st.caption(f"Model artifacts can be saved under {MODELS_DIR} when you expand the ML phase.")
            st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()