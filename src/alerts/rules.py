from __future__ import annotations

import pandas as pd


def build_alerts(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=["timestamp", "tower_id", "severity", "alert_type", "message", "value"])

    latest_rows = data.sort_values("timestamp").groupby("tower_id", as_index=False).tail(1)
    alerts: list[dict[str, object]] = []

    for _, row in latest_rows.iterrows():
        if row["network_status"] == "OUTAGE":
            alerts.append(_alert(row, "critical", "tower_outage", "Tower outage detected", row["network_status"]))
        if row["latency_ms"] > 80:
            alerts.append(_alert(row, "high", "high_latency", "High latency above threshold", row["latency_ms"]))
        if row["packet_loss_percent"] > 2.5:
            alerts.append(_alert(row, "high", "packet_loss", "Packet loss warning", row["packet_loss_percent"]))
        if row["congestion_level"] > 0.8:
            alerts.append(_alert(row, "critical", "severe_congestion", "Severe congestion detected", row["congestion_level"]))
        if row["signal_strength_dbm"] < -100:
            alerts.append(_alert(row, "medium", "weak_signal", "Weak signal quality detected", row["signal_strength_dbm"]))

    return pd.DataFrame(alerts)


def _alert(row: pd.Series, severity: str, alert_type: str, message: str, value: object) -> dict[str, object]:
    return {
        "timestamp": row["timestamp"],
        "tower_id": row["tower_id"],
        "severity": severity,
        "alert_type": alert_type,
        "message": message,
        "value": value,
    }