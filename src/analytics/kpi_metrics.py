from __future__ import annotations

import pandas as pd


def compute_kpi_summary(data: pd.DataFrame) -> dict[str, float]:
    if data.empty:
        return {}

    summary = {
        "avg_latency_ms": round(data["latency_ms"].mean(), 2),
        "avg_packet_loss_percent": round(data["packet_loss_percent"].mean(), 2),
        "avg_signal_strength_dbm": round(data["signal_strength_dbm"].mean(), 2),
        "avg_download_speed_mbps": round(data["download_speed_mbps"].mean(), 2),
        "avg_upload_speed_mbps": round(data["upload_speed_mbps"].mean(), 2),
        "avg_active_users": round(data["active_users"].mean(), 0),
        "avg_congestion_level": round(data["congestion_level"].mean(), 3),
        "availability_percent": round(data["availability_percent"].mean(), 2),
        "avg_qos_score": round(data["qos_score"].mean(), 2),
    }
    return summary


def tower_performance_ranking(data: pd.DataFrame) -> pd.DataFrame:
    ranking = (
        data.groupby(["tower_id", "location"], as_index=False)
        .agg(
            avg_latency_ms=("latency_ms", "mean"),
            avg_packet_loss_percent=("packet_loss_percent", "mean"),
            avg_download_speed_mbps=("download_speed_mbps", "mean"),
            avg_signal_strength_dbm=("signal_strength_dbm", "mean"),
            avg_congestion_level=("congestion_level", "mean"),
            avg_qos_score=("qos_score", "mean"),
            availability_percent=("availability_percent", "mean"),
        )
        .sort_values(["avg_qos_score", "avg_download_speed_mbps"], ascending=False)
        .reset_index(drop=True)
    )
    ranking["rank"] = ranking.index + 1
    return ranking


def congestion_analysis(data: pd.DataFrame) -> pd.DataFrame:
    analysis = (
        data.groupby("tower_id", as_index=False)
        .agg(avg_congestion_level=("congestion_level", "mean"), peak_congestion=("congestion_level", "max"), high_congestion_minutes=("congestion_level", lambda series: int((series > 0.75).sum() * 5)))
        .sort_values("avg_congestion_level", ascending=False)
    )
    return analysis


def peak_hour_analysis(data: pd.DataFrame) -> pd.DataFrame:
    hourly = data.copy()
    hourly["hour"] = hourly["timestamp"].dt.hour
    result = (
        hourly.groupby(["hour"], as_index=False)
        .agg(
            avg_latency_ms=("latency_ms", "mean"),
            avg_active_users=("active_users", "mean"),
            avg_download_speed_mbps=("download_speed_mbps", "mean"),
            avg_congestion_level=("congestion_level", "mean"),
        )
        .sort_values("hour")
    )
    return result


def network_health_score(data: pd.DataFrame) -> float:
    if data.empty:
        return 0.0

    latency_score = max(0, 100 - data["latency_ms"].mean())
    packet_score = max(0, 100 - data["packet_loss_percent"].mean() * 20)
    signal_score = min(100, max(0, (data["signal_strength_dbm"].mean() + 120) * 1.25))
    congestion_score = max(0, 100 - data["congestion_level"].mean() * 100)
    qos_score = data["qos_score"].mean()

    overall = (latency_score + packet_score + signal_score + congestion_score + qos_score) / 5
    return round(float(np_clip(overall, 0, 100)), 2)


def sla_qos_summary(data: pd.DataFrame, latency_threshold: float = 50.0, packet_loss_threshold: float = 1.0) -> dict[str, float]:
    total = len(data)
    if total == 0:
        return {"sla_compliance_percent": 0.0, "qos_above_80_percent": 0.0}

    sla_ok = data[(data["latency_ms"] <= latency_threshold) & (data["packet_loss_percent"] <= packet_loss_threshold)]
    qos_ok = data[data["qos_score"] >= 80]

    return {
        "sla_compliance_percent": round((len(sla_ok) / total) * 100, 2),
        "qos_above_80_percent": round((len(qos_ok) / total) * 100, 2),
    }


def np_clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))