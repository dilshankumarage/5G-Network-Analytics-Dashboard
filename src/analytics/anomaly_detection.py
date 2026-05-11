from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def detect_statistical_anomalies(data: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
    if data.empty:
        return data.assign(anomaly_score=[], is_anomaly=[])

    frame = data.copy()
    metrics = ["latency_ms", "packet_loss_percent", "active_users", "congestion_level"]
    z_scores = pd.DataFrame(index=frame.index)
    for metric in metrics:
        series = frame[metric].astype(float)
        std = series.std(ddof=0)
        if std == 0 or pd.isna(std):
            z_scores[metric] = 0.0
        else:
            z_scores[metric] = (series - series.mean()) / std

    frame["anomaly_score"] = z_scores.abs().max(axis=1)
    frame["is_anomaly"] = frame["anomaly_score"] > z_threshold
    return frame


def detect_isolation_forest_anomalies(data: pd.DataFrame, contamination: float = 0.03) -> pd.DataFrame:
    if data.empty:
        return data.assign(iforest_score=[], iforest_anomaly=[])

    frame = data.copy()
    features = frame[["latency_ms", "packet_loss_percent", "signal_strength_dbm", "download_speed_mbps", "upload_speed_mbps", "active_users", "congestion_level"]].astype(float)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    model = IsolationForest(contamination=contamination, random_state=42)
    predictions = model.fit_predict(scaled_features)
    frame["iforest_score"] = model.decision_function(scaled_features)
    frame["iforest_anomaly"] = predictions == -1
    return frame