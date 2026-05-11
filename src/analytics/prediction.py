from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split


def predict_congestion_risk(data: pd.DataFrame) -> dict[str, object]:
    if len(data) < 200 or data["congestion_level"].nunique() < 2:
        return {
            "available": False,
            "message": "Not enough data for a reliable congestion model yet.",
        }

    frame = data.copy()
    frame["target"] = (frame["congestion_level"] > 0.7).astype(int)
    features = frame[["latency_ms", "packet_loss_percent", "signal_strength_dbm", "download_speed_mbps", "upload_speed_mbps", "active_users"]]
    target = frame["target"]

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42, stratify=target)
    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    latest_row = features.tail(1)
    latest_probability = float(model.predict_proba(latest_row)[0][1])

    return {
        "available": True,
        "accuracy": round(float(accuracy_score(y_test, predictions)), 3),
        "congestion_probability": round(latest_probability, 3),
        "feature_importances": pd.Series(model.feature_importances_, index=features.columns).sort_values(ascending=False),
    }


def forecast_latency(data: pd.DataFrame) -> dict[str, object]:
    if len(data) < 200:
        return {
            "available": False,
            "message": "Not enough data for a stable latency forecast.",
        }

    frame = data.copy().sort_values("timestamp")
    frame["hour"] = frame["timestamp"].dt.hour
    features = frame[["packet_loss_percent", "signal_strength_dbm", "download_speed_mbps", "upload_speed_mbps", "active_users", "congestion_level", "hour"]]
    target = frame["latency_ms"]

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)
    model = RandomForestRegressor(n_estimators=120, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    next_latency = float(model.predict(features.tail(1))[0])

    return {
        "available": True,
        "mae": round(float(mean_absolute_error(y_test, predictions)), 2),
        "predicted_next_latency_ms": round(next_latency, 2),
    }