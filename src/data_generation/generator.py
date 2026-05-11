from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import DEFAULT_DATASET_PATH, TOWER_LOCATIONS


@dataclass(frozen=True)
class GeneratorConfig:
    days: int = 30
    frequency: str = "5min"
    seed: int = 42


def _time_features(timestamp: pd.Timestamp) -> tuple[int, int, int]:
    hour = timestamp.hour
    day_of_week = timestamp.dayofweek
    is_weekend = int(day_of_week >= 5)
    return hour, day_of_week, is_weekend


def _peak_factor(hour: int) -> float:
    if 7 <= hour <= 10:
        return 1.25
    if 18 <= hour <= 22:
        return 1.4
    if 0 <= hour <= 5:
        return 0.7
    return 1.0


def _weekend_factor(is_weekend: int) -> float:
    return 0.9 if is_weekend else 1.0


def generate_telecom_dataset(config: GeneratorConfig | None = None) -> pd.DataFrame:
    config = config or GeneratorConfig()
    rng = np.random.default_rng(config.seed)

    end_time = pd.Timestamp.now().floor("5min")
    start_time = end_time - pd.Timedelta(days=config.days)
    timestamps = pd.date_range(start=start_time, end=end_time, freq=config.frequency)

    rows: list[dict[str, object]] = []
    tower_profiles: dict[str, dict[str, object]] = {}

    for tower_id, location in TOWER_LOCATIONS:
        capacity = int(rng.integers(180, 420))
        base_users = int(rng.integers(70, 160))
        signal_base = float(rng.uniform(-78, -64))
        tower_profiles[tower_id] = {
            "location": location,
            "capacity": capacity,
            "base_users": base_users,
            "signal_base": signal_base,
        }

    outage_map: dict[str, set[pd.Timestamp]] = {}
    for tower_id in tower_profiles:
        outage_starts = rng.choice(len(timestamps), size=3, replace=False)
        outage_periods: set[pd.Timestamp] = set()
        for start_idx in outage_starts:
            for offset in range(6):
                index = min(start_idx + offset, len(timestamps) - 1)
                outage_periods.add(timestamps[index])
        outage_map[tower_id] = outage_periods

    for tower_id, profile in tower_profiles.items():
        for timestamp in timestamps:
            hour, day_of_week, is_weekend = _time_features(timestamp)
            peak = _peak_factor(hour)
            weekend = _weekend_factor(is_weekend)
            seasonal_noise = rng.normal(0, 8)

            traffic_multiplier = peak * weekend
            expected_users = profile["base_users"] * traffic_multiplier
            traffic_spike = 1.0 + (0.35 if rng.random() < 0.02 else 0.0)
            active_users = int(max(0, expected_users * traffic_spike + seasonal_noise + rng.normal(0, 12)))

            congestion_ratio = active_users / profile["capacity"]
            congestion_level = float(np.clip((congestion_ratio - 0.55) / 0.7, 0, 1))

            is_outage = timestamp in outage_map[tower_id]
            status = "OUTAGE" if is_outage else "UP"
            if not is_outage:
                if congestion_level > 0.8:
                    status = "CONGESTED"
                elif congestion_level > 0.55:
                    status = "DEGRADED"

            signal_strength = profile["signal_base"] - congestion_level * 10 + rng.normal(0, 2)
            if rng.random() < 0.01:
                signal_strength -= rng.uniform(8, 18)

            latency_ms = 16 + congestion_level * 70 + rng.normal(0, 5)
            packet_loss_percent = np.clip(0.15 + congestion_level * 3.5 + rng.normal(0, 0.25), 0, 8)

            download_speed_mbps = np.clip(920 - congestion_level * 760 + (signal_strength + 120) * 2.5 + rng.normal(0, 35), 20, 1200)
            upload_speed_mbps = np.clip(180 - congestion_level * 140 + (signal_strength + 120) * 0.8 + rng.normal(0, 8), 5, 250)

            if is_outage:
                active_users = int(active_users * 0.1)
                latency_ms = latency_ms + rng.uniform(80, 180)
                packet_loss_percent = np.clip(packet_loss_percent + rng.uniform(4, 8), 0, 100)
                download_speed_mbps = rng.uniform(0, 8)
                upload_speed_mbps = rng.uniform(0, 3)
                signal_strength = rng.uniform(-120, -110)
                congestion_level = 1.0

            availability = 99.95 if not is_outage else 92.5
            qos_score = np.clip(
                100
                - (latency_ms * 0.5)
                - (packet_loss_percent * 8)
                - max(0, (congestion_level - 0.6) * 30)
                + (download_speed_mbps / 25)
                + (upload_speed_mbps / 25),
                0,
                100,
            )

            rows.append(
                {
                    "timestamp": timestamp,
                    "tower_id": tower_id,
                    "location": profile["location"],
                    "active_users": active_users,
                    "latency_ms": round(float(max(latency_ms, 0.1)), 2),
                    "packet_loss_percent": round(float(packet_loss_percent), 2),
                    "signal_strength_dbm": round(float(signal_strength), 2),
                    "download_speed_mbps": round(float(download_speed_mbps), 2),
                    "upload_speed_mbps": round(float(upload_speed_mbps), 2),
                    "congestion_level": round(float(np.clip(congestion_level, 0, 1)), 3),
                    "network_status": status,
                    "availability_percent": round(float(availability), 2),
                    "qos_score": round(float(qos_score), 2),
                }
            )

    frame = pd.DataFrame(rows).sort_values(["timestamp", "tower_id"]).reset_index(drop=True)
    return frame


def save_dataset(frame: pd.DataFrame, path: Path | str = DEFAULT_DATASET_PATH) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path


def generate_and_save_dataset(path: Path | str = DEFAULT_DATASET_PATH, config: GeneratorConfig | None = None) -> Path:
    frame = generate_telecom_dataset(config=config)
    return save_dataset(frame, path)