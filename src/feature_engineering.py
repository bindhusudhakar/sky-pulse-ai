"""
src/feature_engineering.py
Advanced feature engineering for flight delay prediction.
Adds derived features to improve model performance.
"""

import pandas as pd
import numpy as np


# ── Weather severity mapping ──────────────────────────────────────────────────
WEATHER_SEVERITY = {
    "Clear": 0,
    "Cloudy": 1,
    "Rain": 2,
    "Wind": 3,
    "Fog": 4,
    "Snow": 5,
    "Thunderstorm": 6,
}

# ── Airline reliability scores (historical on-time performance, 0–1) ──────────
AIRLINE_RELIABILITY = {
    "AS": 0.82,  # Alaska – best
    "DL": 0.80,  # Delta
    "AA": 0.76,  # American
    "UA": 0.75,  # United
    "WN": 0.74,  # Southwest
    "B6": 0.70,  # JetBlue
    "F9": 0.64,  # Frontier
    "NK": 0.62,  # Spirit – worst
}

# ── Airport hub tier (1=mega hub, 2=major, 3=regional) ───────────────────────
AIRPORT_TIER = {
    "ATL": 1, "LAX": 1, "ORD": 1, "DFW": 1,
    "DEN": 2, "JFK": 1, "SFO": 2, "SEA": 2,
    "LAS": 2, "MCO": 2, "MIA": 2, "PHX": 2,
    "BOS": 2, "MSP": 2, "DTW": 2, "PHL": 2,
    "LGA": 1, "BWI": 3, "SLC": 3, "CLT": 2,
}


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive cyclical and categorical time features."""
    df = df.copy()

    # Peak hour flag (morning rush + evening rush)
    df["is_peak_hour"] = df["dep_hour"].apply(
        lambda h: 1 if h in [7, 8, 9, 17, 18, 19] else 0
    )

    # Early morning (less congestion, but crew/logistics risk)
    df["is_early_morning"] = (df["dep_hour"] < 7).astype(int)

    # Red-eye / late night
    df["is_red_eye"] = (df["dep_hour"] >= 21).astype(int)

    # Weekend flag
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Holiday months (approximate high-traffic months)
    df["is_holiday_month"] = df["month"].isin([6, 7, 8, 11, 12]).astype(int)

    # Cyclical encoding for hour (captures wrap-around midnight)
    df["hour_sin"] = np.sin(2 * np.pi * df["dep_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["dep_hour"] / 24)

    # Cyclical encoding for month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert weather conditions into numeric severity + interaction features."""
    df = df.copy()

    df["origin_weather_severity"] = df["origin_weather"].map(WEATHER_SEVERITY).fillna(0)
    df["dest_weather_severity"] = df["dest_weather"].map(WEATHER_SEVERITY).fillna(0)

    # Combined weather severity (both endpoints matter)
    df["combined_weather_severity"] = (
        df["origin_weather_severity"] * 0.7 + df["dest_weather_severity"] * 0.3
    )

    # Severe weather flag
    df["severe_weather"] = (df["origin_weather_severity"] >= 4).astype(int)

    # Weather * congestion interaction (bad weather at busy airport = high risk)
    df["weather_congestion_risk"] = (
        df["origin_weather_severity"] * df["airport_congestion"] / 100
    )

    return df


def add_congestion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute congestion-based risk features."""
    df = df.copy()

    # Congestion tier
    df["congestion_tier"] = pd.cut(
        df["airport_congestion"],
        bins=[0, 60, 70, 80, 90, 100],
        labels=[0, 1, 2, 3, 4],
    ).astype(float)

    # High congestion flag
    df["high_congestion"] = (df["airport_congestion"] > 80).astype(int)

    # Airport tier
    df["origin_airport_tier"] = df["origin"].map(AIRPORT_TIER).fillna(2)
    df["dest_airport_tier"] = df["dest"].map(AIRPORT_TIER).fillna(2)

    # Hub-to-hub route (most congestion-sensitive)
    df["hub_to_hub"] = (
        (df["origin_airport_tier"] == 1) & (df["dest_airport_tier"] == 1)
    ).astype(int)

    return df


def add_airline_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add airline reliability and operational features."""
    df = df.copy()

    df["airline_reliability"] = df["airline"].map(AIRLINE_RELIABILITY).fillna(0.72)

    # Low-cost carrier flag (Spirit, Frontier, Southwest, JetBlue)
    df["is_lcc"] = df["airline"].isin(["NK", "F9", "WN", "B6"]).astype(int)

    return df


def add_flight_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add flight-level operational features."""
    df = df.copy()

    # Long-haul flag
    df["is_long_haul"] = (df["distance"] > 2000).astype(int)

    # Short-haul (more turnaround risk)
    df["is_short_haul"] = (df["distance"] < 500).astype(int)

    # Tight turnaround risk
    df["tight_turnaround"] = (df["turnaround_time"] < 30).astype(int)

    # Maintenance + old aircraft compound risk
    df["maint_age_risk"] = df["maintenance_flag"] * np.log1p(df["aircraft_age"])

    # NAS delay present
    df["has_nas_delay"] = (df["nas_delay"] > 0).astype(int)

    # Carrier delay risk (normalized)
    df["carrier_risk"] = np.log1p(df["carrier_delay_history"])

    return df


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in sequence."""
    print("[FeatureEng] Starting feature engineering...")
    n_before = len(df.columns)

    df = add_time_features(df)
    df = add_weather_features(df)
    df = add_congestion_features(df)
    df = add_airline_features(df)
    df = add_flight_features(df)

    n_after = len(df.columns)
    print(f"[FeatureEng] Features: {n_before} → {n_after} (+{n_after - n_before} engineered)")
    return df


def get_engineered_feature_columns():
    """Return complete list of features for model training."""
    return [
        # Base features
        "month", "day_of_week", "dep_hour", "distance",
        "airport_congestion", "aircraft_age", "turnaround_time",
        "maintenance_flag", "carrier_delay_history", "nas_delay",
        # Encoded categoricals
        "airline_encoded", "origin_encoded", "dest_encoded",
        "origin_weather_encoded", "dest_weather_encoded",
        # Engineered time
        "is_peak_hour", "is_early_morning", "is_red_eye",
        "is_weekend", "is_holiday_month",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        # Engineered weather
        "origin_weather_severity", "dest_weather_severity",
        "combined_weather_severity", "severe_weather",
        "weather_congestion_risk",
        # Engineered congestion
        "congestion_tier", "high_congestion",
        "origin_airport_tier", "dest_airport_tier", "hub_to_hub",
        # Engineered airline
        "airline_reliability", "is_lcc",
        # Engineered flight
        "is_long_haul", "is_short_haul", "tight_turnaround",
        "maint_age_risk", "has_nas_delay", "carrier_risk",
    ]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__.replace("feature_engineering.py", "")))
    from preprocess import preprocess_pipeline

    X_base, y, df, maps = preprocess_pipeline()
    df_eng = engineer_all_features(df)
    print("\nNew feature columns:")
    for col in get_engineered_feature_columns():
        print(f"  {col}: {df_eng[col].dtype}")
