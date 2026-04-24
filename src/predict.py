"""
src/predict.py
Inference module for flight delay prediction.
Loads saved model and returns delay prediction with probability and top factors.
"""

import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"

sys.path.insert(0, str(Path(__file__).parent))
from feature_engineering import (
    WEATHER_SEVERITY, AIRLINE_RELIABILITY, AIRPORT_TIER,
    engineer_all_features, get_engineered_feature_columns,
)


def load_artifacts():
    """Load model, scaler, and metadata."""
    metadata_path = MODEL_DIR / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            "Model not trained yet. Run: python src/train_model.py"
        )
    with open(metadata_path) as f:
        metadata = json.load(f)

    best_name = metadata["best_model_name"]
    filename_map = {
        "Random Forest": "random_forest.pkl",
        "Decision Tree": "decision_tree.pkl",
        "Logistic Regression": "logistic_regression.pkl",
    }

    model = joblib.load(MODEL_DIR / filename_map[best_name])
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    return model, scaler, metadata


def build_input_row(flight_params: dict, encoding_maps: dict) -> pd.DataFrame:
    """
    Convert user-supplied flight parameters into a feature DataFrame row.

    Parameters
    ----------
    flight_params : dict with keys:
        airline, origin, dest, dep_hour, month, day_of_week,
        distance, origin_weather, dest_weather, airport_congestion,
        aircraft_age, turnaround_time, maintenance_flag,
        carrier_delay_history, nas_delay
    encoding_maps : dict from metadata
    """
    row = flight_params.copy()

    # Encode categoricals
    for col in ["airline", "origin", "dest", "origin_weather", "dest_weather"]:
        mapping = encoding_maps.get(col, {})
        value = row.get(col, "")
        row[f"{col}_encoded"] = mapping.get(value, 0)

    df = pd.DataFrame([row])
    df = engineer_all_features(df)

    feature_cols = get_engineered_feature_columns()
    existing = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]

    for mc in missing:
        df[mc] = 0

    return df[feature_cols]


def predict_delay(flight_params: dict):
    """
    Predict whether a flight will be delayed.

    Returns
    -------
    dict with:
        delayed: bool
        delay_probability: float (0–1)
        risk_level: str ("Low" / "Medium" / "High" / "Critical")
        top_factors: list of (feature, value, description) tuples
        model_used: str
    """
    model, scaler, metadata = load_artifacts()
    encoding_maps = metadata["encoding_maps"]
    feature_cols = get_engineered_feature_columns()

    X = build_input_row(flight_params, encoding_maps)
    X = X.fillna(0)

    best_name = metadata["best_model_name"]
    if best_name == "Logistic Regression":
        X_model = scaler.transform(X)
    else:
        X_model = X.values

    prob = model.predict_proba(X_model)[0][1]
    delayed = prob >= 0.5

    # Risk level
    if prob < 0.3:
        risk_level = "Low"
    elif prob < 0.55:
        risk_level = "Medium"
    elif prob < 0.75:
        risk_level = "High"
    else:
        risk_level = "Critical"

    # Feature importance from metadata
    rf_result = next(
        (r for r in metadata["results"] if r["name"] == "Random Forest"), None
    )
    if rf_result:
        importance_map = rf_result["feature_importance"]
    else:
        importance_map = {}

    # Top contributing factors for this flight
    factors = []
    param = flight_params

    FACTOR_CHECKS = [
        ("origin_weather", lambda p: WEATHER_SEVERITY.get(p.get("origin_weather", "Clear"), 0) >= 4,
         f"Severe weather at origin: {flight_params.get('origin_weather')}"),
        ("airport_congestion", lambda p: p.get("airport_congestion", 0) > 80,
         f"High airport congestion: {flight_params.get('airport_congestion')} flights/hr"),
        ("maintenance_flag", lambda p: p.get("maintenance_flag", 0) == 1,
         "Active maintenance flag on aircraft"),
        ("dep_hour", lambda p: p.get("dep_hour", 12) >= 17,
         f"Evening departure (cascade delays risk): {flight_params.get('dep_hour')}:00"),
        ("aircraft_age", lambda p: p.get("aircraft_age", 0) > 15,
         f"Aging aircraft: {flight_params.get('aircraft_age')} years old"),
        ("turnaround_time", lambda p: p.get("turnaround_time", 45) < 30,
         f"Tight turnaround: {flight_params.get('turnaround_time'):.0f} minutes"),
        ("airline_reliability", lambda p: AIRLINE_RELIABILITY.get(p.get("airline", ""), 0.75) < 0.68,
         f"Low on-time carrier reliability: {param.get('airline')}"),
        ("nas_delay", lambda p: p.get("nas_delay", 0) > 10,
         f"Active NAS delay: {flight_params.get('nas_delay'):.0f} min"),
        ("dest_weather", lambda p: WEATHER_SEVERITY.get(p.get("dest_weather", "Clear"), 0) >= 3,
         f"Adverse destination weather: {flight_params.get('dest_weather')}"),
        ("day_of_week", lambda p: p.get("day_of_week", 0) in [4, 5],
         "Weekend high-traffic period"),
    ]

    for feat, check_fn, description in FACTOR_CHECKS:
        try:
            if check_fn(param):
                importance = importance_map.get(feat, 0.0)
                factors.append({
                    "feature": feat,
                    "description": description,
                    "importance": round(importance, 4),
                })
        except Exception:
            pass

    # Sort by importance
    factors.sort(key=lambda x: x["importance"], reverse=True)

    # If no flags triggered, add generic top features
    if not factors:
        top_feats = list(importance_map.keys())[:3]
        for f in top_feats:
            factors.append({"feature": f, "description": f.replace("_", " ").title(),
                           "importance": importance_map[f]})

    return {
        "delayed": bool(delayed),
        "delay_probability": round(float(prob), 4),
        "risk_level": risk_level,
        "top_factors": factors[:5],
        "model_used": best_name,
        "confidence": round(abs(prob - 0.5) * 2, 4),
    }


if __name__ == "__main__":
    # Sample prediction
    sample_flight = {
        "airline": "NK",
        "origin": "ORD",
        "dest": "JFK",
        "dep_hour": 18,
        "month": 1,
        "day_of_week": 4,
        "distance": 1200,
        "origin_weather": "Snow",
        "dest_weather": "Cloudy",
        "airport_congestion": 88,
        "aircraft_age": 18,
        "turnaround_time": 25,
        "maintenance_flag": 0,
        "carrier_delay_history": 22.0,
        "nas_delay": 5.0,
    }

    result = predict_delay(sample_flight)
    print("\n=== PREDICTION RESULT ===")
    print(f"  Delayed:     {result['delayed']}")
    print(f"  Probability: {result['delay_probability']:.1%}")
    print(f"  Risk Level:  {result['risk_level']}")
    print(f"  Model Used:  {result['model_used']}")
    print(f"\n  Top Factors:")
    for f in result["top_factors"]:
        print(f"    • {f['description']}")
