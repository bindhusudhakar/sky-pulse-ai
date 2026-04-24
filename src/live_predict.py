"""
src/live_predict.py
===================
Runs our trained Random Forest model on a list of live flights
returned by live_feed.fetch_live_flights().

Each flight dict is passed through the same feature-engineering
pipeline used during training, then scored. Returns an enriched
list ready to display in the Live Monitor page.
"""

import sys
import json
import math
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from feature_engineering import (
    WEATHER_SEVERITY,
    AIRLINE_RELIABILITY,
    AIRPORT_TIER,
)


# ── Risk level thresholds ─────────────────────────────────────────────────────
def _risk_level(prob: float) -> str:
    if prob < 0.30: return "Low"
    if prob < 0.55: return "Medium"
    if prob < 0.75: return "High"
    return "Critical"


def _risk_color(risk: str) -> str:
    return {
        "Low":      "#16A34A",
        "Medium":   "#D97706",
        "High":     "#DC2626",
        "Critical": "#991B1B",
    }.get(risk, "#6B7280")


# ── Load artifacts once ───────────────────────────────────────────────────────
_model    = None
_metadata = None
_enc_maps = None

def _load():
    global _model, _metadata, _enc_maps
    if _model is None:
        MODEL_DIR  = BASE_DIR / "models"
        _model     = joblib.load(MODEL_DIR / "best_model.pkl")
        with open(MODEL_DIR / "metadata.json") as f:
            _metadata  = json.load(f)
        _enc_maps  = _metadata["encoding_maps"]
        _feat_cols = _metadata["feature_columns"]
    return _model, _metadata, _enc_maps, _metadata["feature_columns"]


# ── Single-flight feature engineering ─────────────────────────────────────────
def _build_features(flight: Dict[str, Any],
                    enc_maps: Dict,
                    feature_cols: List[str]) -> np.ndarray:
    """
    Build the 42-feature vector for one flight dict.
    Mirrors feature_engineering.py but operates on a single dict.
    """
    hour  = flight["dep_hour"]
    month = flight["month"]
    dow   = flight["day_of_week"]
    dist  = flight["distance"]
    cong  = flight["airport_congestion"]
    age   = flight["aircraft_age"]
    tt    = flight["turnaround_time"]
    maint = flight["maintenance_flag"]
    ch    = flight["carrier_delay_history"]
    nas   = flight["nas_delay"]
    aln   = flight["airline"]
    orig  = flight["origin"]
    dest  = flight["dest"]
    o_wx  = flight["origin_weather"]
    d_wx  = flight["dest_weather"]

    # Encoded categoricals (default to 0 if unseen)
    aln_enc  = enc_maps.get("airline",       {}).get(aln,  0)
    orig_enc = enc_maps.get("origin",        {}).get(orig, 0)
    dest_enc = enc_maps.get("dest",          {}).get(dest, 0)
    owx_enc  = enc_maps.get("origin_weather",{}).get(o_wx, 0)
    dwx_enc  = enc_maps.get("dest_weather",  {}).get(d_wx, 0)

    # Time features
    is_peak_hour    = int(hour in [7, 8, 9, 17, 18, 19, 20])
    is_early_morning= int(hour in [5, 6])
    is_red_eye      = int(hour in [0, 1, 2, 3, 4])
    is_weekend      = int(dow >= 5)
    is_holiday_month= int(month in [6, 7, 8, 11, 12])
    hour_sin        = math.sin(2 * math.pi * hour  / 24)
    hour_cos        = math.cos(2 * math.pi * hour  / 24)
    month_sin       = math.sin(2 * math.pi * month / 12)
    month_cos       = math.cos(2 * math.pi * month / 12)

    # Weather features
    o_sev = WEATHER_SEVERITY.get(o_wx, 0)
    d_sev = WEATHER_SEVERITY.get(d_wx, 0)
    comb_wx    = o_sev * 0.7 + d_sev * 0.3
    severe_wx  = int(o_sev >= 4 or d_sev >= 4)
    wx_cong    = o_sev * cong / 100.0

    # Congestion tiers
    cong_tier  = 2 if cong > 80 else 1 if cong > 65 else 0
    high_cong  = int(cong > 80)

    # Airport tiers
    o_tier = AIRPORT_TIER.get(orig, 1)
    d_tier = AIRPORT_TIER.get(dest, 1)
    h2h    = int(o_tier == 3 and d_tier == 3)

    # Airline features
    reliability = AIRLINE_RELIABILITY.get(aln, 0.72)
    is_lcc      = int(aln in ["NK", "F9", "WN", "B6"])
    is_long     = int(dist > 2000)
    is_short    = int(dist < 800)

    # Operational
    tight_tt  = int(tt < 30)
    maint_age = maint * math.log1p(age)
    has_nas   = int(nas > 0)
    carr_risk = math.log1p(ch)

    # Build dict matching feature_columns order
    feature_dict = {
        "month":                  month,
        "day_of_week":            dow,
        "dep_hour":               hour,
        "distance":               dist,
        "airport_congestion":     cong,
        "aircraft_age":           age,
        "turnaround_time":        tt,
        "maintenance_flag":       maint,
        "carrier_delay_history":  ch,
        "nas_delay":              nas,
        "airline_encoded":        aln_enc,
        "origin_encoded":         orig_enc,
        "dest_encoded":           dest_enc,
        "origin_weather_encoded": owx_enc,
        "dest_weather_encoded":   dwx_enc,
        "is_peak_hour":           is_peak_hour,
        "is_early_morning":       is_early_morning,
        "is_red_eye":             is_red_eye,
        "is_weekend":             is_weekend,
        "is_holiday_month":       is_holiday_month,
        "hour_sin":               hour_sin,
        "hour_cos":               hour_cos,
        "month_sin":              month_sin,
        "month_cos":              month_cos,
        "origin_weather_severity":o_sev,
        "dest_weather_severity":  d_sev,
        "combined_weather_severity": comb_wx,
        "severe_weather":         severe_wx,
        "weather_congestion_risk":wx_cong,
        "congestion_tier":        cong_tier,
        "high_congestion":        high_cong,
        "origin_airport_tier":    o_tier,
        "dest_airport_tier":      d_tier,
        "hub_to_hub":             h2h,
        "airline_reliability":    reliability,
        "is_lcc":                 is_lcc,
        "is_long_haul":           is_long,
        "is_short_haul":          is_short,
        "tight_turnaround":       tight_tt,
        "maint_age_risk":         maint_age,
        "has_nas_delay":          has_nas,
        "carrier_risk":           carr_risk,
    }

    # Return as ordered numpy array matching model's expected feature order
    return np.array([feature_dict.get(col, 0) for col in feature_cols],
                    dtype=float)


# ── Batch prediction ──────────────────────────────────────────────────────────
def predict_live_flights(flights: List[Dict]) -> List[Dict]:
    """
    Score a list of flight dicts from live_feed.
    Returns the same list enriched with prediction fields:
      - delay_prob   : float 0–1
      - delay_pct    : str "43.2%"
      - risk_level   : "Low" | "Medium" | "High" | "Critical"
      - risk_color   : hex colour string
      - delayed      : bool
    """
    if not flights:
        return []

    model, metadata, enc_maps, feat_cols = _load()

    # Build feature matrix
    X = np.vstack([
        _build_features(f, enc_maps, feat_cols) for f in flights
    ])

    # Batch predict
    probs = model.predict_proba(X)[:, 1]

    enriched = []
    for flight, prob in zip(flights, probs):
        risk  = _risk_level(prob)
        fl    = dict(flight)
        fl["delay_prob"]  = round(float(prob), 4)
        fl["delay_pct"]   = f"{prob * 100:.1f}%"
        fl["risk_level"]  = risk
        fl["risk_color"]  = _risk_color(risk)
        fl["delayed"]     = prob >= 0.50
        enriched.append(fl)

    # Sort: highest risk first
    enriched.sort(key=lambda f: f["delay_prob"], reverse=True)
    return enriched
