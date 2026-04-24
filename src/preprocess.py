"""
src/preprocess.py
Data loading, cleaning, and preprocessing for the aviation delay prediction system.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_FILE = DATA_DIR / "flights.csv"


def load_data(filepath=None):
    """Load flight data from CSV."""
    path = filepath or RAW_FILE
    df = pd.read_csv(path, parse_dates=["flight_date"])
    print(f"[Preprocess] Loaded {len(df):,} records from {path.name}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw flight data:
    - Drop duplicates
    - Handle missing values
    - Fix data types
    - Remove impossible values
    """
    initial_len = len(df)

    # Drop exact duplicates
    df = df.drop_duplicates()

    # Fill numeric NaNs with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  [Clean] Filled {col} NaNs with median={median_val:.2f}")

    # Fill categorical NaNs with mode
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isna().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"  [Clean] Filled {col} NaNs with mode={mode_val}")

    # Sanity bounds
    df = df[df["distance"] > 0]
    df = df[df["aircraft_age"] >= 0]
    df = df[df["turnaround_time"] > 0]
    df = df[df["airport_congestion"] > 0]

    # Ensure binary target
    df["is_delayed"] = df["is_delayed"].astype(int)

    cleaned_len = len(df)
    print(f"[Preprocess] Cleaned: {initial_len:,} → {cleaned_len:,} records "
          f"({initial_len - cleaned_len} removed)")
    return df.reset_index(drop=True)


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode categorical columns.
    Returns encoded dataframe and encoding maps.
    """
    encoding_maps = {}
    cat_cols = ["airline", "origin", "dest", "origin_weather", "dest_weather"]

    for col in cat_cols:
        if col in df.columns:
            categories = sorted(df[col].unique())
            mapping = {v: i for i, v in enumerate(categories)}
            df[f"{col}_encoded"] = df[col].map(mapping)
            encoding_maps[col] = mapping

    print(f"[Preprocess] Encoded {len(encoding_maps)} categorical columns")
    return df, encoding_maps


def get_feature_columns():
    """Return the list of feature columns used for ML."""
    return [
        "month",
        "day_of_week",
        "dep_hour",
        "distance",
        "airport_congestion",
        "aircraft_age",
        "turnaround_time",
        "maintenance_flag",
        "carrier_delay_history",
        "nas_delay",
        "airline_encoded",
        "origin_encoded",
        "dest_encoded",
        "origin_weather_encoded",
        "dest_weather_encoded",
    ]


def preprocess_pipeline(filepath=None):
    """Full preprocessing pipeline. Returns (X, y, df, encoding_maps)."""
    df = load_data(filepath)
    df = clean_data(df)
    df, encoding_maps = encode_categoricals(df)

    features = get_feature_columns()
    X = df[features]
    y = df["is_delayed"]

    print(f"[Preprocess] Feature matrix: {X.shape}, Target distribution:")
    print(f"  Not Delayed: {(y == 0).sum():,}  |  Delayed: {(y == 1).sum():,}")
    return X, y, df, encoding_maps


if __name__ == "__main__":
    X, y, df, maps = preprocess_pipeline()
    print("\nSample features:")
    print(X.head())
