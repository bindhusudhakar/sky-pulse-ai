"""
generate_dataset.py
Generates a realistic synthetic aviation delay dataset (~10,000 flights)
mimicking Kaggle airline delay datasets with added features.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

AIRLINES = {
    "AA": "American Airlines",
    "DL": "Delta Air Lines",
    "UA": "United Airlines",
    "WN": "Southwest Airlines",
    "B6": "JetBlue Airways",
    "AS": "Alaska Airlines",
    "NK": "Spirit Airlines",
    "F9": "Frontier Airlines",
}

AIRPORTS = {
    "ATL": ("Atlanta", "GA", 33.64, -84.43),
    "LAX": ("Los Angeles", "CA", 33.94, -118.41),
    "ORD": ("Chicago O'Hare", "IL", 41.98, -87.90),
    "DFW": ("Dallas/Fort Worth", "TX", 32.90, -97.04),
    "DEN": ("Denver", "CO", 39.86, -104.67),
    "JFK": ("New York JFK", "NY", 40.64, -73.78),
    "SFO": ("San Francisco", "CA", 37.62, -122.38),
    "SEA": ("Seattle", "WA", 47.44, -122.31),
    "LAS": ("Las Vegas", "NV", 36.08, -115.15),
    "MCO": ("Orlando", "FL", 28.43, -81.31),
    "MIA": ("Miami", "FL", 25.79, -80.29),
    "PHX": ("Phoenix", "AZ", 33.43, -112.01),
    "BOS": ("Boston", "MA", 42.36, -71.01),
    "MSP": ("Minneapolis", "MN", 44.88, -93.22),
    "DTW": ("Detroit", "MI", 42.21, -83.35),
    "PHL": ("Philadelphia", "PA", 39.87, -75.24),
    "LGA": ("New York LaGuardia", "NY", 40.78, -73.87),
    "BWI": ("Baltimore", "MD", 39.18, -76.67),
    "SLC": ("Salt Lake City", "UT", 40.79, -111.98),
    "CLT": ("Charlotte", "NC", 35.21, -80.94),
}

WEATHER_CONDITIONS = ["Clear", "Cloudy", "Rain", "Snow", "Fog", "Thunderstorm", "Wind"]

WEATHER_DELAY_MULTIPLIER = {
    "Clear": 0.05,
    "Cloudy": 0.12,
    "Rain": 0.35,
    "Snow": 0.65,
    "Fog": 0.50,
    "Thunderstorm": 0.80,
    "Wind": 0.40,
}

AIRPORT_CONGESTION_BASE = {
    "ATL": 92, "LAX": 88, "ORD": 85, "DFW": 82, "DEN": 75,
    "JFK": 80, "SFO": 78, "SEA": 65, "LAS": 70, "MCO": 72,
    "MIA": 74, "PHX": 68, "BOS": 76, "MSP": 60, "DTW": 62,
    "PHL": 65, "LGA": 83, "BWI": 58, "SLC": 55, "CLT": 67,
}


def get_seasonal_weather(month, airport):
    """Simulate seasonal weather patterns per airport."""
    cold_airports = ["ORD", "MSP", "DTW", "BOS", "PHL", "LGA", "BWI"]
    fog_airports = ["SFO", "SEA", "LAX"]
    storm_airports = ["MIA", "MCO", "ATL", "DFW"]

    winter = month in [12, 1, 2]
    summer = month in [6, 7, 8]

    weights = {c: 1.0 for c in WEATHER_CONDITIONS}

    if winter and airport in cold_airports:
        weights["Snow"] = 8.0
        weights["Cloudy"] = 4.0
        weights["Clear"] = 0.5
    elif winter:
        weights["Rain"] = 3.0
        weights["Cloudy"] = 3.0

    if summer and airport in storm_airports:
        weights["Thunderstorm"] = 5.0
        weights["Rain"] = 3.0

    if airport in fog_airports:
        weights["Fog"] = 4.0

    if summer:
        weights["Clear"] = 3.0

    total = sum(weights.values())
    probs = [weights[c] / total for c in WEATHER_CONDITIONS]
    return np.random.choice(WEATHER_CONDITIONS, p=probs)


def generate_dataset(n=10000):
    records = []
    airport_list = list(AIRPORTS.keys())
    airline_list = list(AIRLINES.keys())

    start_date = datetime(2023, 1, 1)

    for i in range(n):
        # Flight metadata
        flight_date = start_date + timedelta(days=np.random.randint(0, 365))
        month = flight_date.month
        day_of_week = flight_date.weekday()  # 0=Mon, 6=Sun
        hour = np.random.choice(
            range(5, 23),
            p=np.array([2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 5, 5, 6, 7, 6, 4, 3]) / 97,
        )

        airline = np.random.choice(airline_list)
        origin = np.random.choice(airport_list)
        dest = np.random.choice([a for a in airport_list if a != origin])

        # Distance
        o_lat, o_lon = AIRPORTS[origin][2], AIRPORTS[origin][3]
        d_lat, d_lon = AIRPORTS[dest][2], AIRPORTS[dest][3]
        dist_km = np.sqrt((o_lat - d_lat) ** 2 + (o_lon - d_lon) ** 2) * 111
        distance = max(200, dist_km + np.random.normal(0, 50))

        # Weather
        origin_weather = get_seasonal_weather(month, origin)
        dest_weather = get_seasonal_weather(month, dest)

        # Congestion (flights per hour) - busier at peak hours
        base_cong = AIRPORT_CONGESTION_BASE[origin]
        peak_factor = 1.3 if hour in [7, 8, 17, 18, 19] else 0.8 if hour < 7 else 1.0
        congestion = int(base_cong * peak_factor * np.random.uniform(0.85, 1.15))

        # Aircraft age (years)
        aircraft_age = np.random.randint(1, 25)

        # Turnaround time (minutes)
        turnaround_time = np.random.normal(45, 15)
        turnaround_time = max(15, turnaround_time)

        # Maintenance flag
        maintenance_flag = int(np.random.random() < 0.08)

        # Carrier delay history (average delay for this airline)
        carrier_delay_history = np.random.normal(15, 10)

        # NAS delay (National Airspace System)
        nas_delay = np.random.exponential(5) if np.random.random() < 0.3 else 0

        # --- Delay Logic ---
        delay_prob = 0.15  # base 15%

        # Weather impact
        delay_prob += WEATHER_DELAY_MULTIPLIER[origin_weather] * 0.4
        delay_prob += WEATHER_DELAY_MULTIPLIER[dest_weather] * 0.2

        # Congestion impact
        if congestion > 85:
            delay_prob += 0.20
        elif congestion > 70:
            delay_prob += 0.10

        # Time of day - evening cascade delays
        if hour >= 17:
            delay_prob += 0.15

        # Weekend effect
        if day_of_week in [4, 5]:  # Fri, Sat
            delay_prob += 0.08

        # Airline reliability
        if airline in ["NK", "F9"]:
            delay_prob += 0.12
        elif airline in ["AS", "DL"]:
            delay_prob -= 0.05

        # Maintenance
        delay_prob += maintenance_flag * 0.25

        # Aircraft age
        if aircraft_age > 15:
            delay_prob += 0.08

        # Turnaround pressure
        if turnaround_time < 30:
            delay_prob += 0.10

        # Distance (longer = more exposure)
        if distance > 2000:
            delay_prob += 0.05

        delay_prob = np.clip(delay_prob, 0.02, 0.95)
        is_delayed = int(np.random.random() < delay_prob)

        # Delay minutes
        if is_delayed:
            delay_minutes = int(
                np.random.exponential(45)
                * (1 + WEATHER_DELAY_MULTIPLIER[origin_weather])
                * (1.5 if maintenance_flag else 1.0)
            )
            delay_minutes = max(15, delay_minutes)
        else:
            delay_minutes = np.random.randint(-5, 15)

        records.append(
            {
                "flight_date": flight_date.strftime("%Y-%m-%d"),
                "month": month,
                "day_of_week": day_of_week,
                "dep_hour": hour,
                "airline": airline,
                "airline_name": AIRLINES[airline],
                "origin": origin,
                "dest": dest,
                "origin_city": AIRPORTS[origin][0],
                "dest_city": AIRPORTS[dest][0],
                "distance": round(distance, 1),
                "origin_weather": origin_weather,
                "dest_weather": dest_weather,
                "airport_congestion": congestion,
                "aircraft_age": aircraft_age,
                "turnaround_time": round(turnaround_time, 1),
                "maintenance_flag": maintenance_flag,
                "carrier_delay_history": round(max(0, carrier_delay_history), 1),
                "nas_delay": round(nas_delay, 1),
                "dep_delay_minutes": delay_minutes,
                "is_delayed": is_delayed,
            }
        )

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    print("Generating dataset...")
    df = generate_dataset(10000)
    df.to_csv("flights.csv", index=False)
    print(f"Dataset saved: {len(df)} records")
    print(f"Delay rate: {df['is_delayed'].mean():.1%}")
    print(df.head())
