"""
src/live_feed.py
================
Fetches real-time flight data from the OpenSky Network REST API.
No API key required. Completely free.

OpenSky API docs: https://openskynetwork.github.io/opensky-api/rest.html

Mapping strategy:
  - Callsign prefix (e.g. "UAL" → "UA") maps to our airline codes
  - Position (lat/lon) maps to nearest known airport
  - Time-of-day, date come from system clock
  - Weather is estimated from season + region (OpenSky has no weather)
  - Congestion is estimated from number of nearby flights in the API response
"""

import math
import time
import datetime
import urllib.request
import urllib.error
import json
import random
from typing import Optional

# ── OpenSky bounding box: continental US ──────────────────────────────────────
OPENSKY_URL = (
    "https://opensky-network.org/api/states/all"
    "?lamin=24.0&lomin=-125.0&lamax=49.0&lomax=-66.0"
)

# ── ICAO callsign prefix → our airline code mapping ──────────────────────────
CALLSIGN_TO_AIRLINE = {
    # American
    "AAL": "AA", "AA":  "AA",
    # Delta
    "DAL": "DL", "DL":  "DL",
    # United
    "UAL": "UA", "UA":  "UA",
    # Southwest
    "SWA": "WN", "WN":  "WN",
    # JetBlue
    "JBU": "B6", "B6":  "B6",
    # Alaska
    "ASA": "AS", "AS":  "AS",
    # Spirit
    "NKS": "NK", "NK":  "NK",
    # Frontier
    "FFT": "F9", "F9":  "F9",
}

# ── Known airport positions (lat, lon) ────────────────────────────────────────
AIRPORT_COORDS = {
    "ATL": (33.6407, -84.4277),
    "LAX": (33.9425, -118.4081),
    "ORD": (41.9742, -87.9073),
    "DFW": (32.8998, -97.0403),
    "DEN": (39.8561, -104.6737),
    "JFK": (40.6413, -73.7781),
    "SFO": (37.6213, -122.3790),
    "SEA": (47.4502, -122.3088),
    "LAS": (36.0840, -115.1537),
    "MCO": (28.4312, -81.3081),
    "MIA": (25.7959, -80.2870),
    "PHX": (33.4373, -112.0078),
    "BOS": (42.3656, -71.0096),
    "MSP": (44.8848, -93.2223),
    "DTW": (42.2162, -83.3554),
    "PHL": (39.8744, -75.2424),
    "LGA": (40.7772, -73.8726),
    "BWI": (39.1754, -76.6683),
    "SLC": (40.7884, -111.9778),
    "CLT": (35.2140, -80.9431),
}

# ── Regional weather estimation (by lat/lon + month) ─────────────────────────
def _estimate_weather(lat: float, lon: float, month: int) -> str:
    """
    Rough weather estimate based on geography and season.
    OpenSky has no weather data, so we approximate:
    - Winter months + northern latitudes → Snow/Cloudy
    - Summer south → Thunderstorm risk
    - West coast → Cloudy/Clear
    - Always probabilistic to mimic real variety
    """
    is_winter  = month in [12, 1, 2]
    is_summer  = month in [6, 7, 8]
    is_north   = lat > 41           # Chicago / NY and above
    is_south   = lat < 33           # Miami / Phoenix level
    is_coast_w = lon < -115         # West coast

    rng = random.random()

    if is_winter and is_north:
        # Northern winter: high chance of snow/fog
        if rng < 0.30:   return "Snow"
        elif rng < 0.55: return "Cloudy"
        elif rng < 0.70: return "Fog"
        elif rng < 0.85: return "Rain"
        else:            return "Clear"

    elif is_summer and is_south:
        # Southern summer: thunderstorm season
        if rng < 0.25:   return "Thunderstorm"
        elif rng < 0.45: return "Rain"
        elif rng < 0.65: return "Cloudy"
        else:            return "Clear"

    elif is_coast_w:
        # West coast: generally mild
        if rng < 0.40:   return "Clear"
        elif rng < 0.70: return "Cloudy"
        elif rng < 0.85: return "Fog"
        else:            return "Rain"

    else:
        # Default mid-country / spring-fall
        if rng < 0.45:   return "Clear"
        elif rng < 0.65: return "Cloudy"
        elif rng < 0.80: return "Rain"
        elif rng < 0.90: return "Wind"
        else:            return "Fog"


def _haversine(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km between two lat/lon points."""
    R = 6371
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(d_lon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def _nearest_airport(lat: float, lon: float) -> Optional[str]:
    """Return the IATA code of the nearest known airport within 200 km."""
    best_code, best_dist = None, float("inf")
    for code, (a_lat, a_lon) in AIRPORT_COORDS.items():
        d = _haversine(lat, lon, a_lat, a_lon)
        if d < best_dist:
            best_dist, best_code = d, code
    # Only claim it if we're within 200 km
    return best_code if best_dist < 200 else None


def _parse_airline(callsign: str) -> str:
    """Extract airline code from ICAO callsign. Returns 'XX' if unknown."""
    if not callsign:
        return "XX"
    cs = callsign.strip().upper()
    # Try 3-letter ICAO prefix first
    if cs[:3] in CALLSIGN_TO_AIRLINE:
        return CALLSIGN_TO_AIRLINE[cs[:3]]
    # Try 2-letter IATA prefix
    if cs[:2] in CALLSIGN_TO_AIRLINE:
        return CALLSIGN_TO_AIRLINE[cs[:2]]
    return "XX"


def _count_nearby(states: list, lat: float, lon: float,
                  radius_deg: float = 2.0) -> int:
    """Count flights within ~200 km as a congestion proxy."""
    return sum(
        1 for s in states
        if s[6] is not None and s[5] is not None
        and abs(s[6] - lat) < radius_deg
        and abs(s[5] - lon) < radius_deg
    )


def fetch_live_flights(max_flights: int = 50) -> dict:
    """
    Fetch current flights from OpenSky and transform into
    the format our prediction pipeline expects.

    Returns:
        {
            "flights": [
                {
                    "callsign":     "UAL123",
                    "airline":      "UA",
                    "origin":       "ORD",    # nearest airport
                    "dest":         "LAX",    # estimated
                    "dep_hour":     17,
                    "month":        4,
                    "day_of_week":  4,
                    "distance":     2800.0,
                    "airport_congestion": 72,
                    "aircraft_age": 8,        # estimated average
                    "turnaround_time": 45.0,
                    "maintenance_flag": 0,
                    "carrier_delay_history": 12.0,
                    "nas_delay":    0.0,
                    "origin_weather": "Clear",
                    "dest_weather":   "Cloudy",
                    "altitude_ft":  35000,
                    "speed_knots":  480,
                    "lat":          41.97,
                    "lon":          -87.90,
                },
                ...
            ],
            "timestamp":    "2024-04-23 17:32:01 UTC",
            "total_us_flights": 4821,
            "source":       "OpenSky Network",
            "status":       "live" | "demo",
        }
    """
    now = datetime.datetime.utcnow()
    month      = now.month
    day_of_week = now.weekday()   # 0=Monday
    dep_hour   = now.hour

    # ── Try OpenSky API ───────────────────────────────────────────────────────
    states = []
    status = "demo"
    try:
        req = urllib.request.Request(
            OPENSKY_URL,
            headers={"User-Agent": "SkyPulse/2.0 (academic project)"},
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            raw = json.loads(resp.read())
            states = raw.get("states") or []
            status = "live"
    except Exception:
        # Fall back to synthetic demo data so the page always works
        states = []
        status = "demo"

    total_us = len(states)

    # ── If live data came back, process it ───────────────────────────────────
    flights = []

    if status == "live" and states:
        # Filter: airborne only, US callsigns, valid position
        airborne = [
            s for s in states
            if s[8] is False                # not on ground
            and s[1] and s[1].strip()       # has callsign
            and s[5] is not None            # has longitude
            and s[6] is not None            # has latitude
            and s[7] is not None            # has altitude > 0
            and (s[7] or 0) > 1000          # actually flying
        ]

        # Sort by altitude descending (cruise flights most interesting)
        airborne.sort(key=lambda s: s[7] or 0, reverse=True)

        for s in airborne[:max_flights]:
            icao24, callsign, country = s[0], s[1], s[2]
            lon, lat = s[5], s[6]
            alt_m    = s[7] or 0
            on_ground = s[8]
            vel_ms   = s[9] or 0
            heading  = s[10] or 0

            airline = _parse_airline(callsign)
            if airline == "XX":
                continue   # skip unknown / non-US airlines

            # Nearest airport = likely origin or nearby hub
            origin = _nearest_airport(lat, lon)
            if not origin:
                # Assign nearest airport even if > 200 km when we're en-route
                origin = min(
                    AIRPORT_COORDS.keys(),
                    key=lambda c: _haversine(lat, lon, *AIRPORT_COORDS[c]),
                )

            # Estimate destination: airport in rough heading direction
            # Simple heuristic: pick an airport ≥ 300 km away in heading quadrant
            heading_rad = math.radians(heading)
            dest_candidates = [
                (code, _haversine(lat, lon, a_lat, a_lon))
                for code, (a_lat, a_lon) in AIRPORT_COORDS.items()
                if code != origin
            ]
            # Filter to airports that are at least 300 km away
            far_enough = [(c, d) for c, d in dest_candidates if d > 300]
            if far_enough:
                # Pick the one most aligned with heading
                def heading_score(item):
                    code, _ = item
                    a_lat, a_lon = AIRPORT_COORDS[code]
                    angle = math.degrees(math.atan2(a_lon - lon, a_lat - lat)) % 360
                    diff  = abs(angle - heading) % 360
                    return min(diff, 360 - diff)
                dest = min(far_enough, key=heading_score)[0]
            else:
                dest = random.choice([c for c in AIRPORT_COORDS if c != origin])

            # Distance origin → dest
            distance = _haversine(*AIRPORT_COORDS[origin], *AIRPORT_COORDS[dest])

            # Congestion: count nearby airborne flights as proxy
            congestion = min(30 + _count_nearby(states, lat, lon) * 3, 100)

            # Weather estimate from position + season
            o_weather = _estimate_weather(lat, lon, month)
            d_lat, d_lon = AIRPORT_COORDS[dest]
            d_weather = _estimate_weather(d_lat, d_lon, month)

            # Convert units
            alt_ft      = int(alt_m * 3.28084)
            speed_knots = int(vel_ms * 1.94384)

            flights.append({
                "callsign":              callsign.strip(),
                "airline":               airline,
                "origin":                origin,
                "dest":                  dest,
                "dep_hour":              dep_hour,
                "month":                 month,
                "day_of_week":           day_of_week,
                "distance":              round(distance, 1),
                "airport_congestion":    congestion,
                "aircraft_age":          random.randint(4, 18),   # estimated
                "turnaround_time":       float(random.randint(30, 75)),
                "maintenance_flag":      0,
                "carrier_delay_history": _carrier_history(airline),
                "nas_delay":             0.0,
                "origin_weather":        o_weather,
                "dest_weather":          d_weather,
                "altitude_ft":           alt_ft,
                "speed_knots":           speed_knots,
                "lat":                   round(lat, 4),
                "lon":                   round(lon, 4),
                "icao24":                icao24,
            })

    # ── Demo fallback (always produces realistic synthetic data) ──────────────
    if not flights:
        flights = _generate_demo_flights(
            month, day_of_week, dep_hour, max_flights
        )
        total_us = len(flights)

    return {
        "flights":          flights,
        "timestamp":        now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "total_us_flights": total_us,
        "source":           "OpenSky Network",
        "status":           status,
    }


def _carrier_history(airline: str) -> float:
    """Historical average carrier delay by airline (minutes)."""
    history = {
        "AA": 14.2, "DL": 9.8,  "UA": 13.5,
        "WN": 11.1, "B6": 16.4, "AS": 7.2,
        "NK": 22.8, "F9": 19.3, "XX": 15.0,
    }
    return history.get(airline, 12.0)


def _generate_demo_flights(month: int, dow: int,
                            hour: int, n: int) -> list:
    """
    Generate realistic synthetic live flights when OpenSky is unreachable.
    Uses real airline + route combinations with sensible randomisation.
    """
    from constants import AIRPORTS, AIRLINES, WEATHER_OPTIONS
    import random

    airline_codes = list(AIRLINES.keys())
    weather_opts  = WEATHER_OPTIONS

    flights = []
    for i in range(n):
        airline  = random.choice(airline_codes)
        origin   = random.choice(AIRPORTS)
        dest     = random.choice([a for a in AIRPORTS if a != origin])
        distance = _haversine(*AIRPORT_COORDS[origin], *AIRPORT_COORDS[dest])

        o_lat, o_lon = AIRPORT_COORDS[origin]
        d_lat, d_lon = AIRPORT_COORDS[dest]

        # Simulate en-route position (somewhere between origin and dest)
        frac = random.uniform(0.1, 0.9)
        cur_lat = o_lat + frac * (d_lat - o_lat)
        cur_lon = o_lon + frac * (d_lon - o_lon)

        o_wx = _estimate_weather(o_lat, o_lon, month)
        d_wx = _estimate_weather(d_lat, d_lon, month)

        fake_call = f"{airline}{random.randint(100,999)}"

        flights.append({
            "callsign":              fake_call,
            "airline":               airline,
            "origin":                origin,
            "dest":                  dest,
            "dep_hour":              (hour - random.randint(0,3)) % 24,
            "month":                 month,
            "day_of_week":           dow,
            "distance":              round(distance, 1),
            "airport_congestion":    random.randint(40, 95),
            "aircraft_age":          random.randint(3, 20),
            "turnaround_time":       float(random.randint(25, 90)),
            "maintenance_flag":      1 if random.random() < 0.05 else 0,
            "carrier_delay_history": _carrier_history(airline),
            "nas_delay":             random.uniform(0, 15),
            "origin_weather":        o_wx,
            "dest_weather":          d_wx,
            "altitude_ft":           random.randint(28000, 38000),
            "speed_knots":           random.randint(420, 510),
            "lat":                   round(cur_lat, 4),
            "lon":                   round(cur_lon, 4),
            "icao24":                f"demo{i:04d}",
        })
    return flights
