"""
constants.py  —  SkyPulse v2
Single source of truth for ALL shared values.
Every import in every page file is satisfied here.
"""

# ── App identity ───────────────────────────────────────────────────────────────
APP_NAME    = "SkyPulse"
APP_VERSION = "2.0"
APP_TAGLINE = "Smart Aviation Traffic & Delay Prediction System"

# ── Navigation pages ──────────────────────────────────────────────────────────
# Each entry: (emoji, page_name, short_description)
NAV_PAGES = [
    ("📡", "Live Monitor",        "Real-time flights scored by AI — auto-refreshes every 30s"),
    ("🎯", "Delay Predictor",     "Predict if your flight will be delayed"),
    ("🌍", "Congestion Analysis", "Airport traffic density by hour and day"),
    ("🛫", "Route Analysis",      "Best and worst performing routes"),
    ("🌦", "Weather Impact",      "How weather conditions affect delays"),
    ("📊", "Model Performance",   "ML model accuracy and feature importance"),
    ("📋", "Data Explorer",       "Browse and filter the full flight dataset"),
]

# ── Airlines ──────────────────────────────────────────────────────────────────
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

# ── Airports ──────────────────────────────────────────────────────────────────
AIRPORT_NAMES = {
    "ATL": "Atlanta Hartsfield-Jackson",
    "LAX": "Los Angeles International",
    "ORD": "Chicago O'Hare International",
    "DFW": "Dallas/Fort Worth International",
    "DEN": "Denver International",
    "JFK": "New York John F. Kennedy",
    "SFO": "San Francisco International",
    "SEA": "Seattle-Tacoma International",
    "LAS": "Las Vegas Harry Reid",
    "MCO": "Orlando International",
    "MIA": "Miami International",
    "PHX": "Phoenix Sky Harbor",
    "BOS": "Boston Logan International",
    "MSP": "Minneapolis-Saint Paul",
    "DTW": "Detroit Metropolitan",
    "PHL": "Philadelphia International",
    "LGA": "New York LaGuardia",
    "BWI": "Baltimore/Washington International",
    "SLC": "Salt Lake City International",
    "CLT": "Charlotte Douglas International",
}
AIRPORTS = list(AIRPORT_NAMES.keys())

# ── Weather ───────────────────────────────────────────────────────────────────
WEATHER_OPTIONS = ["Clear", "Cloudy", "Rain", "Wind", "Fog", "Snow", "Thunderstorm"]

WEATHER_EMOJIS = {
    "Clear":        "☀️",
    "Cloudy":       "🌥️",
    "Rain":         "🌧️",
    "Wind":         "💨",
    "Fog":          "🌫️",
    "Snow":         "❄️",
    "Thunderstorm": "⛈️",
}

# Severity scale 0–6 (used by weather page + predict module)
WEATHER_SEVERITY = {
    "Clear": 0, "Cloudy": 1, "Rain": 2, "Wind": 3,
    "Fog": 4, "Snow": 5, "Thunderstorm": 6,
}

WEATHER_DESCRIPTIONS = {
    "Clear":        "Clear skies — minimal delay impact",
    "Cloudy":       "Overcast — slight delay risk",
    "Rain":         "Rain — moderate delay risk",
    "Wind":         "Strong winds — moderate delay risk",
    "Fog":          "Fog — significant visibility impact",
    "Snow":         "Snow — high delay risk",
    "Thunderstorm": "Thunderstorm — severe delay risk",
}

# ── Time helpers ──────────────────────────────────────────────────────────────
DAYS   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


def fmt_hour(h: int) -> str:
    """Convert 24-hour int to friendly 12-hour string.  17 → '5:00 PM'"""
    if h == 0:  return "12:00 AM (Midnight)"
    if h < 12:  return f"{h}:00 AM"
    if h == 12: return "12:00 PM (Noon)"
    return f"{h - 12}:00 PM"

# Alias so older code that still uses hour_label() also works
hour_label = fmt_hour

# ── UI colour palette (light theme) ──────────────────────────────────────────
# Exported as BOTH `C` and `COLORS` so any import name works.
C = {
    "primary":    "#2563EB",
    "primary_lt": "#EFF4FF",
    "success":    "#16A34A",
    "success_lt": "#F0FDF4",
    "danger":     "#DC2626",
    "danger_lt":  "#FFF1F1",
    "warning":    "#D97706",
    "warning_lt": "#FFFBEB",
    "purple":     "#7C3AED",
    "purple_lt":  "#F5F3FF",
    "neutral":    "#6B7280",
    "surface":    "#FFFFFF",
    "bg":         "#F7F8FA",
    "border":     "#E8EAF0",
    "text":       "#1A1D23",
    "muted":      "#6B7280",
    # aliases used by a few older page references
    "text_muted": "#6B7280",
}
COLORS = C  # alias — both names work
