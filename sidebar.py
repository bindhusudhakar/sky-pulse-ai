"""
sidebar.py - SkyPulse v2
All sidebar navigation + flight parameter controls.
Returns (selected_page_name, flight_params_dict).
"""
import streamlit as st
from constants import (
    APP_NAME, APP_TAGLINE, NAV_PAGES,
    AIRLINES, AIRPORTS, AIRPORT_NAMES,
    WEATHER_OPTIONS, WEATHER_EMOJIS,
    DAYS, MONTHS, C, fmt_hour,
)


def _divider(label):
    st.markdown(
        f'<p style="font-size:0.65rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.1em;color:{C["muted"]};margin:1.2rem 0 0.5rem;">'
        f'{label}</p>',
        unsafe_allow_html=True,
    )


def render():
    with st.sidebar:
        # Brand header
        st.markdown(
            f'<div style="padding:0.5rem 0 1rem;">'
            f'<div style="display:flex;align-items:center;gap:8px;">'
            f'<span style="font-size:1.5rem;">🛩</span>'
            f'<div>'
            f'<div style="font-size:1.2rem;font-weight:700;color:{C["text"]};'
            f'letter-spacing:-0.02em;">{APP_NAME}</div>'
            f'<div style="font-size:0.62rem;color:{C["muted"]};letter-spacing:0.03em;">'
            f'{APP_TAGLINE}</div>'
            f'</div></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<hr style="border:none;border-top:1.5px solid {C["border"]};margin:0 0 0.75rem;">',
            unsafe_allow_html=True,
        )

        # Navigation
        _divider("Navigate")
        labels = [f"{icon}  {name}" for icon, name, _ in NAV_PAGES]
        choice  = st.radio("nav", labels, label_visibility="collapsed")
        selected = choice.split("  ", 1)[1] if "  " in choice else choice

        # Show page description
        desc = next((d for _, n, d in NAV_PAGES if n == selected), "")
        if desc:
            st.markdown(
                f'<p style="font-size:0.78rem;color:{C["primary"]};'
                f'margin:-4px 0 0.5rem;font-style:italic;">{desc}</p>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<hr style="border:none;border-top:1px solid {C["border"]};margin:0.75rem 0;">',
            unsafe_allow_html=True,
        )

        # ── Flight Parameters ──────────────────────────────────────────────
        _divider("Flight Parameters")

        airline_code = st.selectbox(
            "Airline",
            list(AIRLINES.keys()),
            format_func=lambda x: f"{AIRLINES[x]} ({x})",
            help="The operating airline for this flight",
        )
        origin = st.selectbox(
            "Origin Airport",
            AIRPORTS, index=2,
            format_func=lambda x: f"{AIRPORT_NAMES[x]} ({x})",
            help="The airport the flight departs from",
        )
        dest_options = [a for a in AIRPORTS if a != origin]
        dest = st.selectbox(
            "Destination Airport",
            dest_options,
            index=min(5, len(dest_options) - 1),
            format_func=lambda x: f"{AIRPORT_NAMES[x]} ({x})",
            help="The airport the flight arrives at",
        )

        col1, col2 = st.columns(2)
        with col1:
            dep_hour = st.slider(
                "Departure Time",
                min_value=5, max_value=22, value=17,
                help="Hour of departure in 24-hour format (5=5AM, 17=5PM)",
            )
        with col2:
            month = st.selectbox(
                "Month", range(1, 13),
                format_func=lambda m: MONTHS[m - 1], index=11,
                help="Month of travel",
            )

        # Readable time display
        st.markdown(
            f'<p style="font-size:0.8rem;color:{C["primary"]};margin:-6px 0 8px;font-weight:500;">'
            f'Departure at {fmt_hour(dep_hour)}</p>',
            unsafe_allow_html=True,
        )

        day_of_week = st.selectbox(
            "Day of Week", range(7),
            format_func=lambda d: DAYS[d], index=4,
            help="Day of the week the flight operates",
        )
        origin_weather = st.selectbox(
            "Weather at Origin",
            WEATHER_OPTIONS,
            format_func=lambda w: f"{WEATHER_EMOJIS[w]} {w}",
            help="Weather conditions at the departure airport",
        )
        dest_weather = st.selectbox(
            "Weather at Destination",
            WEATHER_OPTIONS,
            format_func=lambda w: f"{WEATHER_EMOJIS[w]} {w}",
            help="Weather conditions at the arrival airport",
        )

        st.markdown(
            f'<hr style="border:none;border-top:1px solid {C["border"]};margin:0.75rem 0;">',
            unsafe_allow_html=True,
        )

        # ── Technical Details ──────────────────────────────────────────────
        _divider("Technical Details")

        distance = st.number_input(
            "Flight Distance (km)", 200, 6000, 1100, step=50,
            help="Approximate distance between origin and destination in kilometres",
        )
        congestion = st.slider(
            "Airport Congestion", 30, 100, 82,
            help="Number of flights per hour at the origin airport. Higher = more crowded = more delays.",
        )
        st.markdown(
            f'<p style="font-size:0.75rem;color:{C["muted"]};margin:-8px 0 8px;">'
            f'{congestion} flights/hr at {origin} '
            f'({"HIGH" if congestion > 80 else "MODERATE" if congestion > 65 else "LOW"} congestion)</p>',
            unsafe_allow_html=True,
        )

        aircraft_age = st.slider(
            "Aircraft Age (years)", 1, 25, 8,
            help="Age of the aircraft. Older aircraft carry higher maintenance risk.",
        )
        turnaround = st.slider(
            "Turnaround Time (min)", 15, 120, 45,
            help="Time between the previous flight landing and this departure. Under 30 min is risky.",
        )
        maintenance = st.checkbox(
            "Active Maintenance Flag",
            help="Check if there is a known maintenance issue flagged on this aircraft",
        )
        carrier_hist = st.slider(
            "Carrier Delay History (min)", 0.0, 50.0, 12.0, 0.5,
            help="Average delay minutes historically accumulated by this airline",
        )
        nas_delay = st.slider(
            "NAS Delay (min)", 0.0, 60.0, 0.0, 0.5,
            help="National Airspace System delay — caused by ATC, weather programs, or heavy traffic",
        )

    flight_params = {
        "airline":               airline_code,
        "origin":                origin,
        "dest":                  dest,
        "dep_hour":              dep_hour,
        "month":                 month,
        "day_of_week":           day_of_week,
        "distance":              float(distance),
        "origin_weather":        origin_weather,
        "dest_weather":          dest_weather,
        "airport_congestion":    congestion,
        "aircraft_age":          aircraft_age,
        "turnaround_time":       float(turnaround),
        "maintenance_flag":      int(maintenance),
        "carrier_delay_history": carrier_hist,
        "nas_delay":             nas_delay,
    }
    return selected, flight_params
