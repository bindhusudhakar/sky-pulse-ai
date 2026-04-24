"""
pages/live_monitor.py — SkyPulse v2
=====================================
Live Flight Monitor — auto-refreshes every 30 seconds.
Fetches real flights from OpenSky Network, scores each one with
our trained Random Forest model, and displays risk rankings.
"""

import sys
import time
import datetime
import streamlit as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from live_feed    import fetch_live_flights
from live_predict import predict_live_flights
from constants    import (
    C, AIRLINES, AIRPORT_NAMES, WEATHER_EMOJIS, WEATHER_SEVERITY,
)

# ── Auto-refresh interval ─────────────────────────────────────────────────────
REFRESH_SECONDS = 30


# ── Small UI helpers ──────────────────────────────────────────────────────────

def _label(text):
    st.markdown(
        f'<p style="font-size:0.68rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.09em;color:{C["muted"]};margin:1.2rem 0 0.5rem;">'
        f'{text}</p>',
        unsafe_allow_html=True,
    )


def _badge(text, color, bg, size="0.72rem"):
    return (
        f'<span style="display:inline-block;padding:2px 10px;border-radius:99px;'
        f'font-size:{size};font-weight:700;background:{bg};color:{color};">'
        f'{text}</span>'
    )


def _insight(text):
    st.markdown(
        f'<div style="background:{C["primary_lt"]};border-left:3px solid {C["primary"]};'
        f'border-radius:0 10px 10px 0;padding:0.65rem 0.9rem;margin:0.5rem 0 0.8rem;'
        f'font-size:0.82rem;color:{C["text"]};">'
        f'<strong style="color:{C["primary"]};">ℹ️ </strong>{text}</div>',
        unsafe_allow_html=True,
    )


def _status_bar(feed: dict):
    """Top status bar: source, timestamp, flight count, countdown."""
    is_live  = feed.get("status") == "live"
    dot_col  = C["success"] if is_live else C["warning"]
    dot_text = "🟢 LIVE — OpenSky Network" if is_live else "🟡 DEMO — OpenSky unavailable"
    total    = feed.get("total_us_flights", 0)
    ts       = feed.get("timestamp", "—")

    st.markdown(
        f'<div style="background:{C["surface"]};border:1px solid {C["border"]};'
        f'border-radius:12px;padding:0.7rem 1.2rem;margin-bottom:1rem;'
        f'display:flex;align-items:center;gap:1.5rem;flex-wrap:wrap;">'
        f'<span style="font-size:0.82rem;font-weight:600;color:{dot_col};">{dot_text}</span>'
        f'<span style="font-size:0.8rem;color:{C["muted"]};">🕐 {ts}</span>'
        f'<span style="font-size:0.8rem;color:{C["muted"]};">✈ {total:,} flights tracked over US</span>'
        f'<span style="font-size:0.8rem;color:{C["primary"]};margin-left:auto;">'
        f'Auto-refresh every {REFRESH_SECONDS}s</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _kpi_strip(flights: list):
    """Summary metrics across all scored flights."""
    total     = len(flights)
    critical  = sum(1 for f in flights if f["risk_level"] == "Critical")
    high      = sum(1 for f in flights if f["risk_level"] == "High")
    medium    = sum(1 for f in flights if f["risk_level"] == "Medium")
    low       = sum(1 for f in flights if f["risk_level"] == "Low")
    avg_prob  = sum(f["delay_prob"] for f in flights) / max(total, 1)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Flights Monitored",  total)
    c2.metric("🔴 Critical Risk",    critical)
    c3.metric("🟠 High Risk",        high)
    c4.metric("🟡 Medium Risk",      medium)
    c5.metric("🟢 Low Risk",         low)
    c6.metric("Avg Delay Prob",     f"{avg_prob:.0%}")


def _risk_distribution(flights: list):
    """Horizontal stacked bar showing risk distribution."""
    total = max(len(flights), 1)
    counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    for f in flights:
        counts[f["risk_level"]] = counts.get(f["risk_level"], 0) + 1

    risk_colors = {
        "Critical": "#991B1B",
        "High":     "#DC2626",
        "Medium":   "#D97706",
        "Low":      "#16A34A",
    }

    bars = ""
    for level, color in risk_colors.items():
        pct = counts[level] / total * 100
        if pct > 0:
            bars += (
                f'<div style="width:{pct:.1f}%;background:{color};height:100%;'
                f'display:flex;align-items:center;justify-content:center;'
                f'font-size:0.68rem;font-weight:700;color:white;'
                f'min-width:30px;" title="{level}: {counts[level]} flights">'
                f'{counts[level]}</div>'
            )

    st.markdown(
        f'<div style="margin:0.5rem 0 1rem;">'
        f'<p style="font-size:0.72rem;color:{C["muted"]};margin-bottom:0.3rem;">'
        f'Risk distribution across {total} monitored flights:</p>'
        f'<div style="display:flex;height:28px;border-radius:8px;overflow:hidden;'
        f'border:1px solid {C["border"]};">{bars}</div>'
        f'<div style="display:flex;gap:1rem;margin-top:0.4rem;">'
        + "".join(
            f'<span style="font-size:0.7rem;color:{c};">■ {lv} ({counts[lv]})</span>'
            for lv, c in risk_colors.items()
        )
        + f'</div></div>',
        unsafe_allow_html=True,
    )


def _flight_card(f: dict, rank: int):
    """Render one flight as a styled card."""
    rc        = f["risk_color"]
    risk      = f["risk_level"]
    risk_bg   = {"Critical":"#FFD6D6","High":"#FFF1F1",
                 "Medium":"#FFFBEB","Low":"#F0FDF4"}.get(risk, C["bg"])

    airline_name = AIRLINES.get(f["airline"], f["airline"])
    o_name = AIRPORT_NAMES.get(f["origin"], f["origin"])
    d_name = AIRPORT_NAMES.get(f["dest"],   f["dest"])
    o_wx   = WEATHER_EMOJIS.get(f["origin_weather"], "")
    d_wx   = WEATHER_EMOJIS.get(f["dest_weather"],   "")

    # Delay probability bar
    prob_pct = f["delay_prob"] * 100
    bar_w    = int(prob_pct)

    st.markdown(
        f'<div style="background:{C["surface"]};border:1px solid {rc}44;'
        f'border-left:4px solid {rc};border-radius:12px;padding:0.9rem 1.1rem;'
        f'margin-bottom:0.6rem;">'

        # Row 1: rank + callsign + route + risk badge
        f'<div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.5rem;">'
        f'<span style="font-size:0.72rem;color:{C["muted"]};width:1.2rem;">#{rank}</span>'
        f'<span style="font-family:DM Mono,monospace;font-weight:700;'
        f'font-size:0.95rem;color:{C["text"]};">{f["callsign"]}</span>'
        f'<span style="font-size:0.85rem;color:{C["muted"]};">·</span>'
        f'<span style="font-size:0.85rem;color:{C["text"]};">'
        f'{f["origin"]} → {f["dest"]}</span>'
        f'<span style="font-size:0.8rem;color:{C["muted"]};flex:1;">{airline_name}</span>'
        f'<span style="display:inline-block;padding:2px 10px;border-radius:99px;'
        f'font-size:0.7rem;font-weight:700;background:{risk_bg};color:{rc};">'
        f'{risk} Risk</span>'
        f'</div>'

        # Row 2: origin/dest detail
        f'<div style="font-size:0.78rem;color:{C["muted"]};margin-bottom:0.5rem;">'
        f'{o_name} {o_wx}{f["origin_weather"]} → {d_name} {d_wx}{f["dest_weather"]} · '
        f'{f["distance"]:,.0f} km · Alt {f["altitude_ft"]:,} ft · {f["speed_knots"]} kts'
        f'</div>'

        # Row 3: probability bar
        f'<div style="display:flex;align-items:center;gap:0.6rem;">'
        f'<span style="font-size:0.72rem;color:{C["muted"]};width:9rem;">Delay probability</span>'
        f'<div style="flex:1;background:{C["border"]};border-radius:4px;height:7px;">'
        f'<div style="width:{bar_w}%;background:{rc};height:7px;border-radius:4px;'
        f'transition:width 0.3s;"></div></div>'
        f'<span style="font-family:DM Mono,monospace;font-size:0.82rem;'
        f'font-weight:700;color:{rc};width:4rem;text-align:right;">'
        f'{f["delay_pct"]}</span>'
        f'</div>'

        f'</div>',
        unsafe_allow_html=True,
    )


def _map_section(flights: list):
    """Simple text-based position summary since we can't use folium here."""
    critical = [f for f in flights if f["risk_level"] in ("Critical", "High")]
    if not critical:
        return

    _label("🗺 High-Risk Flight Positions")
    st.markdown(
        f'<p style="font-size:0.82rem;color:{C["muted"]};margin-bottom:0.6rem;">'
        f'Current positions of Critical and High risk flights:</p>',
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    for i, f in enumerate(critical[:9]):
        rc = f["risk_color"]
        with cols[i % 3]:
            st.markdown(
                f'<div style="background:{C["surface"]};border:1px solid {rc}55;'
                f'border-radius:10px;padding:0.6rem 0.8rem;margin-bottom:0.5rem;">'
                f'<div style="font-family:DM Mono,monospace;font-weight:700;'
                f'font-size:0.88rem;color:{C["text"]};">{f["callsign"]}</div>'
                f'<div style="font-size:0.75rem;color:{C["muted"]};">'
                f'{f["origin"]} → {f["dest"]}</div>'
                f'<div style="font-size:0.72rem;color:{C["muted"]};">'
                f'Lat {f["lat"]:.2f}° · Lon {f["lon"]:.2f}°</div>'
                f'<div style="font-size:0.75rem;color:{rc};font-weight:600;margin-top:3px;">'
                f'{f["delay_pct"]} delay risk</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── Main page render ──────────────────────────────────────────────────────────

def render():
    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="margin-bottom:0.5rem;">'
        f'<h2 style="font-size:1.5rem;font-weight:700;color:{C["text"]};margin:0 0 3px;">'
        f'📡 Live Flight Monitor</h2>'
        f'<p style="color:{C["muted"]};font-size:0.88rem;margin:0;">'
        f'Real-time flight risk scoring using OpenSky Network data. '
        f'Every flight is scored by our trained Random Forest model — '
        f'ranked from highest to lowest delay probability. '
        f'Auto-refreshes every <strong>30 seconds</strong>.'
        f'</p></div>',
        unsafe_allow_html=True,
    )

    # ── Controls row ──────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1, 1])
    with ctrl1:
        max_flights = st.selectbox(
            "Flights to monitor",
            [20, 30, 50, 75, 100],
            index=1,
            help="How many flights to fetch and score per refresh cycle",
        )
    with ctrl2:
        risk_filter = st.multiselect(
            "Show risk levels",
            ["Critical", "High", "Medium", "Low"],
            default=["Critical", "High", "Medium", "Low"],
            help="Filter the table to specific risk levels only",
        )
    with ctrl3:
        airline_filter = st.multiselect(
            "Filter airlines",
            list(AIRLINES.keys()),
            default=[],
            format_func=lambda x: f"{AIRLINES[x]} ({x})",
            help="Leave empty to show all airlines",
        )
    with ctrl4:
        st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
        manual_refresh = st.button("🔄 Refresh Now", key="manual_refresh")

    # ── Auto-refresh using st.rerun ───────────────────────────────────────────
    # Track last fetch time in session state
    now_ts = time.time()
    last_fetch = st.session_state.get("live_last_fetch", 0)
    cached     = st.session_state.get("live_feed_cache", None)

    needs_refresh = (
        manual_refresh
        or cached is None
        or (now_ts - last_fetch) >= REFRESH_SECONDS
    )

    if needs_refresh:
        with st.spinner("Fetching live flights from OpenSky…"):
            feed = fetch_live_flights(max_flights=max_flights)

        with st.spinner("Scoring flights with Random Forest model…"):
            enriched = predict_live_flights(feed["flights"])

        st.session_state["live_feed_cache"]  = feed
        st.session_state["live_enriched"]    = enriched
        st.session_state["live_last_fetch"]  = now_ts
    else:
        feed     = st.session_state["live_feed_cache"]
        enriched = st.session_state["live_enriched"]

    # ── Schedule next auto-refresh ────────────────────────────────────────────
    # Show countdown and trigger rerun after REFRESH_SECONDS
    elapsed = int(now_ts - st.session_state.get("live_last_fetch", now_ts))
    remaining = max(0, REFRESH_SECONDS - elapsed)

    # Status bar
    _status_bar(feed)

    # Countdown
    st.markdown(
        f'<div style="background:{C["bg"]};border:1px solid {C["border"]};'
        f'border-radius:8px;padding:0.4rem 0.8rem;margin-bottom:0.8rem;'
        f'display:inline-block;">'
        f'<span style="font-size:0.78rem;color:{C["muted"]};">'
        f'⏱ Next auto-refresh in <strong style="color:{C["primary"]};">'
        f'{remaining}s</strong></span></div>',
        unsafe_allow_html=True,
    )

    # ── Apply filters ─────────────────────────────────────────────────────────
    filtered = enriched
    if risk_filter:
        filtered = [f for f in filtered if f["risk_level"] in risk_filter]
    if airline_filter:
        filtered = [f for f in filtered if f["airline"] in airline_filter]

    if not filtered:
        st.info("No flights match the current filters. Try widening your selection.")
        # Still schedule refresh
        time.sleep(1)
        st.rerun()
        return

    # ── KPI strip ─────────────────────────────────────────────────────────────
    _label("📊 Current Snapshot")
    _kpi_strip(enriched)   # always show totals from unfiltered data

    # ── Risk distribution bar ─────────────────────────────────────────────────
    _risk_distribution(enriched)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        f"✈ Flight List ({len(filtered)})",
        "🗺 High-Risk Positions",
        "📖 How This Works",
    ])

    with tab1:
        _label(f"Flights Ranked by Delay Risk — Showing {len(filtered)} of {len(enriched)}")

        if feed.get("status") == "demo":
            st.warning(
                "⚠️ **Demo mode** — OpenSky Network is currently unreachable "
                "(this sandbox has no internet). The flights below are realistic "
                "synthetic data using current time, date, and weather estimation. "
                "When you run this locally, it will automatically switch to live data.",
                icon=None,
            )

        _insight(
            "Flights are ranked highest-to-lowest delay probability. "
            "Red = Critical (>75% chance of delay), Orange = High (55–75%), "
            "Yellow = Medium (30–55%), Green = Low (<30%). "
            "Click 'Refresh Now' to get the latest data instantly."
        )

        for i, flight in enumerate(filtered, 1):
            _flight_card(flight, i)

    with tab2:
        _map_section(enriched)
        _insight(
            "Positions shown are current aircraft locations from OpenSky telemetry. "
            "In demo mode, positions are estimated mid-route between origin and destination."
        )

    with tab3:
        st.markdown(
            f'<div style="max-width:700px;">'

            f'<h3 style="font-size:1rem;font-weight:700;color:{C["text"]};margin:1rem 0 0.5rem;">'
            f'How Live Monitoring Works</h3>'

            f'<div style="background:{C["surface"]};border:1px solid {C["border"]};'
            f'border-radius:12px;padding:1.2rem 1.4rem;margin-bottom:1rem;">'

            f'<p style="font-size:0.85rem;color:{C["text"]};margin-bottom:0.8rem;">'
            f'<strong>Step 1 — Data Fetch (OpenSky API)</strong><br>'
            f'Every 30 seconds, SkyPulse calls the OpenSky Network REST API '
            f'to retrieve all currently airborne flights over the continental US. '
            f'OpenSky is a crowdsourced ADS-B network — it receives transponder signals '
            f'from real aircraft and makes the data freely available. '
            f'No API key or account is required.</p>'

            f'<p style="font-size:0.85rem;color:{C["text"]};margin-bottom:0.8rem;">'
            f'<strong>Step 2 — Data Mapping</strong><br>'
            f'OpenSky provides raw transponder data: aircraft position, altitude, speed, '
            f'callsign, and heading. SkyPulse maps this to our model\'s expected inputs:<br>'
            f'• Callsign prefix (e.g. "UAL") → Airline code (UA = United)<br>'
            f'• GPS position → Nearest known airport (origin)<br>'
            f'• Heading + position → Estimated destination airport<br>'
            f'• Current time → Departure hour, day of week, month<br>'
            f'• Region + season → Estimated weather condition<br>'
            f'• Nearby flight count → Airport congestion estimate</p>'

            f'<p style="font-size:0.85rem;color:{C["text"]};margin-bottom:0.8rem;">'
            f'<strong>Step 3 — ML Scoring</strong><br>'
            f'Each flight is passed through the same 42-feature engineering pipeline '
            f'used during training, then scored by our Random Forest model. '
            f'The model outputs a delay probability (0–100%). Flights above 50% '
            f'are classified as "likely delayed".</p>'

            f'<p style="font-size:0.85rem;color:{C["text"]};margin:0;">'
            f'<strong>Important Limitation</strong><br>'
            f'Our model was trained on historical departure-time features. '
            f'These live flights are already airborne — so the probability represents '
            f'"would this flight have been predicted to delay at departure time?" '
            f'rather than a current en-route disruption forecast. '
            f'Weather is estimated, not real-time. For production use, '
            f'integrating a weather API (e.g. OpenWeatherMap) would improve accuracy.</p>'

            f'</div>'

            f'<h3 style="font-size:1rem;font-weight:700;color:{C["text"]};margin:1rem 0 0.5rem;">'
            f'API Details</h3>'

            f'<div style="background:{C["surface"]};border:1px solid {C["border"]};'
            f'border-radius:12px;padding:1.2rem 1.4rem;">'
            f'<p style="font-size:0.85rem;color:{C["text"]};margin:0;">'
            f'<strong>OpenSky Network</strong> — '
            f'<a href="https://opensky-network.org" style="color:{C["primary"]};">'
            f'opensky-network.org</a><br>'
            f'• Endpoint: <code>https://opensky-network.org/api/states/all</code><br>'
            f'• Parameters: bounding box covering continental US<br>'
            f'• Rate limit: 10 requests/minute (anonymous)<br>'
            f'• Data fields: ICAO24, callsign, position, altitude, velocity, heading<br>'
            f'• No API key required — completely free</p>'
            f'</div>'

            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Auto-rerun after delay ────────────────────────────────────────────────
    # Sleep 1 second then rerun — Streamlit will re-render and check
    # if REFRESH_SECONDS have elapsed
    time.sleep(1)
    st.rerun()
