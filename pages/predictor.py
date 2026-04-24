"""
pages/predictor.py — SkyPulse v2
Delay Predictor page. Compact layout — all key info visible without scrolling.
"""
import sys
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from predict import predict_delay
from constants import AIRLINES, AIRPORT_NAMES, WEATHER_EMOJIS, DAYS, MONTHS, C, fmt_hour


def _label(text):
    st.markdown(
        f'<p style="font-size:0.68rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.09em;color:{C["muted"]};margin:1rem 0 0.5rem;">'
        f'{text}</p>',
        unsafe_allow_html=True,
    )


def _row(label, val, mono=False):
    f = "font-family:'DM Mono',monospace;" if mono else ""
    return (
        f'<tr>'
        f'<td style="color:{C["muted"]};font-size:0.8rem;padding:5px 0;'
        f'width:40%;border-bottom:1px solid {C["border"]};">{label}</td>'
        f'<td style="color:{C["text"]};font-size:0.83rem;padding:5px 0;'
        f'{f}border-bottom:1px solid {C["border"]};">{val}</td>'
        f'</tr>'
    )


def _gauge(prob, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob,
        title={"text": "Delay Probability %",
               "font": {"color": C["muted"], "size": 11, "family": "DM Sans"}},
        number={"suffix": "%",
                "font": {"color": C["text"], "size": 26, "family": "DM Mono"}},
        gauge={
            "axis": {"range": [0, 100],
                     "tickfont": {"color": C["muted"], "size": 9}},
            "bar":  {"color": color, "thickness": 0.26},
            "bgcolor": C["bg"], "borderwidth": 0,
            "steps": [
                {"range": [0,  30], "color": "#F0FDF4"},
                {"range": [30, 55], "color": "#FFFBEB"},
                {"range": [55, 80], "color": "#FFF4F4"},
                {"range": [80,100], "color": "#FFE4E4"},
            ],
            "threshold": {"line": {"color": C["text"], "width": 2},
                          "thickness": 0.75, "value": 50},
        },
    ))
    fig.update_layout(
        height=210,
        paper_bgcolor=C["surface"],
        font={"color": C["text"], "family": "DM Sans"},
        margin=dict(t=40, b=5, l=15, r=15),
    )
    return fig


def render(flight_params, df):
    # ── Page header ───────────────────────────────────────────────────────────
    col_title, col_hint = st.columns([3, 1])
    with col_title:
        st.markdown(
            f'<h2 style="font-size:1.5rem;font-weight:700;color:{C["text"]};margin:0 0 2px;">'
            f'🎯 Delay Predictor</h2>'
            f'<p style="color:{C["muted"]};font-size:0.88rem;margin:0 0 1rem;">'
            f'Review your flight summary below, then click <strong>Run Prediction</strong>.</p>',
            unsafe_allow_html=True,
        )
    with col_hint:
        # Prominent sidebar hint
        st.markdown(
            f'<div style="background:{C["primary_lt"]};border:1px solid {C["primary"]}44;'
            f'border-radius:10px;padding:0.5rem 0.75rem;margin-top:0.2rem;text-align:center;">'
            f'<p style="font-size:0.75rem;color:{C["primary"]};margin:0;font-weight:600;">'
            f'◀ Change flight settings<br>in the left sidebar</p></div>',
            unsafe_allow_html=True,
        )

    # ── Main 3-column layout ──────────────────────────────────────────────────
    col_summary, col_result, col_factors = st.columns([1.1, 1.1, 0.9], gap="medium")

    p = flight_params

    # ── Column 1: Flight Summary ──────────────────────────────────────────────
    with col_summary:
        _label("✈ Flight Summary")
        maint_html = (
            f'<span style="color:{C["danger"]};font-weight:600;">⚠ Active</span>'
            if p["maintenance_flag"] else
            f'<span style="color:{C["success"]};font-weight:600;">✓ Clear</span>'
        )
        rows_html = (
            _row("Route", f'{p["origin"]} → {p["dest"]}', mono=True)
            + _row("From", AIRPORT_NAMES.get(p["origin"], p["origin"]))
            + _row("To",   AIRPORT_NAMES.get(p["dest"],   p["dest"]))
            + _row("Airline", AIRLINES.get(p["airline"], p["airline"]))
            + _row("Departure", f'{MONTHS[p["month"]-1]} · {fmt_hour(p["dep_hour"])}')
            + _row("Day", DAYS[p["day_of_week"]])
            + _row("Distance", f'{p["distance"]:,.0f} km')
            + _row("Origin Wx", f'{WEATHER_EMOJIS[p["origin_weather"]]} {p["origin_weather"]}')
            + _row("Dest Wx",   f'{WEATHER_EMOJIS[p["dest_weather"]]} {p["dest_weather"]}')
            + _row("Congestion", f'{p["airport_congestion"]} flights/hr')
            + _row("Aircraft Age", f'{p["aircraft_age"]} yrs')
            + _row("Turnaround", f'{int(p["turnaround_time"])} min')
            + _row("Maintenance", maint_html)
        )
        st.markdown(
            f'<div style="background:{C["surface"]};border:1px solid {C["border"]};'
            f'border-radius:14px;padding:1rem 1.2rem;">'
            f'<table style="width:100%;border-collapse:collapse;">'
            f'{rows_html}</table></div>',
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("🚀  Run Prediction", key="predict_btn"):
            with st.spinner("Running inference…"):
                result = predict_delay(flight_params)
            st.session_state["pred_result"] = result

    # ── Column 2: Prediction Result ───────────────────────────────────────────
    with col_result:
        _label("📊 Prediction Result")
        result = st.session_state.get("pred_result")

        if result:
            delayed  = result["delayed"]
            prob_pct = result["delay_probability"] * 100
            risk     = result["risk_level"]
            sc       = C["danger"] if delayed else C["success"]
            st_text  = "⚠ Likely Delayed" if delayed else "✓ On Time"
            risk_cfg = {
                "Low":      (C["success"], C["success_lt"]),
                "Medium":   (C["warning"], C["warning_lt"]),
                "High":     (C["danger"],  C["danger_lt"]),
                "Critical": (C["danger"],  "#FFD6D6"),
            }
            rc, rb = risk_cfg.get(risk, (C["muted"], C["bg"]))

            # Result card
            st.markdown(
                f'<div style="background:{C["surface"]};border:1px solid {sc}44;'
                f'border-top:4px solid {sc};border-radius:14px;'
                f'padding:1.2rem 1.4rem;text-align:center;margin-bottom:0.5rem;">'
                f'<p style="font-size:0.65rem;font-weight:700;text-transform:uppercase;'
                f'letter-spacing:0.1em;color:{C["muted"]};margin-bottom:0.3rem;">Verdict</p>'
                f'<p style="font-size:1.6rem;font-weight:700;color:{sc};'
                f'font-family:DM Mono,monospace;margin-bottom:0.2rem;">{st_text}</p>'
                f'<p style="font-size:0.88rem;color:{C["muted"]};margin-bottom:0.6rem;">'
                f'Delay probability: <strong style="color:{sc}">{prob_pct:.1f}%</strong></p>'
                f'<span style="display:inline-block;padding:3px 14px;border-radius:99px;'
                f'font-size:0.72rem;font-weight:700;background:{rb};color:{rc};">'
                f'{risk} Risk</span>'
                f'<br><span style="font-size:0.68rem;color:{C["muted"]};'
                f'display:block;margin-top:0.4rem;">Model: {result["model_used"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Gauge
            st.plotly_chart(_gauge(prob_pct, sc), use_container_width=True)

            # What this means
            st.markdown(
                f'<div style="background:{C["bg"]};border:1px solid {C["border"]};'
                f'border-radius:8px;padding:0.6rem 0.8rem;">'
                f'<p style="font-size:0.75rem;color:{C["muted"]};margin:0;">'
                f'<strong>Reading the gauge:</strong> Values above 50% = likely delayed. '
                f'Threshold line (|) marks the 50% decision boundary.</p></div>',
                unsafe_allow_html=True,
            )

        else:
            # Empty state — prominent call to action
            st.markdown(
                f'<div style="background:{C["surface"]};border:2px dashed {C["border"]};'
                f'border-radius:14px;padding:3rem 1.5rem;text-align:center;">'
                f'<div style="font-size:2.5rem;margin-bottom:0.5rem;">🛩</div>'
                f'<p style="color:{C["muted"]};font-size:0.9rem;margin:0 0 0.5rem;">'
                f'Click the button on the left to get a prediction.</p>'
                f'<p style="color:{C["primary"]};font-size:0.8rem;font-weight:600;margin:0;">'
                f'↙ Run Prediction</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Column 3: Factors + Stats ─────────────────────────────────────────────
    with col_factors:
        result = st.session_state.get("pred_result")

        if result and result.get("top_factors"):
            _label("⚡ Why This Prediction?")
            st.markdown(
                f'<p style="font-size:0.78rem;color:{C["muted"]};margin-bottom:0.5rem;">'
                f'Top risk factors detected:</p>',
                unsafe_allow_html=True,
            )
            for fac in result["top_factors"]:
                sc_fac = C["danger"] if result["delayed"] else C["warning"]
                st.markdown(
                    f'<div style="padding:7px 10px;background:{C["danger_lt"]};'
                    f'border-left:3px solid {sc_fac};border-radius:0 8px 8px 0;'
                    f'margin-bottom:5px;font-size:0.8rem;color:{C["text"]};">'
                    f'● {fac["description"]}</div>',
                    unsafe_allow_html=True,
                )
        else:
            _label("⚡ Risk Factors")
            st.markdown(
                f'<div style="background:{C["bg"]};border:1px solid {C["border"]};'
                f'border-radius:10px;padding:1.5rem;text-align:center;">'
                f'<p style="color:{C["muted"]};font-size:0.82rem;margin:0;">'
                f'Run a prediction to see which factors are contributing most to the result.</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Dataset stats — compact
        _label("📁 Dataset")
        total  = len(df)
        dn     = int(df["is_delayed"].sum())
        avg_d  = df[df["is_delayed"] == 1]["dep_delay_minutes"].mean()
        st.markdown(
            f'<div style="background:{C["surface"]};border:1px solid {C["border"]};'
            f'border-radius:12px;padding:0.8rem 1rem;">'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">'
            f'<div style="text-align:center;">'
            f'<div style="font-family:DM Mono,monospace;font-size:1.2rem;'
            f'font-weight:500;color:{C["text"]};">{total:,}</div>'
            f'<div style="font-size:0.65rem;color:{C["muted"]};text-transform:uppercase;'
            f'letter-spacing:0.06em;">Total flights</div></div>'
            f'<div style="text-align:center;">'
            f'<div style="font-family:DM Mono,monospace;font-size:1.2rem;'
            f'font-weight:500;color:{C["danger"]};">{dn/total:.0%}</div>'
            f'<div style="font-size:0.65rem;color:{C["muted"]};text-transform:uppercase;'
            f'letter-spacing:0.06em;">Delay rate</div></div>'
            f'<div style="text-align:center;">'
            f'<div style="font-family:DM Mono,monospace;font-size:1.2rem;'
            f'font-weight:500;color:{C["text"]};">{df["origin"].nunique()}</div>'
            f'<div style="font-size:0.65rem;color:{C["muted"]};text-transform:uppercase;'
            f'letter-spacing:0.06em;">Airports</div></div>'
            f'<div style="text-align:center;">'
            f'<div style="font-family:DM Mono,monospace;font-size:1.2rem;'
            f'font-weight:500;color:{C["warning"]};">{avg_d:.0f} min</div>'
            f'<div style="font-size:0.65rem;color:{C["muted"]};text-transform:uppercase;'
            f'letter-spacing:0.06em;">Avg delay</div></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        # Try me tip
        st.markdown(
            f'<div style="background:{C["primary_lt"]};border-left:3px solid {C["primary"]};'
            f'border-radius:0 8px 8px 0;padding:0.6rem 0.8rem;margin-top:0.8rem;">'
            f'<p style="font-size:0.75rem;color:{C["text"]};margin:0;">'
            f'💡 <strong>Try:</strong> Set Spirit (NK), ORD origin, Snow weather, '
            f'6 PM departure → Critical risk</p></div>',
            unsafe_allow_html=True,
        )
