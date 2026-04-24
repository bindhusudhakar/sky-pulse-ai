"""
pages/explorer.py - SkyPulse Data Explorer page.
"""
import streamlit as st
from constants import COLORS

def render(df):
    st.markdown(f'<h2 style="font-size:1.55rem;font-weight:700;color:{COLORS["text"]};margin-bottom:0.2rem;">📋 Data Explorer</h2><p style="color:{COLORS["text_muted"]};font-size:0.9rem;margin-bottom:1.4rem;">Browse the full flight dataset. Filter by airline, weather, or delay status. Download filtered results as CSV.</p>',unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    with c1:
        f_airline = st.multiselect("Filter by Airline", sorted(df["airline"].unique()), default=list(df["airline"].unique()))
    with c2:
        f_weather = st.multiselect("Filter by Origin Weather", sorted(df["origin_weather"].unique()), default=list(df["origin_weather"].unique()))
    with c3:
        f_status  = st.selectbox("Flight Status", ["All Flights","Delayed Only","On-Time Only"])

    filtered = df[df["airline"].isin(f_airline) & df["origin_weather"].isin(f_weather)]
    if f_status == "Delayed Only":   filtered = filtered[filtered["is_delayed"]==1]
    elif f_status == "On-Time Only": filtered = filtered[filtered["is_delayed"]==0]

    total_f   = len(filtered)
    delayed_f = int(filtered["is_delayed"].sum())

    c1b,c2b,c3b,c4b = st.columns(4)
    c1b.metric("Matching Records", f"{total_f:,}")
    c2b.metric("Delayed",          f"{delayed_f:,}")
    c3b.metric("On-Time",          f"{total_f-delayed_f:,}")
    if total_f > 0:
        c4b.metric("Delay Rate", f"{delayed_f/total_f:.1%}")

    display_cols = ["flight_date","airline_name","origin","dest","dep_hour","origin_weather","airport_congestion","aircraft_age","dep_delay_minutes","is_delayed"]
    existing = [c for c in display_cols if c in filtered.columns]
    st.dataframe(filtered[existing].head(500), use_container_width=True, height=440)
    st.caption("Showing up to 500 records. Download below for full results.")

    csv = filtered[existing].to_csv(index=False)
    st.download_button("⬇ Export as CSV", data=csv, file_name="skypulse_flights.csv", mime="text/csv")
