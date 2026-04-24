"""
pages/routes.py - SkyPulse v2
"""
import sys, streamlit as st, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from visualize import plot_route_efficiency, plot_delay_by_airline
from constants import C

def _label(t):
    st.markdown(f'<p style="font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.09em;color:{C["muted"]};margin:1.4rem 0 0.5rem;">{t}</p>', unsafe_allow_html=True)

def _insight(text):
    st.markdown(f'<div style="background:{C["primary_lt"]};border-left:3px solid {C["primary"]};border-radius:0 10px 10px 0;padding:0.75rem 1rem;margin:0.5rem 0 1rem;font-size:0.83rem;color:{C["text"]};"><strong style="color:{C["primary"]};" >Insight: </strong>{text}</div>', unsafe_allow_html=True)

def render(df):
    st.markdown(f'<h2 style="font-size:1.6rem;font-weight:700;color:{C["text"]};margin:0 0 4px;">Route Analysis</h2><p style="color:{C["muted"]};font-size:0.92rem;margin:0 0 1.5rem;">Discover which routes and airlines are most reliable.</p>', unsafe_allow_html=True)
    tab1,tab2,tab3 = st.tabs(["Route Efficiency","Airline Rankings","Stats Table"])
    with tab1:
        _label("How It Works")
        st.markdown(f'<p style="font-size:0.83rem;color:{C["muted"]};margin-bottom:0.8rem;">Green bars = low delay routes. Red bars = high delay routes. Only routes with 20+ flights shown.</p>', unsafe_allow_html=True)
        c1,c2 = st.columns([5,1])
        with c2: top_n = st.slider("Routes",8,20,12,step=2)
        with c1: st.plotly_chart(plot_route_efficiency(df,top_n),use_container_width=True)
        _insight("Hub-to-hub routes tend to have higher delay rates due to compounding congestion at both ends.")
    with tab2:
        _label("How It Works")
        st.markdown(f'<p style="font-size:0.83rem;color:{C["muted"]};margin-bottom:0.8rem;">Airlines ranked by overall delay rate. Green below 58%, amber 58-70%, red above 70%.</p>', unsafe_allow_html=True)
        st.plotly_chart(plot_delay_by_airline(df),use_container_width=True)
        _insight("Low-cost carriers show higher delay rates due to tighter turnarounds and smaller operational buffers.")
    with tab3:
        _label("Route Statistics Table")
        st.markdown(f'<p style="font-size:0.83rem;color:{C["muted"]};margin-bottom:0.8rem;">All routes with 15+ flights, sorted by delay rate (worst first).</p>', unsafe_allow_html=True)
        stats = df.groupby(["origin","dest"]).agg(Flights=("is_delayed","count"),Delay_Rate=("is_delayed","mean"),Avg_Min=("dep_delay_minutes","mean")).reset_index()
        stats = stats[stats["Flights"]>=15].copy()
        stats["Route"] = stats["origin"] + " to " + stats["dest"]
        stats["Delay Rate"] = (stats["Delay_Rate"]*100).round(1).astype(str)+"%"
        stats["Avg Delay"]  = stats["Avg_Min"].round(0).astype(int).astype(str)+" min"
        disp = stats[["Route","Flights","Delay Rate","Avg Delay"]].sort_values("Delay_Rate",ascending=False).head(30).reset_index(drop=True)
        st.dataframe(disp,use_container_width=True,height=400)
