"""pages/weather.py - SkyPulse v2"""
import sys, streamlit as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from visualize import plot_weather_impact, plot_monthly_delay_trend
from constants import C, WEATHER_SEVERITY, WEATHER_EMOJIS

def _label(t):
    st.markdown(f'<p style="font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.09em;color:{C["muted"]};margin:1.4rem 0 0.5rem;">{t}</p>', unsafe_allow_html=True)

def _alert(kind, text):
    cfg = {"critical":(C["danger"],C["danger_lt"],"CRITICAL"),"warning":(C["warning"],C["warning_lt"],"WARNING"),"ok":(C["success"],C["success_lt"],"OK")}
    col,bg,icon = cfg[kind]
    st.markdown(f'<div style="background:{bg};border-left:3px solid {col};border-radius:0 10px 10px 0;padding:0.75rem 1rem;margin-bottom:8px;font-size:0.85rem;color:{C["text"]};"><strong>[{icon}]</strong> {text}</div>', unsafe_allow_html=True)

def _insight(text):
    st.markdown(f'<div style="background:{C["primary_lt"]};border-left:3px solid {C["primary"]};border-radius:0 10px 10px 0;padding:0.75rem 1rem;margin:0.5rem 0 1rem;font-size:0.83rem;color:{C["text"]};"><strong style="color:{C["primary"]};">Insight: </strong>{text}</div>', unsafe_allow_html=True)

def render(df, flight_params):
    st.markdown(f'<h2 style="font-size:1.6rem;font-weight:700;color:{C["text"]};margin:0 0 4px;">Weather Impact</h2><p style="color:{C["muted"]};font-size:0.92rem;margin:0 0 1.5rem;">How weather conditions drive flight delays — and a live assessment for your configured flight.</p>', unsafe_allow_html=True)
    tab1,tab2,tab3 = st.tabs(["Weather vs Delays","Monthly Trends","Live Assessment"])
    with tab1:
        _label("How It Works")
        st.markdown(f'<p style="font-size:0.83rem;color:{C["muted"]};margin-bottom:0.8rem;">Bars show average delay in minutes. Blue line shows delay rate percentage. Both increase with weather severity.</p>', unsafe_allow_html=True)
        st.plotly_chart(plot_weather_impact(df), use_container_width=True)
        _insight("Thunderstorms cause the longest and most frequent delays. Even cloudy conditions add measurable delay compared to clear skies due to pilot fuel buffer requests and ATC flow control.")
        _label("Severity Scale")
        sev_items = [("Clear","0","~50%",C["success_lt"],C["success"]),("Cloudy","1","~56%",C["success_lt"],C["success"]),("Rain","2","~65%",C["warning_lt"],C["warning"]),("Wind","3","~69%",C["warning_lt"],C["warning"]),("Fog","4","~72%",C["danger_lt"],C["danger"]),("Snow","5","~78%",C["danger_lt"],C["danger"]),("Thunderstorm","6","~84%",C["danger_lt"],C["danger"])]
        cols = st.columns(7)
        for col,(label,sev,rate,bg,fc) in zip(cols,sev_items):
            emoji = WEATHER_EMOJIS.get(label,"")
            with col:
                st.markdown(f'<div style="background:{bg};border:1px solid {fc}44;border-radius:10px;padding:0.65rem 0.4rem;text-align:center;"><div style="font-size:1.3rem;">{emoji}</div><div style="font-size:0.72rem;font-weight:700;color:{fc};margin:3px 0;">Sev {sev}</div><div style="font-size:0.65rem;color:{C["muted"]};">{rate}</div><div style="font-size:0.68rem;color:{C["text"]};margin-top:2px;">{label}</div></div>', unsafe_allow_html=True)
    with tab2:
        _label("How It Works")
        st.markdown(f'<p style="font-size:0.83rem;color:{C["muted"]};margin-bottom:0.8rem;">Delay rate percentage per calendar month. Summer and winter months are worst due to storms and snow/ice respectively.</p>', unsafe_allow_html=True)
        st.plotly_chart(plot_monthly_delay_trend(df), use_container_width=True)
        _insight("December and January have the highest delay rates due to winter storms at northern hubs. July and August spike from afternoon thunderstorms at southern hubs. April, May, and September are the most reliable months to fly.")
    with tab3:
        _label("Live Weather Assessment")
        st.markdown(f'<p style="font-size:0.83rem;color:{C["muted"]};margin-bottom:0.8rem;">Based on the weather you selected in the sidebar:</p>', unsafe_allow_html=True)
        ow=flight_params["origin_weather"]; dw=flight_params["dest_weather"]
        o_sev=WEATHER_SEVERITY.get(ow,0); d_sev=WEATHER_SEVERITY.get(dw,0)
        month=flight_params["month"]; orig=flight_params["origin"]
        if o_sev>=5: _alert("critical",f"Extreme origin weather: {WEATHER_EMOJIS[ow]} {ow} — ATC ground stop likely. Major delays expected.")
        elif o_sev>=4: _alert("critical",f"Severe origin weather: {WEATHER_EMOJIS[ow]} {ow} — Significant delays expected.")
        elif o_sev>=2: _alert("warning",f"Moderate origin weather: {WEATHER_EMOJIS[ow]} {ow} — Monitor for developing delays.")
        else: _alert("ok",f"Clear origin: {WEATHER_EMOJIS[ow]} {ow} — No weather impact expected.")
        if d_sev>=4: _alert("critical",f"Severe destination weather: {WEATHER_EMOJIS[dw]} {dw} — Holding patterns likely.")
        elif d_sev>=2: _alert("warning",f"Adverse destination weather: {WEATHER_EMOJIS[dw]} {dw} — Possible arrival delays.")
        else: _alert("ok",f"Clear destination: {WEATHER_EMOJIS[dw]} {dw} — No issues expected.")
        if month in [12,1,2] and orig in ["ORD","MSP","DTW","BOS","PHL","LGA"]:
            _alert("warning",f"Winter operations at {orig} — De-icing protocols may add 20-40 min.")
        if month in [6,7,8] and orig in ["MIA","MCO","ATL","DFW"]:
            _alert("warning",f"Summer convective season at {orig} — Afternoon thunderstorm risk elevated.")
        _label("Statistics by Weather Condition")
        stats = df.groupby("origin_weather").agg(Flights=("is_delayed","count"),Delay_Rate=("is_delayed","mean"),Avg_Min=("dep_delay_minutes","mean"),Max_Delay=("dep_delay_minutes","max")).reset_index().sort_values("Delay_Rate",ascending=False)
        stats["Delay Rate"]=(stats["Delay_Rate"]*100).round(1).astype(str)+"%"
        stats["Avg Delay"]=stats["Avg_Min"].round(0).astype(int).astype(str)+" min"
        stats["Max Delay"]=stats["Max_Delay"].round(0).astype(int).astype(str)+" min"
        stats=stats.rename(columns={"origin_weather":"Condition"})
        st.dataframe(stats[["Condition","Flights","Delay Rate","Avg Delay","Max Delay"]],use_container_width=True,hide_index=True)
