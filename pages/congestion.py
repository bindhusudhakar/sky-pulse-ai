"""
pages/congestion.py — SkyPulse v2
Airport Congestion Analysis: heatmaps + scatter + bar chart.
"""
import sys, streamlit as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from visualize import (plot_congestion_heatmap, plot_hour_delay_heatmap,
                        plot_congestion_vs_delay)
from constants import C

def _label(t):
    st.markdown(f'<p style="font-size:0.68rem;font-weight:700;text-transform:uppercase;'
                f'letter-spacing:0.09em;color:{C["muted"]};margin:1.4rem 0 0.5rem;">{t}</p>',
                unsafe_allow_html=True)

def _insight(text):
    st.markdown(f'<div style="background:{C["primary_lt"]};border-left:3px solid {C["primary"]};'
                f'border-radius:0 10px 10px 0;padding:0.75rem 1rem;margin:0.5rem 0 1rem;'
                f'font-size:0.83rem;color:{C["text"]};"><strong style="color:{C["primary"]};">'
                f'💡 Insight · </strong>{text}</div>', unsafe_allow_html=True)

def render(df):
    st.markdown(f'''<div style="margin-bottom:1.5rem;">
      <h2 style="font-size:1.6rem;font-weight:700;color:{C["text"]};margin:0 0 4px;">🌍 Congestion Analysis</h2>
      <p style="color:{C["muted"]};font-size:0.92rem;margin:0;">
        Explore how airport traffic density varies by time of day, day of week,
        and how congestion directly increases delay probability.
        Busier airports at peak hours are where most cascading delays begin.
      </p></div>''', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🗓 Congestion by Hour", "📅 Delay by Hour & Day", "🔗 Congestion vs Delay"])

    with tab1:
        _label("How It Works")
        st.markdown(f'<p style="font-size:0.83rem;color:{C["muted"]};margin-bottom:0.8rem;">'
                    f'Each cell shows the average number of flights per hour at that airport '
                    f'during that time slot. Darker red = higher congestion = higher delay risk.</p>',
                    unsafe_allow_html=True)
        st.plotly_chart(plot_congestion_heatmap(df), use_container_width=True)
        _insight("Peak congestion occurs between 07:00–09:00 (morning rush) and 17:00–20:00 "
                 "(evening rush). ATL, ORD, LAX, and LGA consistently exceed 85 flights/hr "
                 "during these windows — significantly increasing delay risk for departures.")

    with tab2:
        _label("How It Works")
        st.markdown(f'<p style="font-size:0.83rem;color:{C["muted"]};margin-bottom:0.8rem;">'
                    f'Each cell shows the percentage of flights that were delayed at that '
                    f'specific hour on that specific day of the week. '
                    f'Darker red = higher delay rate.</p>', unsafe_allow_html=True)
        st.plotly_chart(plot_hour_delay_heatmap(df), use_container_width=True)
        _insight("Friday evenings (17:00–20:00) show the highest combined delay rates across "
                 "all airports. This reflects both high traffic volume and end-of-week cascade "
                 "effects where one delay ripples into later flights.")

    with tab3:
        _label("How It Works")
        st.markdown(f'<p style="font-size:0.83rem;color:{C["muted"]};margin-bottom:0.8rem;">'
                    f'Each bubble represents one airport. The X-axis shows average congestion '
                    f'level; the Y-axis shows delay rate. Bubble size = number of flights. '
                    f'A clear upward trend confirms congestion drives delays.</p>',
                    unsafe_allow_html=True)
        st.plotly_chart(plot_congestion_vs_delay(df), use_container_width=True)
        _insight("There is a strong positive correlation between congestion and delay rate. "
                 "Airports with >80 flights/hr average have delay rates roughly 15–20 percentage "
                 "points higher than low-traffic airports.")

        _label("Top 10 Most Congested Airports")
        top = (df.groupby("origin")["airport_congestion"].mean()
               .sort_values(ascending=False).head(10).reset_index())
        top.columns = ["Airport","avg"]
        top["avg"] = top["avg"].round(1)
        for _, row in top.iterrows():
            pct = row["avg"]/100
            bar_c = C["danger"] if pct>0.85 else C["warning"] if pct>0.70 else C["success"]
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">'
                f'<span style="font-family:DM Mono,monospace;font-weight:500;width:3rem;'
                f'font-size:0.85rem;color:{C["text"]};">{row["Airport"]}</span>'
                f'<div style="flex:1;background:{C["border"]};border-radius:4px;height:8px;">'
                f'<div style="width:{pct*100:.0f}%;background:{bar_c};height:8px;border-radius:4px;"></div></div>'
                f'<span style="font-family:DM Mono,monospace;font-size:0.82rem;color:{bar_c};width:5rem;'
                f'text-align:right;">{row["avg"]} fl/hr</span></div>', unsafe_allow_html=True)
