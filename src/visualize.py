"""
src/visualize.py  —  SkyPulse v2
All Plotly chart functions. Light theme. Zero broken imports.
Every function is self-contained and returns a go.Figure.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

C = {
    "bg": "#F7F8FA", "surface": "#FFFFFF", "border": "#E8EAF0",
    "text": "#1A1D23", "muted": "#6B7280",
    "primary": "#2563EB", "success": "#16A34A",
    "danger": "#DC2626", "warning": "#D97706", "purple": "#7C3AED",
    "dt": "#D97706", "rf": "#2563EB", "lr": "#7C3AED",
}
_BASE = dict(
    paper_bgcolor=C["bg"], plot_bgcolor=C["surface"],
    font=dict(family="DM Sans, sans-serif", color=C["text"]),
    xaxis=dict(gridcolor=C["border"], linecolor=C["border"],
               tickfont=dict(color=C["muted"], size=11)),
    yaxis=dict(gridcolor=C["border"], linecolor=C["border"],
               tickfont=dict(color=C["muted"], size=11)),
    margin=dict(t=60, b=40, l=40, r=40),
    hoverlabel=dict(bgcolor=C["surface"], bordercolor=C["border"],
                    font_family="DM Sans", font_color=C["text"]),
)
def _apply(fig, h=420, title=""):
    fig.update_layout(height=h, title=dict(
        text=title, font=dict(size=15, color=C["text"]), x=0, xanchor="left"),
        **_BASE)
    return fig

def plot_congestion_heatmap(df):
    agg = df.groupby(["origin","dep_hour"])["airport_congestion"].mean().reset_index()
    pivot = agg.pivot(index="origin", columns="dep_hour", values="airport_congestion")
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}:00" for h in pivot.columns],
        y=pivot.index.tolist(),
        colorscale=[[0,"#EFF4FF"],[0.5,"#93C5FD"],[1.0,C["danger"]]],
        text=np.round(pivot.values,0).astype(int),
        texttemplate="%{text}",
        textfont={"size":9,"color":C["text"]},
        hovertemplate="Airport: %{y}<br>Hour: %{x}<br>Congestion: %{z:.0f} flights/hr<extra></extra>",
        colorbar=dict(title="Flights/hr", tickfont=dict(color=C["muted"])),
    ))
    return _apply(fig, h=540, title="Airport Congestion by Hour of Day")

def plot_hour_delay_heatmap(df):
    agg = df.groupby(["day_of_week","dep_hour"])["is_delayed"].mean().reset_index()
    pivot = agg.pivot(index="day_of_week", columns="dep_hour", values="is_delayed")
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    fig = go.Figure(go.Heatmap(
        z=(pivot.values*100).round(1),
        x=[f"{h:02d}:00" for h in pivot.columns],
        y=[days[i] for i in pivot.index],
        colorscale=[[0,"#F0FDF4"],[0.45,"#FEF3C7"],[1.0,C["danger"]]],
        hovertemplate="Day: %{y}<br>Hour: %{x}<br>Delay Rate: %{z:.1f}%<extra></extra>",
        colorbar=dict(title="Delay %", tickfont=dict(color=C["muted"])),
    ))
    return _apply(fig, h=380, title="Delay Rate by Hour & Day of Week")

def plot_congestion_vs_delay(df):
    agg = df.groupby("origin").agg(
        avg_congestion=("airport_congestion","mean"),
        delay_rate=("is_delayed","mean"),
        flights=("is_delayed","count"),
    ).reset_index()
    fig = px.scatter(
        agg, x="avg_congestion", y=(agg["delay_rate"]*100).round(1),
        size="flights", color=(agg["delay_rate"]*100).round(1), text="origin",
        color_continuous_scale=[[0,C["success"]],[0.5,C["warning"]],[1,C["danger"]]],
        labels={"x":"Avg Congestion (flights/hr)","y":"Delay Rate (%)"},
        template="none",
    )
    fig.update_traces(textposition="top center",
                      marker=dict(line=dict(width=1,color=C["surface"])))
    fig.update_layout(height=440, coloraxis_showscale=False,
        title=dict(text="Airport Congestion vs Delay Rate",
                   font=dict(size=15,color=C["text"]),x=0,xanchor="left"),
        xaxis_title="Avg Congestion (flights/hr)", yaxis_title="Delay Rate (%)",
        **_BASE)
    return fig

def plot_delay_by_airline(df):
    agg = (df.groupby(["airline","airline_name"])["is_delayed"]
           .agg(["mean","count"]).reset_index()
           .rename(columns={"mean":"rate","count":"flights"})
           .sort_values("rate", ascending=True))
    colors = [C["danger"] if r>0.70 else C["warning"] if r>0.58 else C["success"]
              for r in agg["rate"]]
    fig = go.Figure(go.Bar(
        x=(agg["rate"]*100).round(1), y=agg["airline_name"], orientation="h",
        marker_color=colors, text=[f"{v:.1f}%" for v in agg["rate"]*100],
        textposition="outside", customdata=agg["flights"],
        hovertemplate="%{y}<br>Delay Rate: %{x:.1f}%<br>Flights: %{customdata:,}<extra></extra>",
    ))
    fig.update_xaxes(title_text="Delay Rate (%)", range=[0,105])
    return _apply(fig, h=380, title="Delay Rate by Airline")

def plot_route_efficiency(df, top_n=12):
    stats = (df.groupby(["origin","dest"])
             .agg(rate=("is_delayed","mean"),flights=("is_delayed","count"))
             .reset_index())
    stats = stats[stats["flights"]>=20]
    stats["route"] = stats["origin"] + " → " + stats["dest"]
    half = top_n//2
    combined = pd.concat([stats.nlargest(half,"rate"),
                          stats.nsmallest(half,"rate")]).sort_values("rate")
    colors = [C["success"] if r<0.55 else C["danger"] for r in combined["rate"]]
    fig = go.Figure(go.Bar(
        x=(combined["rate"]*100).round(1), y=combined["route"], orientation="h",
        marker_color=colors, text=[f"{v:.1f}%" for v in combined["rate"]*100],
        textposition="outside",
        hovertemplate="%{y}<br>Delay Rate: %{x:.1f}%<extra></extra>",
    ))
    fig.update_xaxes(title_text="Delay Rate (%)", range=[0,108])
    return _apply(fig, h=480, title="Route Efficiency — Best & Worst Routes")

def plot_weather_impact(df):
    stats = (df.groupby("origin_weather")
             .agg(avg_delay=("dep_delay_minutes","mean"),
                  delay_rate=("is_delayed","mean"))
             .reset_index().sort_values("avg_delay", ascending=False))
    fig = make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Bar(
        name="Avg Delay (min)", x=stats["origin_weather"],
        y=stats["avg_delay"].round(1), marker_color=C["danger"], opacity=0.8,
        hovertemplate="%{x}<br>Avg Delay: %{y:.1f} min<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        name="Delay Rate %", x=stats["origin_weather"],
        y=(stats["delay_rate"]*100).round(1), mode="lines+markers",
        marker=dict(color=C["primary"],size=10),
        line=dict(color=C["primary"],width=2.5),
        hovertemplate="%{x}<br>Delay Rate: %{y:.1f}%<extra></extra>",
    ), secondary_y=True)
    fig.update_yaxes(title_text="Avg Delay (min)", secondary_y=False)
    fig.update_yaxes(title_text="Delay Rate (%)", secondary_y=True)
    fig.update_layout(height=400, legend=dict(bgcolor=C["surface"],bordercolor=C["border"]),
        title=dict(text="Weather Conditions vs Flight Delays",
                   font=dict(size=15,color=C["text"]),x=0,xanchor="left"),
        **_BASE)
    return fig

def plot_monthly_delay_trend(df):
    mnth = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly = (df.groupby("month")["is_delayed"].agg(["mean","count"]).reset_index()
               .rename(columns={"mean":"rate","count":"flights"}))
    monthly["month_name"] = monthly["month"].apply(lambda m: mnth[m-1])
    fig = go.Figure(go.Scatter(
        x=monthly["month_name"], y=(monthly["rate"]*100).round(1),
        mode="lines+markers+text",
        line=dict(color=C["primary"],width=3),
        marker=dict(size=11,color=C["primary"],line=dict(width=2,color=C["surface"])),
        text=[f"{v:.0f}%" for v in monthly["rate"]*100], textposition="top center",
        fill="tozeroy", fillcolor="rgba(37,99,235,0.07)",
        hovertemplate="Month: %{x}<br>Delay Rate: %{y:.1f}%<extra></extra>",
    ))
    fig.update_yaxes(title_text="Delay Rate (%)")
    return _apply(fig, h=360, title="Monthly Delay Rate Trend")

def plot_feature_importance_plotly(metadata, top_n=15):
    rf = next(r for r in metadata["results"] if r["name"]=="Random Forest")
    items = sorted(rf["feature_importance"].items(), key=lambda x:x[1], reverse=True)[:top_n]
    items = sorted(items, key=lambda x:x[1])
    features = [i[0].replace("_"," ").title() for i in items]
    values   = [i[1] for i in items]
    colors   = [C["danger"] if v>=0.06 else C["warning"] if v>=0.03 else C["primary"]
                for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=features, orientation="h", marker_color=colors,
        text=[f"{v:.4f}" for v in values], textposition="outside",
        hovertemplate="%{y}<br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_xaxes(title_text="Importance Score")
    return _apply(fig, h=520, title=f"Top {top_n} Feature Importances — Random Forest")

def plot_model_comparison_plotly(metadata):
    metrics = ["accuracy","precision","recall","f1","roc_auc"]
    labels  = ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]
    model_colors = {"Decision Tree":C["dt"],"Random Forest":C["rf"],"Logistic Regression":C["lr"]}
    fig = go.Figure()
    for result in metadata["results"]:
        values = [result[m] for m in metrics]
        fig.add_trace(go.Bar(
            name=result["name"], x=labels, y=values,
            marker_color=model_colors.get(result["name"],C["primary"]),
            text=[f"{v:.3f}" for v in values], textposition="outside",
            hovertemplate=f"<b>{result['name']}</b><br>%{{x}}: %{{y:.4f}}<extra></extra>",
        ))
    fig.update_layout(barmode="group", height=440,
        yaxis=dict(range=[0,1.15], title="Score"),
        legend=dict(bgcolor=C["surface"],bordercolor=C["border"]),
        title=dict(text="Model Performance Comparison",
                   font=dict(size=15,color=C["text"]),x=0,xanchor="left"),
        **_BASE)
    return fig
