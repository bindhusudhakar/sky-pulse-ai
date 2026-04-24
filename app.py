"""
app.py — SkyPulse v2
Entry point. Loads CSS, caches data, renders top nav + sidebar + page.
Run with: streamlit run app.py
"""
import sys, json, warnings
from pathlib import Path
import streamlit as st

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR))

st.set_page_config(
    page_title="SkyPulse — Aviation Delay Intelligence",
    page_icon="🛩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load CSS ──────────────────────────────────────────────────────────────────
with open(BASE_DIR / "styles" / "theme.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Inline critical overrides — these MUST be inline to beat Streamlit's own styles
st.markdown("""
<style>
/* Sidebar radio labels always visible */
div[data-testid="stSidebar"] div[role="radiogroup"] label,
div[data-testid="stSidebar"] div[role="radiogroup"] label p,
div[data-testid="stSidebar"] div[role="radiogroup"] label span {
    color: #1A1D23 !important;
    font-size: 0.9rem !important;
    font-weight: 400 !important;
    opacity: 1 !important;
    visibility: visible !important;
}
/* Hide Streamlit's auto file-based page nav */
section[data-testid="stSidebarNav"],
div[data-testid="stSidebarNav"] {
    display: none !important;
    height: 0 !important;
    overflow: hidden !important;
}
/* Metric cards always visible */
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * { color: #1A1D23 !important; opacity: 1 !important; visibility: visible !important; }
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] * { color: #6B7280 !important; opacity: 1 !important; visibility: visible !important; }
[data-testid="stMetricDelta"],
[data-testid="stMetricDelta"] * { opacity: 1 !important; visibility: visible !important; }
div[data-testid="metric-container"] { opacity: 1 !important; visibility: visible !important; }
/* Remove top padding so page doesn't start blank */
.block-container { padding-top: 1rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_df():
    from preprocess import load_data, clean_data, encode_categoricals
    df = load_data()
    df = clean_data(df)
    df, _ = encode_categoricals(df)
    return df

@st.cache_data(show_spinner=False)
def _load_meta():
    with open(BASE_DIR / "models" / "metadata.json") as f:
        return json.load(f)


# ── Page imports ──────────────────────────────────────────────────────────────
from sidebar import render as render_sidebar
from pages.live_monitor import render as page_live_monitor
from pages.predictor    import render as page_predictor
from pages.congestion import render as page_congestion
from pages.routes     import render as page_routes
from pages.weather    import render as page_weather
from pages.models     import render as page_models
from pages.explorer   import render as page_explorer

from constants import NAV_PAGES, C


# ── Top navigation bar ────────────────────────────────────────────────────────
def _top_nav(selected: str) -> str:
    """Render a horizontal nav bar at the top. Returns selected page if clicked."""
    # Build button HTML for each page
    buttons_html = ""
    for icon, name, _ in NAV_PAGES:
        active = "active" if name == selected else ""
        buttons_html += (
            f'<a class="nav-btn {active}" '
            f'onclick="window.parent.postMessage({{type:\'streamlit:setComponentValue\', value:\'{name}\'}}, \'*\')">'
            f'{icon} {name}</a>'
        )

    st.markdown(f"""
    <div class="top-nav">
        <div class="nav-brand">🛩 <strong>SkyPulse</strong></div>
        <div class="nav-links">{buttons_html}</div>
        <div class="nav-hint">← Use sidebar for flight settings</div>
    </div>
    """, unsafe_allow_html=True)
    return selected   # top nav is display-only; sidebar radio drives selection


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Load data
    try:
        df   = _load_df()
        meta = _load_meta()
    except FileNotFoundError as e:
        st.error(f"Missing file: {e}")
        st.code("python data/generate_dataset.py\npython src/train_model.py")
        st.stop()
    except Exception as e:
        st.error(f"Startup error: {e}")
        st.stop()

    # Sidebar (navigation + flight params)
    selected, flight_params = render_sidebar()

    # Top navigation bar (visual only — shows current page, hints at sidebar)
    _top_nav(selected)

    # Thin separator
    st.markdown(
        f'<hr style="border:none;border-top:1px solid {C["border"]};margin:0 0 1.2rem;">',
        unsafe_allow_html=True,
    )

    # Route to selected page
    pages = {
        "Live Monitor":        lambda: page_live_monitor(),
        "Delay Predictor":     lambda: page_predictor(flight_params, df),
        "Congestion Analysis": lambda: page_congestion(df),
        "Route Analysis":      lambda: page_routes(df),
        "Weather Impact":      lambda: page_weather(df, flight_params),
        "Model Performance":   lambda: page_models(meta, BASE_DIR),
        "Data Explorer":       lambda: page_explorer(df),
    }

    fn = pages.get(selected)
    if fn:
        fn()
    else:
        st.error(f"Page not found: '{selected}'. Available: {list(pages.keys())}")


if __name__ == "__main__":
    main()
