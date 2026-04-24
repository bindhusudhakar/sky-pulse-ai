"""pages/models.py - SkyPulse v2"""
import sys, streamlit as st, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from visualize import plot_model_comparison_plotly, plot_feature_importance_plotly
from constants import C

def _label(t):
    st.markdown(f'<p style="font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.09em;color:{C["muted"]};margin:1.4rem 0 0.5rem;">{t}</p>', unsafe_allow_html=True)

def _insight(text):
    st.markdown(f'<div style="background:{C["primary_lt"]};border-left:3px solid {C["primary"]};border-radius:0 10px 10px 0;padding:0.75rem 1rem;margin:0.5rem 0 1rem;font-size:0.83rem;color:{C["text"]};"><strong style="color:{C["primary"]};">Insight: </strong>{text}</div>', unsafe_allow_html=True)

def render(metadata, base_dir):
    st.markdown(f'<h2 style="font-size:1.6rem;font-weight:700;color:{C["text"]};margin:0 0 4px;">Model Performance</h2><p style="color:{C["muted"]};font-size:0.92rem;margin:0 0 1.5rem;">Three ML models trained and compared. This page shows accuracy, key features, and evaluation metrics on 2,000 unseen test flights.</p>', unsafe_allow_html=True)
    best=next(r for r in metadata["results"] if r["name"]==metadata["best_model_name"])
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("Best Model",best["name"].split()[0])
    c2.metric("Accuracy",f'{best["accuracy"]:.1%}')
    c3.metric("F1-Score",f'{best["f1"]:.3f}')
    c4.metric("ROC-AUC",f'{best["roc_auc"]:.3f}')
    c5.metric("Training Flights",f'{metadata["train_size"]:,}')
    st.markdown(f'<div style="background:{C["success_lt"]};border:1px solid {C["success"]}44;border-radius:10px;padding:0.75rem 1rem;margin:1rem 0;font-size:0.83rem;color:{C["text"]};">Winner: {metadata["best_model_name"]} selected based on highest F1-Score ({best["f1"]:.4f}), which best balances catching real delays vs false alarms.</div>', unsafe_allow_html=True)
    tab1,tab2,tab3,tab4=st.tabs(["Model Comparison","Feature Importance","Metrics Table","Evaluation Plots"])
    with tab1:
        _label("How It Works")
        st.markdown(f'<p style="font-size:0.83rem;color:{C["muted"]};margin-bottom:0.8rem;">Each group of bars = one metric. Amber = Decision Tree, Blue = Random Forest, Purple = Logistic Regression. Taller bar = better score.</p>', unsafe_allow_html=True)
        st.plotly_chart(plot_model_comparison_plotly(metadata),use_container_width=True)
        _insight("Random Forest wins on F1-Score (0.754). Logistic Regression has slightly higher ROC-AUC (0.725) meaning it ranks risk better, but Random Forest is more decisive overall.")
        with st.expander("What do these metrics mean?"):
            st.markdown("<p style=\"font-size:0.85rem;\"><strong>Accuracy</strong> — Overall % of correct predictions.<br><strong>Precision</strong> — Of flights predicted delayed, what % actually were? (avoids false alarms)<br><strong>Recall</strong> — Of flights that were delayed, what % did we catch? (avoids missing delays)<br><strong>F1-Score</strong> — Balance of Precision and Recall. Best single metric for imbalanced data.<br><strong>ROC-AUC</strong> — How well the model ranks flights by risk (1.0 = perfect, 0.5 = random).</p>", unsafe_allow_html=True)
    with tab2:
        _label("How It Works")
        st.markdown(f'<p style="font-size:0.83rem;color:{C["muted"]};margin-bottom:0.8rem;">Shows how much each input variable contributed to Random Forest decisions. Red = very important, amber = moderate, blue = minor contribution.</p>', unsafe_allow_html=True)
        st.plotly_chart(plot_feature_importance_plotly(metadata,15),use_container_width=True)
        _insight("Carrier delay history, NAS delay, and weather-congestion risk are the top predictors. Airlines with historical delay problems continue to accumulate them, and bad weather at congested airports multiplies the risk.")
        _label("Top 10 Features")
        rf=next(r for r in metadata["results"] if r["name"]=="Random Forest")
        top10=list(rf["feature_importance"].items())[:10]
        max_v=max(v for _,v in top10)
        for feat,imp in top10:
            w=int(imp/max_v*100)
            bar_c=C["danger"] if imp>=0.06 else C["warning"] if imp>=0.03 else C["primary"]
            st.markdown(f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;"><span style="font-family:DM Mono,monospace;font-size:0.78rem;color:{C["muted"]};width:18rem;overflow:hidden;">{feat}</span><div style="flex:1;background:{C["border"]};border-radius:4px;height:8px;"><div style="width:{w}%;background:{bar_c};height:8px;border-radius:4px;"></div></div><span style="font-family:DM Mono,monospace;font-size:0.78rem;color:{C["text"]};width:5rem;text-align:right;">{imp:.4f}</span></div>', unsafe_allow_html=True)
    with tab3:
        _label("Full Metrics — Evaluated on 2,000 Test Flights")
        rows=[]
        for r in metadata["results"]:
            star="Winner: " if r["name"]==metadata["best_model_name"] else ""
            rows.append({"Model":star+r["name"],"Accuracy":f'{r["accuracy"]:.4f}',"Precision":f'{r["precision"]:.4f}',"Recall":f'{r["recall"]:.4f}',"F1-Score":f'{r["f1"]:.4f}',"ROC-AUC":f'{r["roc_auc"]:.4f}'})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
        with st.expander("What is a Confusion Matrix?"):
            st.markdown("<p style=\"font-size:0.85rem;\">A confusion matrix shows 4 outcome types:<br>True Positive: Predicted delayed, actually delayed (correct)<br>True Negative: Predicted on-time, actually on-time (correct)<br>False Positive: Predicted delayed, actually on-time (false alarm)<br>False Negative: Predicted on-time, actually delayed (missed delay — most costly in aviation)</p>", unsafe_allow_html=True)
    with tab4:
        plot_dir=Path(base_dir)/"models"/"plots"
        plot_files=list(plot_dir.glob("*.png")) if plot_dir.exists() else []
        if plot_files:
            _label("Evaluation Plots from Training")
            st.markdown(f'<p style="font-size:0.83rem;color:{C["muted"]};margin-bottom:0.8rem;">Confusion matrices, ROC curves, and comparison charts generated on the test set.</p>', unsafe_allow_html=True)
            for i in range(0,len(plot_files),2):
                cols=st.columns(2)
                for j,col in enumerate(cols):
                    if i+j<len(plot_files):
                        with col:
                            st.image(str(plot_files[i+j]),caption=plot_files[i+j].stem.replace("_"," ").title(),use_column_width=True)
        else:
            st.info("No evaluation plots found. Run: python generate_plots.py")
