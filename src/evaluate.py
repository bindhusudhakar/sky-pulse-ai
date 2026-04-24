"""
src/evaluate.py
Generate evaluation plots: confusion matrices, ROC curves,
feature importance charts, and model comparison tables.
"""

import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    classification_report
)

BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"
PLOT_DIR = BASE_DIR / "models" / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# ── Styling ────────────────────────────────────────────────────────────────────
PALETTE = {
    "bg": "#0D1117",
    "card": "#161B22",
    "accent": "#00D4AA",
    "accent2": "#FF6B6B",
    "accent3": "#4ECDC4",
    "text": "#E6EDF3",
    "muted": "#8B949E",
    "dt": "#F7931A",
    "rf": "#00D4AA",
    "lr": "#A78BFA",
}

MODEL_COLORS = {
    "Decision Tree": PALETTE["dt"],
    "Random Forest": PALETTE["rf"],
    "Logistic Regression": PALETTE["lr"],
}


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": PALETTE["bg"],
        "axes.facecolor": PALETTE["card"],
        "axes.edgecolor": PALETTE["muted"],
        "axes.labelcolor": PALETTE["text"],
        "xtick.color": PALETTE["muted"],
        "ytick.color": PALETTE["muted"],
        "text.color": PALETTE["text"],
        "grid.color": "#21262D",
        "grid.linewidth": 0.8,
        "font.family": "monospace",
    })


def plot_confusion_matrices(models_dict, X_test, y_test, scaler=None):
    """Plot confusion matrices for all models side by side."""
    setup_style()
    n = len(models_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle("Confusion Matrices — All Models",
                 color=PALETTE["text"], fontsize=16, fontweight="bold", y=1.02)

    for ax, (name, model) in zip(axes, models_dict.items()):
        if name == "Logistic Regression" and scaler:
            X_eval = scaler.transform(X_test)
        else:
            X_eval = X_test

        y_pred = model.predict(X_eval)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(
            cm, annot=True, fmt="d", ax=ax,
            cmap=sns.color_palette(["#0D1117", PALETTE["accent"]], as_cmap=True),
            linewidths=2, linecolor=PALETTE["bg"],
            annot_kws={"size": 14, "weight": "bold", "color": PALETTE["text"]},
            cbar=False,
        )
        ax.set_facecolor(PALETTE["card"])
        ax.set_title(name, color=MODEL_COLORS[name], fontsize=13, pad=10)
        ax.set_xlabel("Predicted", color=PALETTE["muted"])
        ax.set_ylabel("Actual", color=PALETTE["muted"])
        ax.set_xticklabels(["On-Time", "Delayed"], color=PALETTE["text"])
        ax.set_yticklabels(["On-Time", "Delayed"], color=PALETTE["text"], rotation=0)

    plt.tight_layout()
    path = PLOT_DIR / "confusion_matrices.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  Saved: {path.name}")
    return path


def plot_roc_curves(models_dict, X_test, y_test, scaler=None):
    """Plot ROC curves for all models on one chart."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["card"])

    # Diagonal reference
    ax.plot([0, 1], [0, 1], "--", color=PALETTE["muted"], linewidth=1, label="Random Classifier")

    for name, model in models_dict.items():
        if name == "Logistic Regression" and scaler:
            X_eval = scaler.transform(X_test)
        else:
            X_eval = X_test
        y_prob = model.predict_proba(X_eval)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=MODEL_COLORS[name], linewidth=2.5,
                label=f"{name} (AUC = {roc_auc:.3f})")

    ax.set_xlabel("False Positive Rate", color=PALETTE["muted"])
    ax.set_ylabel("True Positive Rate", color=PALETTE["muted"])
    ax.set_title("ROC Curves — Model Comparison",
                 color=PALETTE["text"], fontsize=14, fontweight="bold")
    ax.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["muted"],
              labelcolor=PALETTE["text"])
    ax.grid(True, alpha=0.3)

    path = PLOT_DIR / "roc_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  Saved: {path.name}")
    return path


def plot_feature_importance(metadata, top_n=15):
    """Plot feature importance for Random Forest (most reliable)."""
    setup_style()

    # Get RF importance from metadata
    rf_result = next(r for r in metadata["results"] if r["name"] == "Random Forest")
    importance_dict = rf_result["feature_importance"]

    features = list(importance_dict.keys())[:top_n]
    importances = list(importance_dict.values())[:top_n]

    # Reverse for horizontal bar chart
    features = features[::-1]
    importances = importances[::-1]

    # Color gradient
    colors = [plt.cm.YlOrRd(v / max(importances)) for v in importances]

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["card"])

    bars = ax.barh(features, importances, color=colors, edgecolor=PALETTE["bg"],
                   linewidth=0.5, height=0.7)

    # Value labels
    for bar, val in zip(bars, importances):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left",
                color=PALETTE["muted"], fontsize=9)

    ax.set_xlabel("Importance Score", color=PALETTE["muted"])
    ax.set_title(f"Top {top_n} Feature Importances — Random Forest",
                 color=PALETTE["text"], fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = PLOT_DIR / "feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  Saved: {path.name}")
    return path


def plot_model_comparison(metadata):
    """Bar chart comparing all models on all metrics."""
    setup_style()

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    model_names = [r["name"] for r in metadata["results"]]

    x = np.arange(len(metrics))
    width = 0.25
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["card"])

    for i, (result, offset) in enumerate(zip(metadata["results"], offsets)):
        values = [result[m] for m in metrics]
        bars = ax.bar(x + offset, values, width - 0.02,
                      label=result["name"],
                      color=MODEL_COLORS[result["name"]],
                      alpha=0.85, edgecolor=PALETTE["bg"])
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom",
                    color=PALETTE["text"], fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, color=PALETTE["text"])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", color=PALETTE["muted"])
    ax.set_title("Model Performance Comparison",
                 color=PALETTE["text"], fontsize=14, fontweight="bold")
    ax.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["muted"],
              labelcolor=PALETTE["text"])
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = PLOT_DIR / "model_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  Saved: {path.name}")
    return path


def generate_all_plots(models_dict, X_test, y_test, metadata, scaler=None):
    """Generate all evaluation plots."""
    print("\n[Evaluate] Generating evaluation plots...")
    paths = {}
    paths["confusion"] = plot_confusion_matrices(models_dict, X_test, y_test, scaler)
    paths["roc"] = plot_roc_curves(models_dict, X_test, y_test, scaler)
    paths["importance"] = plot_feature_importance(metadata)
    paths["comparison"] = plot_model_comparison(metadata)
    print(f"[Evaluate] All plots saved to {PLOT_DIR}")
    return paths


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    from train_model import train_all_models

    metadata, trained_models, X_test, y_test, scaler = train_all_models()
    generate_all_plots(trained_models, X_test, y_test, metadata, scaler)
