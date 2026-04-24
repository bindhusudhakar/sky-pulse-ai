"""
src/train_model.py
Train Decision Tree, Random Forest, and Logistic Regression models
for flight delay prediction. Save best model + scaler + metadata.
"""

import sys
import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import preprocess_pipeline
from feature_engineering import engineer_all_features, get_engineered_feature_columns

BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


def get_models():
    """Return dict of models to train."""
    return {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=15,
            min_samples_leaf=8,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=1.0,
            solver="lbfgs",
            random_state=42,
        ),
    }


def evaluate_model(model, X_test, y_test, model_name):
    """Compute and return full evaluation metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "name": model_name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }

    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    print(f"\n  {'─'*40}")
    print(f"  Model: {model_name}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1-Score : {metrics['f1']:.4f}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")

    return metrics


def get_feature_importance(model, feature_names, model_name):
    """Extract feature importance from tree-based models or LR coefficients."""
    importance_data = {}

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        importance_data = dict(zip(feature_names, importances.tolist()))
    elif hasattr(model, "coef_"):
        coefs = np.abs(model.coef_[0])
        importance_data = dict(zip(feature_names, coefs.tolist()))

    # Sort by importance
    importance_data = dict(
        sorted(importance_data.items(), key=lambda x: x[1], reverse=True)
    )
    return importance_data


def train_all_models():
    """Full training pipeline. Returns results dict."""
    print("=" * 60)
    print("  AVIATION DELAY PREDICTION — MODEL TRAINING")
    print("=" * 60)

    # ── Load & preprocess ───────────────────────────────────────
    print("\n[1/5] Loading and preprocessing data...")
    X_base, y, df, encoding_maps = preprocess_pipeline()

    # ── Feature engineering ─────────────────────────────────────
    print("\n[2/5] Engineering features...")
    df_eng = engineer_all_features(df)
    feature_cols = get_engineered_feature_columns()
    # Only keep columns that exist in the dataframe
    feature_cols = [c for c in feature_cols if c in df_eng.columns]
    X = df_eng[feature_cols].copy()

    # Handle any remaining NaNs
    X = X.fillna(X.median(numeric_only=True))

    print(f"       Final feature set: {X.shape[1]} features, {X.shape[0]:,} samples")

    # ── Train/test split ────────────────────────────────────────
    print("\n[3/5] Splitting data (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"       Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── Scale for Logistic Regression ───────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── Train models ────────────────────────────────────────────
    print("\n[4/5] Training models...")
    models = get_models()
    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\n  Training: {name}...")
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
            metrics = evaluate_model(model, X_test_scaled, y_test, name)
        else:
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test, name)

        importance = get_feature_importance(model, feature_cols, name)
        metrics["feature_importance"] = importance
        results.append(metrics)
        trained_models[name] = model

    # ── Select best model ───────────────────────────────────────
    print("\n[5/5] Saving artifacts...")
    best = max(results, key=lambda r: r["f1"])
    print(f"\n  ✅ Best model: {best['name']} (F1={best['f1']:.4f})")

    best_model = trained_models[best["name"]]

    # ── Save artifacts ──────────────────────────────────────────
    joblib.dump(best_model, MODEL_DIR / "best_model.pkl")
    joblib.dump(trained_models["Random Forest"], MODEL_DIR / "random_forest.pkl")
    joblib.dump(trained_models["Decision Tree"], MODEL_DIR / "decision_tree.pkl")
    joblib.dump(trained_models["Logistic Regression"], MODEL_DIR / "logistic_regression.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    metadata = {
        "best_model_name": best["name"],
        "feature_columns": feature_cols,
        "encoding_maps": encoding_maps,
        "results": results,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "delay_rate": float(y.mean()),
    }

    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save comparison table as CSV
    comparison_df = pd.DataFrame([
        {
            "Model": r["name"],
            "Accuracy": r["accuracy"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1-Score": r["f1"],
            "ROC-AUC": r["roc_auc"],
        }
        for r in results
    ])
    comparison_df.to_csv(MODEL_DIR / "model_comparison.csv", index=False)

    print(f"\n  Models saved to: {MODEL_DIR}")
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON TABLE")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    print("=" * 60)

    return metadata, trained_models, X_test, y_test, scaler


if __name__ == "__main__":
    train_all_models()
