"""
generate_plots.py
Run once to generate evaluation plots from already-trained models.
Usage: python generate_plots.py
"""
import sys, json, joblib
from pathlib import Path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "src"))
from preprocess import preprocess_pipeline
from feature_engineering import engineer_all_features, get_engineered_feature_columns
from evaluate import generate_all_plots
from sklearn.model_selection import train_test_split

print("SkyPulse — Generating Evaluation Plots")
X_base, y, df, _ = preprocess_pipeline()
df_eng = engineer_all_features(df)
feature_cols = [c for c in get_engineered_feature_columns() if c in df_eng.columns]
X = df_eng[feature_cols].fillna(0)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
MODEL_DIR = BASE_DIR / "models"
models = {
    "Decision Tree":       joblib.load(MODEL_DIR / "decision_tree.pkl"),
    "Random Forest":       joblib.load(MODEL_DIR / "random_forest.pkl"),
    "Logistic Regression": joblib.load(MODEL_DIR / "logistic_regression.pkl"),
}
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
with open(MODEL_DIR / "metadata.json") as f:
    metadata = json.load(f)
generate_all_plots(models, X_test, y_test, metadata, scaler)
print("Done — plots saved to models/plots/")
