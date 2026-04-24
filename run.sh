#!/usr/bin/env bash
set -e
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   🛩  SkyPulse v2 — Smart Aviation Delay Intelligence  ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "[1/4] Installing dependencies..."
pip install -r requirements.txt -q
echo "      Done"
if [ -f "data/flights.csv" ]; then
  echo "[2/4] Dataset already exists — skipping"
else
  echo "[2/4] Generating dataset..."
  python data/generate_dataset.py
fi
if [ -f "models/metadata.json" ]; then
  echo "[3/4] Trained models found — skipping"
else
  echo "[3/4] Training models..."
  python src/train_model.py
fi
echo "[4/4] Launching dashboard at http://localhost:8501"
echo ""
streamlit run app.py
