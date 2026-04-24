# SkyPulse v2 — Smart Aviation Delay Intelligence

**Run:** `bash run.sh`  or  `streamlit run app.py`

## Structure
```
skypulse_v2/
├── app.py              Entry point (routing only)
├── sidebar.py          Sidebar nav + flight controls
├── constants.py        All shared constants
├── generate_plots.py   Generate evaluation plots
├── run.sh              One-command setup
│
├── styles/
│   └── theme.css       All CSS (light theme)
│
├── pages/
│   ├── predictor.py    Delay Predictor
│   ├── congestion.py   Congestion Analysis
│   ├── routes.py       Route Analysis
│   ├── weather.py      Weather Impact
│   ├── models.py       Model Performance
│   └── explorer.py     Data Explorer
│
├── src/
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── predict.py
│   ├── visualize.py    All 9 chart functions
│   └── evaluate.py
│
├── data/
│   ├── flights.csv     10,000 synthetic flights
│   └── generate_dataset.py
│
└── models/             Pre-trained artifacts
    ├── best_model.pkl
    ├── metadata.json
    └── plots/          Evaluation PNGs
```

## Pages
| Page | What it shows |
|------|--------------|
| Delay Predictor | ML prediction + gauge + factors |
| Congestion Analysis | Airport heatmaps by hour/day |
| Route Analysis | Best/worst routes + airline ranking |
| Weather Impact | Delay vs weather + monthly trend |
| Model Performance | DT vs RF vs LR + feature importance |
| Data Explorer | Filter + export full dataset |
