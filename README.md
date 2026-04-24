# ✈️ SkyPulse — Smart Aviation Delay Prediction System

SkyPulse is a machine learning-based web application that predicts flight delays using real-world factors like weather, airport congestion, and aircraft conditions. It also includes a live monitoring system to track ongoing flights and assess delay risks in real-time.

---

## 🚀 Features

- 🔮 **Delay Predictor**  
  Predicts whether a flight will be delayed or on-time using ML models.

- 📊 **Congestion Analysis**  
  Analyzes how airport traffic affects flight delays.

- 🛣 **Route Analysis**  
  Identifies best and worst routes based on delay rates.

- 🌦 **Weather Impact**  
  Shows how different weather conditions influence delays.

- 🤖 **Model Performance**  
  Compares ML models like Random Forest, Decision Tree, and Logistic Regression.

- 📁 **Data Explorer**  
  Allows users to explore and filter the dataset interactively.

- 📡 **Live Flight Monitor (Real-Time Feature)**  
  Displays real-time flight data and predicts delay risk dynamically using a trained model.

---

## 📡 Live Monitoring (New Feature)

The Live Flight Monitor simulates real-time aviation tracking:

- Tracks multiple flights simultaneously  
- Auto-refreshes data every 30 seconds  
- Classifies flights into:
  - 🟢 Low Risk  
  - 🟡 Medium Risk  
  - 🟠 High Risk  
  - 🔴 Critical Risk  
- Displays delay probability for each flight  
- Helps identify high-risk flights instantly  

👉 This feature demonstrates how real-world aviation systems monitor and manage delays proactively.

---

## 🧠 Technologies Used

- **Frontend:** Streamlit  
- **Backend:** Python  
- **Machine Learning:** Scikit-learn  
- **Data Handling:** Pandas, NumPy  
- **Visualization:** Plotly, Matplotlib  
- **Live Data (Optional):** OpenSky API  

---

## ⚙️ Project Structure
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
---

## 📊 How It Works

1. User inputs flight parameters or selects live monitoring  
2. Data is processed and passed into trained ML models  
3. Model calculates delay probability  
4. Flights are categorized into risk levels  
5. Results are displayed through interactive dashboards  

---

## ▶️ How to Run the Project

```bash
git clone https://github.com/your-username/skypulse.git
cd skypulse_v2
pip install -r requirements.txt
streamlit run app.py
```
---
📌 Key Insights
High airport congestion increases delay probability
Bad weather (fog, storms) significantly affects flights
Peak hours lead to more delays
Machine learning improves prediction accuracy

📢 Conclusion
SkyPulse demonstrates how machine learning can be used to predict and analyze flight delays efficiently. The addition of live monitoring makes it closer to real-world aviation systems, helping users make better travel decisions.

---
##👩‍💻 Author
Bindhu S
