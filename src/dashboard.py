import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from tensorflow.keras.models import load_model
from xgboost import XGBClassifier
from preprocessing import create_sequences

st.title("Interest Rate Forecasting Dashboard")

macro_df = pd.read_csv("data/macro_processed.csv", index_col=0, parse_dates=True)
st.subheader("Macroeconomic Trends")
fig = px.line(macro_df, title="Standardized Indicators")
st.plotly_chart(fig)

lstm_model = load_model("lstm_model.h5")
xgb_model = XGBClassifier()
xgb_model.load_model("xgb_model.json")

st.subheader("Latest Prediction")
macro_data = macro_df.values
sentiment = {"compound": 0.7}
seq_length = 10
X_seq = create_sequences(macro_data, seq_length)[-1].reshape(1, seq_length, -1)
lstm_features = lstm_model.predict(X_seq)
X_combined = np.hstack((macro_data[-1], sentiment["compound"], lstm_features[0]))
prediction = xgb_model.predict(X_combined.reshape(1, -1))[0]
probs = xgb_model.predict_proba(X_combined.reshape(1, -1))[0]

st.write(f"Prediction: {'Hike' if prediction == 1 else 'Cut' if prediction == 2 else 'No Change'}")
st.write(f"Confidence: {max(probs):.2%}")

st.subheader("Model Explainability")
st.image("shap_summary.png", caption="SHAP Summary Plot")