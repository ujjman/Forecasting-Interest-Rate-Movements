# 🏦 Interest Rate Forecasting Model 📈

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-green)](https://xgboost.readthedocs.io/)
[![Flask](https://img.shields.io/badge/Flask-API-lightgrey)](https://flask.palletsprojects.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)](https://streamlit.io/)

> A hybrid deep learning approach for forecasting central bank interest rate decisions using macroeconomic indicators and sentiment analysis of policy statements.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [API Reference](#api-reference)
- [Dashboard](#dashboard)
- [Results](#results)
- [Contributing](#contributing)

## 🔍 Overview

This project implements a hybrid machine learning system to forecast central bank interest rate decisions. By combining LSTM neural networks with gradient boosting techniques and integrating sentiment analysis of central bank communications, the model provides accurate predictions of monetary policy directions (rate hikes, cuts, or holds).

## ✨ Features

- 📊 Automated retrieval of macroeconomic indicators from FRED API
- 📝 Web scraping of FOMC meeting minutes and statements
- 📉 Advanced time series preprocessing with feature engineering
- 🧠 Hybrid model combining LSTM and XGBoost algorithms
- 🔍 NLP-based sentiment analysis of central bank communications
- 🌐 RESTful API for real-time predictions
- 📱 Interactive Streamlit dashboard for visualization
- 🧩 Model explainability using SHAP values

## 📂 Project Structure

```
interest-rate-forecasting/
├── data/                      # Data directory
│   ├── macro_data.csv         # Raw macroeconomic data
│   ├── macro_processed.csv    # Processed features
│   └── cb_statements.txt      # Central bank statements
├── data_collection.py         # Data fetching and scraping code
├── preprocessing.py           # Data preprocessing pipeline
├── modeling.py                # Model training and evaluation
├── api.py                     # Flask API implementation
├── dashboard.py               # Streamlit dashboard
├── lstm_model.h5              # Saved LSTM model
├── xgb_model.json             # Saved XGBoost model
├── shap_summary.png           # Model explainability visualization
├── lstm_training.png          # Training performance chart
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ujjman/Forecasting-Interest-Rate-Movements
   cd Forecasting-Interest-Rate-Movements
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your FRED API key:
   - Get a free API key from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html)
   - Update the `FRED_API_KEY` variable in `data_collection.py`

## 🛠️ Usage

### Data Collection

Fetch macroeconomic indicators and central bank statements:

```bash
python data_collection.py
```

This script fetches data for:
- Consumer Price Index (CPIAUCSL)
- Unemployment Rate (UNRATE)
- Gross Domestic Product (GDP)
- Federal Funds Rate (FEDFUNDS)
- FOMC meeting minutes

### Data Preprocessing

Process and transform the data:

```bash
python preprocessing.py
```

This generates:
- Lagged features
- Moving averages
- Standardized indicators
- Lemmatized and tokenized text
- Sentiment scores

### Model Training

Train the hybrid LSTM+XGBoost model:

```bash
python modeling.py
```

### Running the API

Start the prediction API server:

```bash
python api.py
```

The API will be available at `http://localhost:5000/predict`

### Launching the Dashboard

Start the interactive dashboard:

```bash
streamlit run dashboard.py
```

The dashboard will be available at `http://localhost:8501`

## 🧠 Model Architecture

The project employs a hybrid approach:

### LSTM Component

- Processes sequences of macroeconomic indicators
- Architecture:
  - Input layer: (sequence_length, n_features)
  - LSTM layer 1: 100 units with ReLU activation
  - Dropout: 30%
  - LSTM layer 2: 50 units with ReLU activation
  - Dropout: 30%
  - Output layer: 3 units with softmax activation

### XGBoost Component

- Integrates LSTM features with sentiment scores
- Hyperparameters:
  - Max depth: 5
  - Learning rate: 0.05
  - Number of estimators: 200

### Target Variable

The model predicts one of three classes:
- **0**: No change in interest rates
- **1**: Rate hike
- **2**: Rate cut

## 🌐 API Reference

### Endpoint: `/predict`

**Method**: POST

**Request format**:
```json
{
  "macro_data": [[...]],  // 2D array of macroeconomic indicators
  "sentiment": {
    "compound": 0.7,      // Sentiment compound score
    "pos": 0.2,           // Positive sentiment score
    "neu": 0.7,           // Neutral sentiment score
    "neg": 0.1            // Negative sentiment score
  }
}
```

**Response format**:
```json
{
  "prediction": 0,            // 0: No change, 1: Hike, 2: Cut
  "confidence": 0.85,         // Prediction confidence
  "interpretation": "No Change"  // Human-readable result
}
```

## 📊 Dashboard

The Streamlit dashboard provides:

- Interactive visualization of macroeconomic trends
- Latest interest rate prediction with confidence score
- Model explainability with SHAP value visualization
- Historical prediction accuracy

## 📈 Results

The hybrid model achieves:
- Higher accuracy than traditional econometric models
- Better handling of policy regime shifts
- Improved interpretability through SHAP values
- Successful incorporation of qualitative data from central bank communications

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

📝 **Note**: This model is for research purposes only and should not be used as the sole basis for financial decisions.


📧 **Contact**: ujjwalsini.um@gmail.com