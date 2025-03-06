from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from xgboost import XGBClassifier
from preprocessing import create_sequences

app = Flask(__name__)

try:
    lstm_model = load_model("lstm_model.h5")
    print("LSTM model loaded successfully.")
except Exception as e:
    print(f"Error loading LSTM model: {e}")
    lstm_model = None

try:
    xgb_model = XGBClassifier()
    xgb_model.load_model("xgb_model.json")
    print("XGBoost model loaded successfully.")
except Exception as e:
    print(f"Error loading XGBoost model: {e}")
    xgb_model = None

@app.route("/predict", methods=["POST"])
def predict():
    if lstm_model is None or xgb_model is None:
        return jsonify({"error": "Model loading failed. Check server logs."}), 500
    
    data = request.get_json()
    macro_data = np.array(data["macro_data"])
    sentiment = data["sentiment"]
    
    seq_length = 10
    if len(macro_data) < seq_length:
        macro_data = np.pad(macro_data, ((seq_length - len(macro_data), 0), (0, 0)), mode="edge")
    X_seq = create_sequences(macro_data, seq_length)[-1].reshape(1, seq_length, -1)
    
    lstm_features = lstm_model.predict(X_seq)
    X_combined = np.hstack((macro_data[-1], sentiment["compound"], lstm_features[0]))
    
    prediction = xgb_model.predict(X_combined.reshape(1, -1))
    probs = xgb_model.predict_proba(X_combined.reshape(1, -1))[0]
    
    return jsonify({
        "prediction": int(prediction[0]),
        "confidence": float(max(probs)),
        "interpretation": "Hike" if prediction[0] == 1 else "Cut" if prediction[0] == 2 else "No Change"
    })

if __name__ == "__main__":
    app.run(debug=True)