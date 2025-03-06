import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
from preprocessing import generate_target, create_sequences
from sklearn.utils.class_weight import compute_class_weight

# Load and combine data
def load_data(macro_file="data/macro_processed.csv", sentiment=None, seq_length=10):
    macro_df = pd.read_csv(macro_file, index_col=0, parse_dates=True)
    macro_data = macro_df.values
    target = generate_target(macro_df)
    X_seq = create_sequences(macro_data, seq_length)
    y = target[seq_length:]
    if sentiment:
        sentiment_features = np.array([sentiment["compound"]] * len(X_seq))
        X_combined = np.hstack((X_seq[:, -1, :], sentiment_features.reshape(-1, 1)))
    else:
        X_combined = X_seq[:, -1, :]
    return X_seq, X_combined, y

# Build and train LSTM model
def train_lstm(X_seq, y, seq_length, n_features):
    model = Sequential()
    model.add(LSTM(100, activation="relu", input_shape=(seq_length, n_features), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(50, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation="softmax"))
    
    # Compute class weights to handle imbalance
    class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
    class_weight_dict = dict(enumerate(class_weights))
    
    # Custom optimizer with lower learning rate
    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    # Early stopping
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y, test_size=0.2, random_state=42)
    history = model.fit(X_train, y_train, epochs=50, batch_size=16, 
                        validation_data=(X_test, y_test), 
                        class_weight=class_weight_dict, 
                        callbacks=[early_stopping], verbose=1)
    
    plt.plot(history.history["accuracy"], label="train_accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.legend()
    plt.savefig("lstm_training.png")
    plt.close()
    
    return model

# Train hybrid model (LSTM + XGBoost)
def train_hybrid_model(X_seq, X_combined, y):
    n_features = X_seq.shape[2]
    lstm_model = train_lstm(X_seq, y, seq_length=10, n_features=n_features)
    lstm_features = lstm_model.predict(X_seq)
    
    X_final = np.hstack((X_combined, lstm_features))
    
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", max_depth=5, learning_rate=0.05, n_estimators=200)
    xgb_model.fit(X_train, y_train)
    score = xgb_model.score(X_test, y_test)
    print(f"Hybrid model accuracy: {score:.2f}")
    
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig("shap_summary.png")
    plt.close()
    
    return lstm_model, xgb_model

if __name__ == "__main__":
    sentiment = {"compound": 0.7}
    X_seq, X_combined, y = load_data(sentiment=sentiment)
    lstm_model, xgb_model = train_hybrid_model(X_seq, X_combined, y)
    
    lstm_model.save("lstm_model.h5")
    xgb_model.save_model("xgb_model.json")