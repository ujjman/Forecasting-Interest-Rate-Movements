import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Preprocess macroeconomic data with feature engineering
def preprocess_macro_data(input_file="data/macro_data.csv", output_file="data/macro_processed.csv"):
    df = pd.read_csv(input_file, index_col=0, parse_dates=True)
    df = df.astype(float)
    
    # Add lagged features (1-month lag)
    for col in df.columns:
        df[f"{col}_lag1"] = df[col].shift(1)
    
    # Add moving averages (3-month)
    for col in df.columns:
        if "lag" not in col:
            df[f"{col}_ma3"] = df[col].rolling(window=3).mean()
    
    # Drop rows with NaN from shifting/rolling
    df = df.dropna()
    
    # Standardize all features
    df = (df - df.mean()) / df.std()
    df.to_csv(output_file)
    return df

# Preprocess central bank text
def preprocess_text(input_file="data/cb_statements.txt"):
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    
    return " ".join(tokens), sentiment

# Prepare data for LSTM (create sequences)
def create_sequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
    return np.array(X)

# Generate target variable from FEDFUNDS
def generate_target(df, column="FEDFUNDS"):
    rates = df[column].dropna()
    changes = rates.diff().fillna(0)
    target = np.where(changes > 0, 1, np.where(changes < 0, 2, 0))  # 1 = hike, 2 = cut, 0 = no change
    return target

if __name__ == "__main__":
    macro_df = preprocess_macro_data()
    cleaned_text, sentiment = preprocess_text()
    target = generate_target(macro_df)
    print(f"Processed macro data shape: {macro_df.shape}")
    print(f"Sentiment: {sentiment}")
    print(f"Target shape: {target.shape}")