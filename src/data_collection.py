import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
from urllib.error import HTTPError

def fetch_macro_data(api_key, series_ids, output_file="data/macro_data.csv", observation_start="1955-01-01"):
    base_url = "https://api.stlouisfed.org/fred/series/observations"
    data = {}
    for series_id in series_ids:
        frequency = "q" if series_id == "GDP" else "m"  # Native frequency
        params = {
            "api_key": api_key,
            "file_type": "json",
            "observation_start": observation_start,
            "frequency": frequency,
            "limit": 100000,
            "sort_order": "asc",
            "series_id": series_id
        }
        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            json_data = response.json()
            data[series_id] = [(obs["date"], obs["value"]) for obs in json_data["observations"]
                              if obs["value"] != "."]
            print(f"Fetched {len(data[series_id])} observations for {series_id}")
        except requests.RequestException as e:
            print(f"Error fetching {series_id}: {e}")
            continue

    if data:
        # Create DataFrames for each series
        dfs = {}
        for sid, values in data.items():
            # Convert to DataFrame with datetime index
            df = pd.DataFrame(
                [float(v) for _, v in values], 
                index=pd.to_datetime([d for d, _ in values]), 
                columns=[sid]
            )
            
            # Special handling for GDP (quarterly data)
            if sid == "GDP":
                # Create a new DataFrame with monthly replication
                df_monthly = pd.DataFrame(index=pd.date_range(start=df.index.min(), end=df.index.max(), freq='M'))
                
                # Replicate quarterly GDP values across three months
                for date, value in df.iterrows():
                    month_start = date.to_period('Q').to_timestamp()
                    quarter_months = pd.date_range(start=month_start, periods=3, freq='M')
                    for q_month in quarter_months:
                        df_monthly.loc[q_month, sid] = value[0]
                
                dfs[sid] = df_monthly
            else:
                # For other series, just resample to monthly
                dfs[sid] = df.resample("M").ffill()

        # Combine all DataFrames
        df_combined = pd.concat(dfs.values(), axis=1)
        
        # Forward fill any remaining gaps
        df_combined = df_combined.ffill()
        
        # Save to CSV
        df_combined.to_csv(output_file)
        return df_combined
    else:
        print("No data fetched.")
        return None

def scrape_cb_statements(url, output_file="data/cb_statements.txt"):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text(separator="\n").strip()
        if "Page not Found" in text or not text:
            raise ValueError("Invalid or empty page content.")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)
        return text
    except (requests.RequestException, ValueError) as e:
        error_msg = f"Error scraping {url}: {e}"
        print(error_msg)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(error_msg)
        return None

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")

    FRED_API_KEY = "YOUR_FRED_API_KEY"  # Replace with your FRED API key
    SERIES_IDS = ["CPIAUCSL", "UNRATE", "GDP", "FEDFUNDS"]
    macro_df = fetch_macro_data(FRED_API_KEY, SERIES_IDS)
    if macro_df is not None:
        print(f"Macro data saved to data/macro_data.csv with shape: {macro_df.shape}")

    FOMC_URL = "https://www.federalreserve.gov/monetarypolicy/fomcminutes20231213.htm"
    cb_text = scrape_cb_statements(FOMC_URL)
    if cb_text:
        print("Central bank statements saved to data/cb_statements.txt")
    else:
        print("Failed to fetch central bank statements. Check URL or connection.")