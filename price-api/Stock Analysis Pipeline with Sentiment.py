# Stock Analysis Pipeline with Sentiment Data
# This script follows the original notebook code structure to extract price information,
# calculate technical indicators, and merge with sentiment data for any stock ticker

import requests
import pandas as pd
import time
from datetime import datetime
import os
import numpy as np

# Step 1: Alpha Vantage API Data Retrieval (from alphavantage.ipynb)
API_KEY = '9MBUP7GPCWD3VV9S'
BASE_URL = 'https://www.alphavantage.co/query'

def get_daily_data(symbol, outputsize='full'):
    """
    Fetch daily data for a given symbol from Alpha Vantage and save as CSV.
    outputsize options: 'compact' (latest 100 data points) or 'full' (20+ years of historical data)
    """
    params = {
        'function': 'TIME_SERIES_DAILY',
        'symbol': symbol,
        'outputsize': outputsize,
        'apikey': API_KEY,
        'datatype': 'json'
    }
    
    print(f"Fetching daily data for {symbol}...")
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code != 200:
        print(f"Error: API request failed with status code {response.status_code}")
        return None
    
    data = response.json()
    
    # Print information message if available (useful for debugging)
    if "Information" in data:
        print(f"API Information Message: {data['Information']}")
        # Return None if this is an error rather than just information
        if "thank you for using Alpha Vantage" in data["Information"]:
            return None
    
    # Check if there's an error message in the response
    if "Error Message" in data:
        print(f"API Error: {data['Error Message']}")
        return None
    
    # Extract time series data
    time_series_key = 'Time Series (Daily)'
    if time_series_key not in data:
        print(f"Error: Expected key '{time_series_key}' not found in response.")
        print("Available keys:", list(data.keys()))
        return None
    
    time_series = data[time_series_key]
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(time_series, orient='index')
    
    # Rename columns to match the desired format
    df.rename(columns={
        '1. open': 'open',
        '2. high': 'high',
        '3. low': 'low',
        '4. close': 'close',
        '5. volume': 'volume'
    }, inplace=True)
    
    # Convert types
    df['open'] = pd.to_numeric(df['open'])
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['close'] = pd.to_numeric(df['close'])
    df['volume'] = pd.to_numeric(df['volume'], downcast='integer')
    
    # Reset index to make the date a column named 'Unnamed: 0'
    df.index.name = 'Date'
    df = df.reset_index()
    df.rename(columns={'Date': 'Unnamed: 0'}, inplace=True)
    
    # Sort from oldest to newest (to match your format)
    df = df.sort_values('Unnamed: 0')
    
    # Save to CSV
    filename = f"{symbol}_daily_data.csv"
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
    
    return df

def get_multiple_symbols(symbols, outputsize='full', delay=15):
    """
    Fetch daily data for multiple symbols, respecting API rate limits.
    Adds a delay between requests to avoid hitting the rate limit.
    """
    results = {}
    
    for i, symbol in enumerate(symbols):
        print(f"Processing {i+1}/{len(symbols)}: {symbol}")
        results[symbol] = get_daily_data(symbol, outputsize)
        
        # Add delay between requests to respect API rate limits (5 calls per minute for free tier)
        if i < len(symbols) - 1:  # No need to wait after the last request
            print(f"Waiting {delay} seconds before next request...")
            time.sleep(delay)
    
    return results

# Step 2: Technical Analysis (from TA_final_1.ipynb)
def preprocess_data(file_path):
    """Load and preprocess the CSV file."""
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Rename "Unnamed: 0" to "Date" and parse it as a datetime column
    df.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    
    # Sort data by date (oldest to newest)
    df = df.sort_values(by="Date").reset_index(drop=True)
    
    return df

def calculate_true_range(df):
    """Calculate True Range for ATR."""
    df['High-Low'] = df['high'] - df['low']
    df['High-Close'] = abs(df['high'] - df['close'].shift(1))
    df['Low-Close'] = abs(df['low'] - df['close'].shift(1))
    df['True Range'] = df[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
    return df

# Volatility signal
def calculate_atr(df, period=14):
    """Calculate Average True Range (ATR) using Wilder's smoothing."""
    df = calculate_true_range(df)
    df['ATR'] = df['True Range'].ewm(alpha=1/period, adjust=False, min_periods=1).mean()
    return df

# Stock trend discovery
def calculate_adx(df, period=14):
    """Calculate Average Directional Index (ADX)."""
    df = calculate_atr(df, period)  # Ensure ATR uses Wilder's smoothing

    # Calculate +DM and -DM
    df['+DM'] = df['high'] - df['high'].shift(1)  # Up move
    df['-DM'] = df['low'].shift(1) - df['low']    # Down move (previous_low - current_low)

    # Set negative DM values to 0
    df['+DM'] = df['+DM'].clip(lower=0)
    df['-DM'] = df['-DM'].clip(lower=0)

    # Zero out the smaller DM
    mask = df['+DM'] > df['-DM']
    df.loc[~mask, '+DM'] = 0
    df.loc[mask, '-DM'] = 0

    # Smooth +DM and -DM using Wilder's method
    df['Smoothed+DM'] = df['+DM'].ewm(alpha=1/period, adjust=False).mean()
    df['Smoothed-DM'] = df['-DM'].ewm(alpha=1/period, adjust=False).mean()

    # Calculate +DI and -DI
    df['+DI'] = (df['Smoothed+DM'] / df['ATR']) * 100
    df['-DI'] = (df['Smoothed-DM'] / df['ATR']) * 100

    # Calculate DX and ADX (smoothed DX)
    df['DX'] = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])) * 100
    df['ADX'] = df['DX'].ewm(alpha=1/period, adjust=False).mean()

    return df

# Volume weights
def calculate_obv(df):
    """Calculate On-Balance Volume (OBV)."""
    df['OBV'] = (df['volume'] * ((df['close'].diff() > 0) * 2 - 1)).cumsum()
    return df

# Buy&Sell signals
def calculate_rsi(df, period=14):
    """Calculate Relative Strength Index (RSI)."""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)  # Default initial RSI to 50
    return df

# Buy&Sell signals
def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    """Calculate Moving Average Convergence Divergence (MACD)."""
    df['EMA_12'] = df['close'].ewm(span=short_window, adjust=False, min_periods=1).mean()
    df['EMA_26'] = df['close'].ewm(span=long_window, adjust=False, min_periods=1).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=signal_window, adjust=False, min_periods=1).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    return df

def calculate_technical_indicators(df):
    """Compute technical indicators including ADX, ATR, OBV, RSI, and MACD."""
    df = df.copy()
    df = calculate_atr(df, period=14)
    df = calculate_adx(df, period=14)
    df = calculate_obv(df)
    df = calculate_rsi(df, period=14)
    df = calculate_macd(df, short_window=12, long_window=26, signal_window=9)
    return df

# Step 3: Process Sentiment Data
def clean_sentiment_data(sentiment_file):
    """
    Clean and preprocess the sentiment data file.
    
    Parameters:
    - sentiment_file: Path to the sentiment CSV file
    
    Returns:
    - Dataframe with cleaned sentiment data
    """
    print(f"Processing sentiment data from {sentiment_file}...")
    
    # Load the sentiment data
    sentiment_df = pd.read_csv(sentiment_file)
    
    # Convert datetime to standard format
    sentiment_df['Datetime'] = pd.to_datetime(sentiment_df['Datetime'])
    
    # Extract just the date part (without time) to match with stock data
    sentiment_df['Date'] = sentiment_df['Datetime'].dt.date
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
    
    # For dates with multiple entries, calculate the average sentiment
    agg_functions = {
        'positive': 'mean',
        'neutral': 'mean',
        'negative': 'mean',
        'score': 'mean'
    }
    
    # Group by date and aggregate using the specified functions
    daily_sentiment = sentiment_df.groupby('Date').agg(agg_functions).reset_index()
    
    # Round to 4 decimal places for readability
    for col in ['positive', 'neutral', 'negative', 'score']:
        daily_sentiment[col] = daily_sentiment[col].round(4)
    
    # Save the cleaned data
    output_file = 'cleaned_sentiment_data.csv'
    daily_sentiment.to_csv(output_file, index=False)
    print(f"Cleaned sentiment data saved to {output_file}")
    
    return daily_sentiment

# Step 4: Data Merging
def merge_technical_and_sentiment(stock_file, technical_file, sentiment_df, output_path):
    """
    Merge stock data, technical indicators, and sentiment data.
    
    Parameters:
    - stock_file: Path to the stock price data CSV
    - technical_file: Path to the technical analysis CSV
    - sentiment_df: Dataframe with sentiment data
    - output_path: Path to save the final merged file
    
    Returns:
    - Final merged dataframe
    """
    print("Merging technical and sentiment data...")
    
    # Load stock data and format it
    stock_df = pd.read_csv(stock_file)
    stock_df.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
    stock_df["Date"] = pd.to_datetime(stock_df["Date"])
    
    # Rename columns to match the desired format for merging
    stock_df.rename(columns={
        "Date": "Date_x",
        "open": "open_x",
        "high": "high_x",
        "low": "low_x",
        "close": "close_x",
        "volume": "volume_x"
    }, inplace=True)
    
    # Load technical indicators
    tech_df = pd.read_csv(technical_file)
    tech_df["Date"] = pd.to_datetime(tech_df["Date"])
    
    # Merge stock data with technical indicators
    merged_df = stock_df.merge(tech_df, left_on="Date_x", right_on="Date", how="left")
    
    # Drop the redundant Date column
    if "Date" in merged_df.columns:
        merged_df.drop(columns=["Date"], inplace=True)
    
    # Now merge with sentiment data
    final_df = merged_df.merge(sentiment_df, left_on="Date_x", right_on="Date", how="left")
    
    # Drop redundant Date column from sentiment data
    if "Date" in final_df.columns:
        final_df.drop(columns=["Date"], inplace=True)
    
    # Fill missing sentiment values with neutral values
    # This happens when we have stock data for dates without sentiment data
    if final_df['positive'].isna().any():
        print("Note: Some dates don't have sentiment data. Filling with neutral values.")
        final_df['positive'] = final_df['positive'].fillna(0.0)
        final_df['neutral'] = final_df['neutral'].fillna(0.0)
        final_df['negative'] = final_df['negative'].fillna(0.0)
        final_df['score'] = final_df['score'].fillna(0.0)
    
    # Save the final merged dataset
    final_df.to_csv(output_path, index=False)
    print(f"Final merged data with sentiment saved to {output_path}")
    
    return final_df

# Step 5: Complete Pipeline
def run_full_pipeline(symbol, sentiment_file='apple_sentiment_scores.csv', use_existing_data=False):
    """
    Run the complete data pipeline from API fetch to technical analysis and sentiment merging.
    
    Parameters:
    - symbol: Stock symbol (e.g., 'AAPL', 'MSFT', etc.)
    - sentiment_file: Path to the sentiment CSV file
    - use_existing_data: If True, skip API fetch if data file exists
    
    Returns:
    - Final merged dataframe with technical indicators and sentiment
    """
    print(f"Starting analysis for {symbol} with sentiment data...")
    
    # Step 1: Fetch or load daily data
    data_file = f"{symbol}_daily_data.csv"
    if not use_existing_data or not os.path.exists(data_file):
        print(f"Fetching daily data for {symbol} from Alpha Vantage API...")
        get_daily_data(symbol)
    else:
        print(f"Using existing data file: {data_file}")
    
    # Step 2: Calculate technical indicators
    print(f"Calculating technical indicators for {symbol}...")
    df = preprocess_data(data_file)
    df_with_indicators = calculate_technical_indicators(df)
    
    # Save the technical analysis results
    tech_file = f"{symbol}_technical_analysis.csv"
    df_with_indicators.to_csv(tech_file, index=False)
    print(f"Technical analysis saved to {tech_file}")
    
    # Generate a summary table of recent data
    summary_df = df_with_indicators[[
        "Date", "close", "ATR", "ADX", "OBV", "RSI", "MACD", "Signal_Line", "MACD_Histogram"
    ]].tail(10)
    summary_file = f"{symbol}_summary_table.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary table saved to {summary_file}")
    
    # Step 3: Clean and process sentiment data
    sentiment_df = clean_sentiment_data(sentiment_file)
    
    # Step 4: Merge everything
    final_file = f"{symbol}_with_sentiment.csv"
    final_df = merge_technical_and_sentiment(data_file, tech_file, sentiment_df, final_file)
    
    print(f"Analysis pipeline completed successfully!")
    return final_df

# Change this to analyze different companies
if __name__ == "__main__":
    # Define the ticker symbol to analyze
    TICKER = 'AAPL'  # You can change this to any stock symbol (e.g., 'MSFT', 'AMZN', 'GOOGL')
    
    # Set this to True to use existing data files if available
    USE_EXISTING_DATA = True
    
    # Sentiment data file path
    SENTIMENT_FILE = 'apple_sentiment_scores.csv'
    
    # Run the pipeline
    result = run_full_pipeline(TICKER, sentiment_file=SENTIMENT_FILE, use_existing_data=USE_EXISTING_DATA)
    
    # Display the first few rows of the final dataset
    if result is not None:
        print("\nPreview of final dataset with sentiment:")
        print(result[["Date_x", "close_x", "ATR", "RSI", "MACD", "positive", "negative", "score"]].head())
        
        # Display some statistics
        print("\nSummary statistics of key indicators with sentiment:")
        print(result[["close_x", "ATR", "RSI", "MACD", "score"]].describe())
