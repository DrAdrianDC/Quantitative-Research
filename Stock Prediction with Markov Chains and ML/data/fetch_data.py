
# This script fetches historical stock data using the yfinance library and calculates daily returns.

# Import libraries

import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data using yfinance.
    
    Parameters:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
    
    Returns:
        pd.DataFrame: DataFrame containing stock data with daily returns.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Daily Return'] = data['Close'].pct_change()
    data.dropna(inplace=True)
    data.to_csv('stock_data.csv')
    return data

if __name__ == "__main__":
    # Example usage
    stock_data = fetch_stock_data(ticker="AAPL", start_date="2014-01-01", end_date="2024-12-04")
    print(stock_data.head())
