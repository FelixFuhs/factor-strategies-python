import yfinance as yf
import pandas as pd
import numpy as np

def load_data(tickers, start_date, end_date):
    """
    Load historical stock data from Yahoo Finance.
    
    Parameters:
    tickers (list): List of stock tickers.
    start_date (str): Start date in 'YYYY-MM-DD' format.
    end_date (str): End date in 'YYYY-MM-DD' format.
    
    Returns:
    pd.DataFrame: DataFrame with stock prices.
    """
    data = yf.download(tickers, start=start_date, end=end_date,auto_adjust=True)['Close']
    return data

def calculate_returns(data):
    """
    Calculate daily returns from stock prices.
    
    Parameters:
    data (pd.DataFrame): DataFrame with stock prices.
    
    Returns:
    pd.DataFrame: DataFrame with daily returns.
    """
    returns = data.pct_change().dropna()
    return returns

def calculate_log_returns(data):
    """
    Calculate log returns from stock prices.
    
    Parameters:
    data (pd.DataFrame): DataFrame with stock prices.
    
    Returns:
    pd.DataFrame: DataFrame with log returns.
    """
    log_returns = np.log(data / data.shift(1)).dropna()
    return log_returns

