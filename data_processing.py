import yfinance as yf
import pandas as pd
import talib
import time


"""
Method to verify if a ticker exists
"""
def ticker_exists(ticker: str) -> bool:
  try:
    info = yf.Ticker(ticker).info
    return "shortName" in info and bool(info["shortName"])
  except Exception:
    return False
  
"""
Method to download the data from each index
"""
def download_data(ticker: str) -> pd.DataFrame:
  if not ticker_exists(ticker):
    raise ValueError(f"{ticker} does not exist.")