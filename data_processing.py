import yfinance as yf
import numpy as np
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
  THRESHOLD = 0.005 #Hourly return threshold
  RSI_PERIOD = 14 #timeperiod parameter for RSI
  EMA_PERIOD = 21 #timeperiod parameter for EMA
  ATR_PERIOD = 14 #timeperiod parameter for ATR

  #Checks if ticker exists, raises exception if false
  if not ticker_exists(ticker):
    raise ValueError(f"{ticker} does not exist.")
  
  #Download data for each ticker individually
  ticker_df = yf.download(ticker,
                          interval="60m",
                          period="1y",
                          progress=False,
                          threads=False)
  
  #Raises an exception if no data was downloaded for ticker
  if ticker_df is None or ticker_df.empty:
    raise ValueError(f"No data found for {ticker}.")
  
  #Flattens DataFrame columns if multi-index
  if(isinstance(ticker_df, pd.MultiIndex)):
    ticker_df.columns = ticker_df.columns.get_level_values(0)
  
  #Filter out 
  
  #Add a ticker column to the DataFrame
  ticker_df["ticker"] = ticker
  #Sort DataFrame chronologically
  ticker_df.sort_index(inplace=True)

  #Create columns for TA-Lib technical indicators
  ticker_df["rsi"] = talib.RSI(ticker_df["Close"], timeperiod=RSI_PERIOD)
  ticker_df["ema"] = talib.EMA(ticker_df["Close"], timeperiod=EMA_PERIOD)
  ticker_df["atr"] = talib.ATR(ticker_df["High"], ticker_df["Low"], ticker_df["Close"], timeperiod=ATR_PERIOD)

  #Give ATR as a percentage of Close price
  ticker_df["atr_pct"] = ticker_df["atr"] / ticker_df["Close"]
  #Clip extreme ATR percentage values
  ticker_df["atr_pct"] = ticker_df["atr_pct"].clip(0, 0.05)

  #Calculate future return over the next 3 intervals and clip extreme values
  ticker_df["return"] = (ticker_df['Close'].shift(-3) / ticker_df['Close']) - 1
  ticker_df['return'] = ticker_df['return'].clip(-0.1, 0.1)
  """Apply classification labels based on return value:
     2: Buy (return > threshold),
     1: Hold (-threshold <= return <= threshold),
     0: Sell (return < -threshold)"""
  ticker_df["return_label"] = np.where(ticker_df["return"] > THRESHOLD, 2,
                                       np.where(ticker_df["return"] < -THRESHOLD, 0, 1))
  
  #Drop rows with missing values
  ticker_df.dropna(inplace=True)

  """
  Converting certain features to float32 for memory and performance efficiency,
  compatability with Pytorch tensors
  """
  for col in ["Open", "High", "Low", "Close", "Volume", "rsi", "ema", "atr", "return"]:
      ticker_df[col] = ticker_df[col].astype("float32")
  
  return ticker_df
    