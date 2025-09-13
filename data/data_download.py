import yfinance as yf
import numpy as np
import pandas as pd
import talib
import time
from datetime import datetime, timezone, timedelta

#Constants used in download_data
THRESHOLD = 0.005 #Hourly return threshold
RSI_PERIOD = 14 #timeperiod parameter for RSI
EMA_PERIOD = 21 #timeperiod parameter for EMA
ATR_PERIOD = 14 #timeperiod parameter for ATR
END = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
START = END - timedelta(days=729)

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
Method to download the data from each index. For loop is added to retry downloads in case
of yfinance failure.
"""
def download_data(ticker: str,
                  max_retries: int = 3,
                  backoff: float = 2.0,
                  interval: str = "60m",
                  period: str = "ytd",
                  for_train: bool =True) -> pd.DataFrame:
  
  #Checks if ticker exists, raises exception if false
  if not ticker_exists(ticker):
    raise ValueError(f"Ticker: {ticker} does not exist.")
  
  #Download data for each ticker individually
  for attempt in range(1, max_retries + 1):
    try:
      if for_train:
        ticker_df = yf.download(ticker,
                                start=START,
                                end=END,
                                interval="60m",
                                progress=False,
                                threads=False)
      
      else:
        ticker_df = yf.download(ticker,
                                period=period,
                                interval=interval,
                                progress=False,
                                threads=False)
       
      #Flattens DataFrame columns if multi-index
      ticker_df = ticker_df.xs(ticker, axis=1, level=1)

      #Raises an exception if no data was downloaded for ticker
      if ticker_df is None or ticker_df.empty:
        raise ValueError(f"No data found for {ticker}.")
      
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
      for col in ["Open", "High", "Low", "Close", "Volume", "rsi", "ema", "atr", "atr_pct", "return"]:
          ticker_df[col] = ticker_df[col].astype("float32")
      
      return ticker_df
    
    #Catches any connectivity or download errors with yfinance
    except Exception as e:
      #If the number of attempts is less than max_retries, attempt download again after sleeping
      if attempt < max_retries:
        sleep_time = backoff ** (attempt - 1)
        print(f"{ticker}: Download attempt failed ({e}). Retrying download in {sleep_time}s...")
        time.sleep(sleep_time)
      #If the max number of retries is reached, print error message and raise the exception
      else:
        print(f"{ticker}: All download attempts failed.")
        raise e