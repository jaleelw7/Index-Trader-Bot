import yfinance as yf
import numpy as np
import pandas as pd
import talib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


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
  if(isinstance(ticker_df.columns, pd.MultiIndex)):
    ticker_df.columns = ticker_df.columns.get_level_values(0)
  
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
  
"""
Normalizes features using rolling z-score. The z-score is computed using the mean
and standard deviation over the number previous intervals given by window_len
"""
def zscore_norm(df: pd.DataFrame, features: list[str], window_size: int = 31) -> pd.DataFrame:
  norm_df = df.opy()
  grouped_df = df.groupby("ticker", group_keys=False)

  for f in features:
    rolling_mean = grouped_df[f].transform(lambda x: x.rolling(window_size, min_periods=window_size).mean())
    rolling_std = grouped_df[f].transform(lambda x: x.rolling(window_size, min_periods=window_size).std(ddof=0))
    #A very small constant is added to the std to prevent division by zero errors
    norm_df[f] = (norm_df[f] - rolling_mean) / (rolling_std + 1e-9)
  
  return norm_df.dropna()

"""
Method to combine the dataframes from multiple tickers into the complete dataset.
"""
def build_dataset(tickers: list[str] = None, features: list[str] = None) -> pd.DataFrame:
  #Default tickers and input features
  if tickers is None:
    tickers = ["SPY", "QQQ", "DIA", "IWM", "VTI"]
  if features is None:
    features = ["Open", "High", "Low", "Close", "Volume", "rsi", "ema", "atr_pct"]
  
  #Download ticker data in parallel instead of sequentially to reduce wait time
  ticker_dfs = []
  with ThreadPoolExecutor(max_workers=5) as exe:
    futures = {exe.submit(download_data, t): t for t in tickers}
    for f in as_completed(futures):
      ticker_dfs.append(f.result())
  
  #Combine the list of ticker DataFrames into one DataFrame
  complete_df = pd.concat(ticker_dfs, axis=0, ignore_index=False)
  #Maintain chronological order by sorting by datetime index
  complete_df.sort_index(inplace=True)
  #Replace the datetime index with a numerical index
  complete_df.reset_index(drop=True, inplace=True)

  #Normalize input features
  complete_df = zscore_norm(complete_df, features)

  return complete_df
