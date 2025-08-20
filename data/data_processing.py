import yfinance as yf
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_download import download_data


"""
Normalizes features using rolling z-score. The z-score is computed using the mean
and standard deviation over the number previous intervals given by window_len
"""
def zscore_norm(df: pd.DataFrame, features: list[str], window_size: int = 31) -> pd.DataFrame:
  norm_df = df.copy()
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
  with ThreadPoolExecutor(max_workers=3) as exe:
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

test_df = build_dataset()
print(test_df)