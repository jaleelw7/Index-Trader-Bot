import torch
import numpy as np
import pandas as pd
from backend.config import FEATURES, WINDOW_SIZE
from backend.service.ticker_download import download_data

def normalize_input(t: str) -> str:
  """
  Normalizes ticker input
  """
  return t.upper().replace(".", "-").strip()

def zscore_norm(df: pd.DataFrame, window_size: int = WINDOW_SIZE) -> pd.DataFrame:
  norm_df = df.copy()
  norm_df.sort_values(["ticker", "date"], inplace=True)
  grouped_df = df.groupby("ticker", group_keys=False)

  for f in FEATURES:
    rolling_mean = grouped_df[f].transform(lambda x: x.rolling(window_size, min_periods=window_size).mean())
    rolling_std = grouped_df[f].transform(lambda x: x.rolling(window_size, min_periods=window_size).std(ddof=0))
    #A very small constant is added to the std to prevent division by zero errors
    norm_df[f] = (norm_df[f] - rolling_mean) / (rolling_std + 1e-9)
    norm_df[f] = norm_df[f].shift(1)
  
  return norm_df.dropna()

def build_input(ticker: str) -> torch.Tensor:
  """
  Builds input tensor for the model
  """
  df = download_data(ticker)
  #Maintain chronological order by sorting by datetime index
  df.sort_index(inplace=True)
  #Replace the datetime index with a numerical index
  df.reset_index(inplace=True)

  #Normalize input features
  df = zscore_norm(df)

  x = df[FEATURES].tail(WINDOW_SIZE).to_numpy(dtype=np.float32) # Get last time window
  x = torch.from_numpy(x).unsqueeze(0) # Convert the time window to a torch Tensor
  return x