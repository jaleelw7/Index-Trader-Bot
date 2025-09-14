import torch
import numpy as np
from data.data_processing import build_dataset
from backend.config import FEATURES, WINDOW_SIZE


def build_input(ticker: str) -> torch.Tensor:
  """
  Builds input tensor for the model
  """
  # Get dataframe for ticker with training indicators and normalization applied (interval= 1 hour, period= 2 years)
  df = build_dataset(tickers=[ticker], features=FEATURES, single_ticker=True)
  # Return None if no data was retrieved
  if df is None or df.empty: return None

  x = df[FEATURES].tail(WINDOW_SIZE).to_numpy(dtype=np.float32) # Get last time window
  x = torch.from_numpy(x).unsqueeze(0) # Convert the time window to a torch Tensor
  return x