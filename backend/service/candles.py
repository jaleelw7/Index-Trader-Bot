import pandas as pd
from functools import lru_cache
from data.data_download import download_data

@lru_cache(maxsize=128)
def get_candles(ticker: str, interval: str = "30m", period: str = "1mo") -> pd.DataFrame:
  """
  Downloads ticker data for frontend display
  """
  df = download_data(ticker, interval=interval, period=period, for_train=False)
  if df is None or df.empty: return df  # Prevents errors from operating on empty or null DataFrame

  df.reset_index(inplace=True) # Replace Datetime index with numerical index
  df.sort_values("Datetime", inplace=True) # Sort dataframe by datetime
  display_features = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
  return df[display_features].dropna()

def serialize_candles(df: pd.DataFrame) -> list[dict]:
  """
  Converts candle data to JSON compatible dictionaries
  """
  return [
    {
      "ts": r.Datetime.isoformat(),
      "open": float(r.Open),
      "high": float(r.High),
      "low": float(r.Low),
      "close": float(r.Close),
      "volume": float(r.Volume)
    } for r in df.itertuples(index=False)
  ]