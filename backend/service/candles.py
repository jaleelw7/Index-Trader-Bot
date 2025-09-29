import pandas as pd
import time
import threading
from backend.service.ticker_download import download_data

_CACHE: dict[tuple, tuple[float, pd.DataFrame | None]] = {}
_CACHE_LOCK = threading.Lock()

def get_candles(ticker: str, interval: str = "60m", period: str = "1mo") -> pd.DataFrame | None:
  """
  Downloads ticker data for frontend display
  """
  """ Check cache before downloading ticker dataframe """
  k = (ticker, interval, period)
  # Set cache time based on interval
  if interval == "1d":
    ttl = 1800
  elif interval == "60m":
    ttl = 300
  else:
    ttl = 90
  now = time.time()
  
  with _CACHE_LOCK:
    cache_hit = _CACHE.get(k)
    if cache_hit and cache_hit[0] > now:
      return cache_hit[1]

  df = download_data(ticker, interval=interval, period=period)
  if df is None or df.empty:
    with _CACHE_LOCK:
      _CACHE[k] = (now + ttl, None)
      return df # Prevents errors from operating on empty or null DataFrame

  df.reset_index(inplace=True) # Replace Datetime index with numerical index
  # Rename date column to timestamp
  if "date" in df.columns:
    df = df.rename(columns={"date": "timestamp"})
  else:
    raise KeyError("No datetime/date column found")
     
  df = df.sort_values("timestamp") # Sort dataframe by timestamp
  display_features = ["timestamp", "Open", "High", "Low", "Close", "Volume"]

  with _CACHE_LOCK:
    _CACHE[k] = (now+ttl, df[display_features].dropna()) # Cache return dataframe
  return df[display_features].dropna()

def serialize_candles(df: pd.DataFrame) -> list[dict]:
  """
  Converts candle data to JSON compatible dictionaries
  """
  return [
    {
      "ts": r.timestamp.tz_localize(None).isoformat(),
      "open": float(r.Open),
      "high": float(r.High),
      "low": float(r.Low),
      "close": float(r.Close),
      "volume": float(r.Volume)
    } for r in df.itertuples(index=False)
  ]