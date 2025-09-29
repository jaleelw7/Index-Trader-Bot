import numpy as np
import pandas as pd
import talib
import time
import threading
import requests
from datetime import datetime, timezone
from backend.service.download_aux import tiingo_fetch

# Constants used in download_data
THRESHOLD = 0.005 #Hourly return threshold
RSI_PERIOD = 14 #timeperiod parameter for RSI
EMA_PERIOD = 21 #timeperiod parameter for EMA
ATR_PERIOD = 14 #timeperiod parameter for ATR

_CACHE: dict[tuple, tuple[float, pd.DataFrame | None]] = {} # Cache for downloads
_CACHE_LOCK =  threading.Lock() # Lock for thread safe caching

def is_transient(exc: Exception) -> bool:
    e = str(exc)
    return ("Too Many Requests" in e or "Read timed out" in e or
            "Connection aborted" in e or "Connection reset" in e or
            "timed out" in e or "ECONNRESET" in e)

def download_data(
    ticker: str,
    max_retries: int = 2,
    backoff: float = 2.0,
    interval: str = "60m",
    period: str = "1mo",
) -> pd.DataFrame | None:
    """
    Downloads candles from Tiingo and computes indicators
    """
    k = (ticker, interval, period) # Cache key

    # Set TTL based on interval
    if interval == "1d":
        ttl = 1800
    elif interval in ("60m", "30m"):
        ttl = 300
    else:
        ttl = 90

    now = time.time()
    with _CACHE_LOCK:
        cache_hit = _CACHE.get(k)
        if cache_hit and cache_hit[0] > now:
            return None if cache_hit[1] is None else cache_hit[1].copy()

    end_dt = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    # Retry loop
    for attempt in range(1, max_retries + 1):
        try:
            df = tiingo_fetch(ticker, end_dt, period, interval)

            # Early return/cache on empty
            if df is None or df.empty:
                with _CACHE_LOCK:
                    _CACHE[k] = (now + ttl, None)
                return df

            # Add ticker column
            df["ticker"] = ticker

            # TA-Lib indicators
            df["rsi"] = talib.RSI(df["Close"], timeperiod=RSI_PERIOD)
            df["ema"] = talib.EMA(df["Close"], timeperiod=EMA_PERIOD)
            df["atr"] = talib.ATR(df["High"], df["Low"], df["Close"], timeperiod=ATR_PERIOD)

            # ATR as % close, clipped
            df["atr_pct"] = (df["atr"] / df["Close"]).clip(0, 0.05)

            # Future return over next 3 intervals
            df["return"] = (df["Close"].shift(-3) / df["Close"]) - 1.0
            df["return"] = df["return"].clip(-0.1, 0.1)

            # Classification labels (2 = Buy, 1 = Hold, 0 = Sell)
            df["return_label"] = np.where(
                df["return"] > THRESHOLD, 2,
                np.where(df["return"] < -THRESHOLD, 0, 1)
            )

            # Drop NA rows
            df.dropna(inplace=True)

            # Convert numerical data to float32 for pytorch
            for col in ["Open", "High", "Low", "Close", "Volume", "rsi", "ema", "atr", "atr_pct", "return"]:
                df[col] = df[col].astype("float32")

            with _CACHE_LOCK:
                _CACHE[k] = (now + ttl, df.copy())
            return df
        
        except requests.RequestException as e:
            # Network/HTTP errors
            if attempt < max_retries:
                sleep_time = backoff * (2 ** (attempt - 1))
                print(f"{ticker}: Tiingo request failed ({e}). Retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)

        except Exception as e:
            if attempt < max_retries and is_transient(e):
                sleep_time = backoff * (2 ** (attempt - 1))
                print(f"{ticker}: Tiingo request failed ({e}). Retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)
            else:
                print(f"{ticker}: All Tiingo requests failed. Last error: {e}")
                with _CACHE_LOCK:
                    _CACHE[k] = (now + ttl, None)
                return None