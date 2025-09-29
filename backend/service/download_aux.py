import os
import pandas as pd
import requests
from datetime import datetime, timezone, timedelta
from backend.config import DownloadError, ApiKeyError

API_KEY = os.getenv("API_KEY")

def date_str(dt: datetime) -> str:
  """Converts UTC datetime to ISO string"""
  if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    # Allow date ('YYYY-MM-DD')
  return dt.isoformat()

def interval_to_freq(interval: str) -> str:
    """
    Maps intervals to Tiingo resampleFreq.
    Allowed: 1,5,15,60,D
    """
    if interval == "1d":
        return "daily"
    elif interval == "60m":
        return "60min"
    elif interval == "15m":
        return "15min"
    elif interval == "5m":
        return "5min"
    elif interval == "1m":
        return "1min"
    # Default to hourly if unknown
    return "60min"

def period_to_start(period: str, end_dt: datetime) -> datetime:
    """
    Calculates period start date based on yfinance periods.
    Supported: 'ytd', '1mo', '3mo', '6mo'
    """
    if period == "1d":
        return end_dt - timedelta(days=1)
    elif period == "5d":
        return end_dt - timedelta(days=5)
    elif period == "1mo":
        return end_dt - timedelta(days=31)
    elif period == "3mo":
        return end_dt - timedelta(days=93)
    elif period == "6mo":
        return end_dt - timedelta(days=186)
    elif period == "1y":
        return end_dt - timedelta(days=365)
    elif period == "ytd":
        return datetime(end_dt.year, 1, 1, tzinfo=timezone.utc)
    #Default to 1y period is unknown
    return end_dt - timedelta(days=365)

def chunk_ranges(start_dt: datetime, end_dt: datetime, max_days: int) -> list[tuple[datetime, datetime]]:
    """Splits a long time range into API-friendly chunks."""
    chunks: list[tuple[datetime, datetime]] = []
    cur = start_dt
    step = timedelta(days=max_days)
    while cur < end_dt:
        nxt = min(cur + step, end_dt)
        chunks.append((cur, nxt))
        cur = nxt
    return chunks

def build_df(objs: list[dict]) -> pd.DataFrame:
    """
    Creates a OHLCV DataFrame from Tiingo JSON
    """
    if not objs:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    df = pd.DataFrame(objs)
    if "date" not in df.columns:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    ts = pd.to_datetime(df["date"], utc=True)
    out_df = pd.DataFrame({
        "Open":   pd.Series(df.get("open",  pd.Series([], dtype="float64")).to_numpy(), index=ts, dtype="float64"),
        "High":   pd.Series(df.get("high",  pd.Series([], dtype="float64")).to_numpy(), index=ts, dtype="float64"),
        "Low":    pd.Series(df.get("low",   pd.Series([], dtype="float64")).to_numpy(), index=ts, dtype="float64"),
        "Close":  pd.Series(df.get("close", pd.Series([], dtype="float64")).to_numpy(), index=ts, dtype="float64"),
        "Volume": pd.Series(df.get("volume",pd.Series([], dtype="float64")).to_numpy(), index=ts, dtype="float64"),
    })
    out_df.sort_index(inplace=True)
    return out_df

def tiingo_fetch(
    ticker: str,
    end_dt: datetime,
    period: str,
    interval: str,
    timeout: float = 15.0,
) -> pd.DataFrame:
    """
    Fetch candles from Tiingo and return OHLCV DataFrame in UTC.
    Uses Intraday for minute/hourly, and Tiingo Daily for daily.
    """
    resample = interval_to_freq(interval)
    start_dt = period_to_start(period, end_dt)

    if API_KEY is None:
        raise ApiKeyError("API key not set or invalid")

    headers = {"Authorization": f"Token {API_KEY}"}
    params_base = {
        # Only return OHLCV fields
        "columns": "open,high,low,close,volume",
    }

    # Daily
    if resample == "daily":
        url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
        params = dict(params_base)
        params["startDate"] = date_str(start_dt.date())
        params["endDate"] = date_str(end_dt.date())
        resp = requests.get(url, headers=headers, params=params, timeout=timeout)
        if resp.status_code == 200:
            return build_df(resp.json())
        elif resp.status_code == 429 or 500 <= resp.status_code < 600:
            raise DownloadError(f"Tiingo daily transient HTTP {resp.status_code}: {resp.text[:200]}")
        # Non-transient → empty
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    # Intraday: Chunk by 7 days to keep payloads small and avoid timeouts
    url = f"https://api.tiingo.com/iex/{ticker}/prices"
    chunks = chunk_ranges(start_dt, end_dt, max_days=7)
    dfs: list[pd.DataFrame] = []

    for (a, b) in chunks:
        params = dict(params_base)
        # Send date only for intraday
        params["startDate"] = a.date().isoformat()
        params["endDate"] = b.date().isoformat()
        params["resampleFreq"] = resample

        resp = requests.get(url, headers=headers, params=params, timeout=timeout)
        if resp.status_code == 200:
            dfs.append(build_df(resp.json()))
        elif resp.status_code in (401, 403):
            raise DownloadError(f"Tiingo IEX auth/plan error {resp.status_code}: {resp.text[:300]}")
        elif resp.status_code == 429 or 500 <= resp.status_code < 600:
            raise DownloadError(f"Tiingo intraday transient HTTP {resp.status_code}: {resp.text[:300]}")
        else:
            print(f"IEX chunk {a}→{b} returned HTTP {resp.status_code}: {resp.text[:200]}")
            dfs.append(pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]))

    if not dfs:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    out_df = pd.concat(dfs).sort_index()
    # Remove overlapping edges
    out_df = out_df[~out_df.index.duplicated(keep="last")]
    return out_df