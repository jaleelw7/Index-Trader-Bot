"""
Constants for use across backend
"""

# Model contract
FEATURES = ["Open", "High", "Low", "Close", "Volume", "rsi", "ema", "atr_pct"]
WINDOW_SIZE = 96
CLASS_ORDER = ["Sell", "Hold", "Buy"]

# API
API_HOST = "0.0.0.0"
API_PORT = 8000
CORS_ORIGINS = "*"
CACHE_TIME = 300 # How long ticker data is cached