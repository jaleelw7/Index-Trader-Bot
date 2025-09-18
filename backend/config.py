"""
Constants for use across backend
"""

# Model contract
FEATURES = ["Open", "High", "Low", "Close", "Volume", "rsi", "ema", "atr_pct"]
WINDOW_SIZE = 96
CLASS_ORDER = ["sell", "hold", "buy"]

# API
API_HOST = "0.0.0.0"
API_PORT = 8000
CORS_ORIGINS = "*"