"""
Constants for use across backend
"""
import torch
import os
from pathlib import Path

class ApiKeyError(RuntimeError):
    """Raised when FINNHUB_API_KEY is not configured on the server."""

class DownloadError(RuntimeError):
    """Raised in place of RuntimeError when downloading from Tiingo"""

# Model contract
FEATURES = ["Open", "High", "Low", "Close", "Volume", "rsi", "ema", "atr_pct"]
WINDOW_SIZE = 96
CLASS_ORDER = ["sell", "hold", "buy"]


DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Model device
# Saved model path
ROOT = Path(__file__).resolve().parent.parent  
DEFAULT_DIR = ROOT / "artifacts" / "models"
SAVE_DIR = Path(os.getenv("SAVE_DIR", DEFAULT_DIR))
SAVE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_FILE = "index_tcn_v1.pth"
SAVE_PATH = SAVE_DIR / MODEL_FILE

# API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", os.getenv("API_PORT", 8000)))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "")