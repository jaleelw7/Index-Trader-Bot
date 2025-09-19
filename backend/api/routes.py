import re
from flask import Blueprint, jsonify, request
from backend.service.candles import get_candles, serialize_candles
from backend.service.preprocessing import build_input
from backend.service.model_inference import pass_input

api_bp = Blueprint("api", __name__) # API route namespace

# Endpoint input format
TICKER_FORMAT = re.compile(r"^[A-Z\.]{1,10}$")
ALLOWED_INTERVAL = {"1m","5m","15m","60m","1d"}
ALLOWED_PERIOD = {"7d","30d","90d","180d"}

@api_bp.get("/candles")
def candles():
  """
  API endpoint for getting ticker data for display
  """
  # Query parameters: ticker, interval, and period for yf.download()
  t = (request.args.get("ticker") or "").upper().strip()
  interval = request.args.get("interval", "60m")
  period = request.args.get("period", "30d")
  # Returns HTTP 400 if no ticker was given
  if not t: return jsonify({"error": "Missing ticker"}), 400
  # Rejects ticker input if not in the ticker format
  if not TICKER_FORMAT.fullmatch(t): return jsonify({"error": "Invalid ticker"}), 400
  # Rejects interval input if not in the allowed intervals
  if interval not in ALLOWED_INTERVAL: return jsonify({"error": "Invalid interval"}), 400
  # Rejects period input if not in the allowed intervals
  if interval not in ALLOWED_PERIOD: return jsonify({"error": "Invalid period"}), 400

  ticker_df = get_candles(t, interval, period) # Download ticker data
  # Returns HTTP 404 if no ticker data was obtained
  if ticker_df is None or ticker_df.empty: return jsonify({"error": f"No data for {t}"}), 404

  # Returns JSON object of ticker data
  return jsonify({"ticker": t, "data": serialize_candles(ticker_df)})

@api_bp.get("/prediction")
def predict():
  """
  Endpoint for getting model predictions
  """
  t = (request.args.get("ticker") or "").upper().strip()
  if not t: jsonify({"error": "Missing ticker"}), 400
  if not TICKER_FORMAT.fullmatch(t): return jsonify({"error": "Invalid ticker"}), 400

  x = build_input(t) # Get input Tensor from ticker DataFrame
  # Return HTTP 404 if no ticker data was obtained
  if x is None: return jsonify({"error": f"No data for {t}"}), 404

  res = pass_input(x) # Pass Tensor to the model and get prediction data
  return jsonify({"ticker": t, **res}) # Return data as a json

@api_bp.get("/healthz")
def health():
  """
  Health endpoint
  """
  return jsonify({"status": "ok"}), 200