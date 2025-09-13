from flask import Blueprint, jsonify, request
from backend.service.candles import get_candles, serialize_candles
from backend.service.model_inference import build_input, pass_input
from backend.config import WINDOW_SIZE
from data.data_processing import build_dataset

api_bp = Blueprint("api", __name__) # API route namespace

@api_bp.get("/candles")
def candles():
  # Query parameters: ticker, interval, and period for yf.download()
  t = (request.args.get("ticker") or "").upper().strip()
  interval = request.args.get("interval", "60m")
  period = request.args.get("period", "30d")
  # Returns HTTP 400 if no ticker was given
  if not t: return jsonify({"error": "Missing ticker"}), 400
  
  ticker_df = get_candles(t, interval, period) # Download ticker data
  # Returns HTTP 404 if no ticker data was obtained
  if not ticker_df or ticker_df.empty: return jsonify({"error": f"No data for {t}"}), 404

  # Returns JSON object of ticker data
  return jsonify({"ticker": t, "data": serialize_candles(ticker_df)})