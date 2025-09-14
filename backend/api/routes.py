from flask import Blueprint, jsonify, request
from backend.service.candles import get_candles, serialize_candles
from backend.service.preprocessing import build_input
from backend.service.model_inference import pass_input

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
  if ticker_df is None or ticker_df.empty: return jsonify({"error": f"No data for {t}"}), 404

  # Returns JSON object of ticker data
  return jsonify({"ticker": t, "data": serialize_candles(ticker_df)})

@api_bp.get("/prediction")
def predict():
  t = (request.args.get("ticker") or "").upper().strip()
  if not t: jsonify({"error": "Missing ticker"}), 400

  x = build_input(t) # Get input Tensor from ticker DataFrame
  # Return HTTP 404 if no ticker data was obtained
  if x is None: return jsonify({"error": f"No data for {t}"}), 404

  res = pass_input(x) # Pass Tensor to the model and get prediction data
  return jsonify({"ticker": t, **res}) # Return data as a json