from flask import jsonify
from backend.config import ApiKeyError, DownloadError

def register_error_hadlers(app):
  """
  Function to create and register API error handlers
  """
  @app.errorhandler(404)
  def not_found(_):
    return jsonify({"error": "Not found"}), 404
  
  @app.errorhandler(500)
  def server_error(e):
    return jsonify({"error": "Server error", "details": str(e)}), 500
  
  @app.errorhandler(ApiKeyError)
  def api_key_error(_):
    return jsonify({"error": "API key invalid or missing."}), 500
  
  @app.errorhandler(DownloadError)
  def tiingo_download_error(e: DownloadError):
    return jsonify({"error": "Tiingo download failed", "details": str(e)}), 500
  
  @app.errorhandler(KeyError)
  def df_key_error(_):
    return jsonify({"error": "DataFrame column error"}), 500