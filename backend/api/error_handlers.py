from flask import jsonify

def register_error_hadlers(app):
  """
  Function to create and register API error handlers
  """
  @app.errorhandler(404)
  def not_found(_):
    return jsonify({"error:" "Not found"}), 404
  
  @app.errorhandler(500)
  def server_error(e):
    return jsonify({"error": "Server error", "details": str(e)}), 500