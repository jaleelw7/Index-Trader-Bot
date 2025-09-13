from flask import Flask
from flask_cors import CORS
from .routes import api_bp
from .error_handlers import register_error_hadlers

def create_app() -> Flask:
  """
  Application factory
  """
  app = Flask(__name__)
  CORS(app) # Configure origins
  app.register_blueprint(api_bp, url_prefix="/api")
  register_error_hadlers(app)
  return app
