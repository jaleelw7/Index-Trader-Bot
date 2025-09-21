from flask import Flask
from flask_cors import CORS
from .routes import api_bp
from .error_handlers import register_error_hadlers
from backend.config import CORS_ORIGINS

def create_app() -> Flask:
  """
  Application factory
  """
  app = Flask(__name__)
  CORS(app, origins=CORS_ORIGINS) # Configure origins
  app.register_blueprint(api_bp, url_prefix="/api")
  register_error_hadlers(app)
  return app
