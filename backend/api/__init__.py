from flask import Flask
from flask_cors import CORS
from .routes import api_bp
from .error_handlers import register_error_hadlers
from backend.config import CORS_ORIGINS

def create_app() -> Flask:
  """
  application factory
  """
  application = Flask(__name__)
  origins = origins = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]
  CORS(application, resources={r"/api/*": {"origins": origins}}) # Configure origins
  application.register_blueprint(api_bp, url_prefix="/api")
  register_error_hadlers(application)
  return application
