from backend.api import create_app
from backend.config import API_HOST, API_PORT
from waitress import serve

"""
API entry point
"""
app = create_app()

if __name__ == "__main__":
  serve(app, host=API_HOST, port=API_PORT, threads=8)