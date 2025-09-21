import os
from backend.api import create_app
from backend.config import API_HOST, API_PORT

"""
API entry point
"""
app = create_app()

if __name__ == "__main__":

  server = os.getenv("APP_SERVER", "werkzeug").lower()
  # For Windows
  if server == "waitress":
      from waitress import serve
      serve(app, host=API_HOST, port=API_PORT, threads=8)
  # For UNIX
  else:
      app.run(host=API_HOST, port=API_PORT, debug=True)