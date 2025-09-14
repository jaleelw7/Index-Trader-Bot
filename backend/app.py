from api import create_app
from config import API_HOST, API_PORT

"""
API entry point
"""
app = create_app()

if __name__ == "__main__":
  app.run(host=API_HOST, port=API_PORT, debug=True)