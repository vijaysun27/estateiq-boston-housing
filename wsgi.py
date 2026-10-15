"""WSGI entrypoint for EstateIQ application."""

import os
from dotenv import load_dotenv
from app import create_app

# Load environment variables from .env if present
load_dotenv()

env_name = os.getenv("FLASK_ENV", "production")
app = create_app(config_name=env_name)

if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "0") in ("1", "true", "True")
    app.run(host=host, port=port, debug=debug)
