# ---------------------------------------------------------------------------
# app.py — Application factory and entry point
# ---------------------------------------------------------------------------

from flask import Flask

from core.routes import main


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.register_blueprint(main)
    return app


if __name__ == "__main__":
    create_app().run(debug=True)