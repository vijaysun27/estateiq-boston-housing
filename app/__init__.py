"""Flask Application Factory for EstateIQ (Phase 4 Hardened)."""

import logging
import os
from flask import Flask, Response, jsonify, render_template, request
from app.config import config_by_name
from app.services.predictor import PredictorService


def create_app(config_name: str = "default") -> Flask:
    """Construct and configure the Flask application instance."""
    app = Flask(__name__)

    # Load configuration
    config_obj = config_by_name.get(config_name, config_by_name["default"])
    app.config.from_object(config_obj)

    # Configure logging
    log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    )

    # Initialize predictor service (loads model once per worker process, never retrains)
    predictor = PredictorService(
        pipeline_path=app.config.get("MODEL_PIPELINE_PATH"),
        keras_path=app.config.get("KERAS_MODEL_PATH"),
        metadata_path=app.config.get("METADATA_PATH"),
    )
    predictor.load_model()
    app.predictor = predictor

    # Register blueprints
    from app.routes.main import main_bp
    from app.routes.prediction import prediction_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(prediction_bp)

    # Register error handlers and security headers
    register_error_handlers(app)
    register_security_headers(app)

    return app


def register_security_headers(app: Flask) -> None:
    """Attach baseline HTTP security headers to all responses."""

    @app.after_request
    def set_headers(response: Response) -> Response:
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "font-src 'self' https://fonts.gstatic.com; "
            "style-src 'self' https://fonts.googleapis.com 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self';"
        )
        return response


def register_error_handlers(app: Flask) -> None:
    """Register HTTP error handling templates and API JSON fallbacks."""

    @app.errorhandler(400)
    def bad_request_error(error):
        if request.path.startswith("/api/"):
            return jsonify({"success": False, "error": "Bad request"}), 400
        return render_template("errors/404.html"), 400

    @app.errorhandler(404)
    def not_found_error(error):
        if request.path.startswith("/api/"):
            return jsonify({"success": False, "error": "Endpoint not found"}), 404
        return render_template("errors/404.html"), 404

    @app.errorhandler(405)
    def method_not_allowed_error(error):
        if request.path.startswith("/api/"):
            return jsonify({"success": False, "error": "Method not allowed for this endpoint"}), 405
        return render_template("errors/404.html"), 405

    @app.errorhandler(413)
    def payload_too_large_error(error):
        if request.path.startswith("/api/"):
            return jsonify({
                "success": False,
                "error": "Request payload exceeds maximum allowed size (16 KB)."
            }), 413
        return render_template("errors/500.html"), 413

    @app.errorhandler(500)
    def internal_error(error):
        if request.path.startswith("/api/"):
            return jsonify({"success": False, "error": "Internal server error"}), 500
        return render_template("errors/500.html"), 500
