"""API endpoints for prediction, health status, and sample data."""

import logging
from flask import Blueprint, current_app, jsonify, request
from app.services.predictor import ModelNotAvailableError
from app.utils.validation import (
    FEATURE_SPECIFICATIONS,
    validate_input_features,
)

logger = logging.getLogger(__name__)
prediction_bp = Blueprint("prediction", __name__, url_prefix="/api")


@prediction_bp.route("/health", methods=["GET"])
def health():
    """Return application health and model readiness without exposing filesystem paths or secrets."""
    predictor = getattr(current_app, "predictor", None)
    if predictor is None:
        return (
            jsonify({
                "status": "model_unavailable",
                "application": "running",
                "model_ready": False,
                "prediction_service": "unavailable",
                "message": "Predictor service not initialized",
            }),
            503,
        )

    is_ready = predictor.is_ready()
    status_code = 200 if is_ready else 503

    return (
        jsonify({
            "status": "healthy" if is_ready else "model_unavailable",
            "application": "running",
            "model_ready": is_ready,
            "prediction_service": "available" if is_ready else "unavailable",
            "model_info": predictor.get_model_info(),
        }),
        status_code,
    )


@prediction_bp.route("/sample", methods=["GET"])
def sample():
    """Return a representative sample input for quick demonstration."""
    sample_features = {
        name: spec["default"] for name, spec in FEATURE_SPECIFICATIONS.items()
    }
    return jsonify({
        "success": True,
        "sample": sample_features,
        "description": "Representative suburban Boston census tract benchmark observation",
    }), 200


@prediction_bp.route("/predict", methods=["POST"])
def predict():
    """Accept property features, validate them, and generate valuation prediction."""
    predictor = getattr(current_app, "predictor", None)
    if predictor is None or not predictor.is_ready():
        return (
            jsonify({
                "success": False,
                "error": "Prediction service is currently unavailable. Trained model artifact is not loaded.",
            }),
            503,
        )

    # Validate JSON payload header and structure
    if not request.is_json:
        return (
            jsonify({
                "success": False,
                "error": "Request body must be valid JSON with 'application/json' Content-Type header.",
            }),
            400,
        )

    raw_data = request.get_json(silent=True)
    if raw_data is None:
        return (
            jsonify({
                "success": False,
                "error": "Malformed or empty JSON payload.",
            }),
            400,
        )

    is_valid, validation_result = validate_input_features(raw_data)
    if not is_valid:
        return (
            jsonify({
                "success": False,
                "error": validation_result.get("error", "Validation error"),
                "details": validation_result.get("field_errors", {}),
            }),
            400,
        )

    try:
        prediction_result = predictor.predict(validation_result)
        return (
            jsonify({
                "success": True,
                "data": prediction_result,
            }),
            200,
        )
    except ModelNotAvailableError as e:
        logger.error("Model unavailable during prediction: %s", e)
        return (
            jsonify({
                "success": False,
                "error": "Model artifact is unavailable to process the request.",
            }),
            503,
        )
    except ValueError as ve:
        logger.warning("Feature schema validation failed during inference: %s", ve)
        return (
            jsonify({
                "success": False,
                "error": str(ve),
            }),
            422,
        )
    except Exception as e:
        logger.exception("Unexpected error during inference: %s", e)
        return (
            jsonify({
                "success": False,
                "error": "An unexpected error occurred while processing the prediction.",
            }),
            500,
        )
