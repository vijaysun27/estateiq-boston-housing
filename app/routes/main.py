"""Main web routes for EstateIQ."""

from flask import Blueprint, current_app, jsonify, render_template
from app.utils.validation import FEATURE_SPECIFICATIONS

main_bp = Blueprint("main", __name__)


@main_bp.route("/healthz", methods=["GET"])
def healthz():
    """Lightweight HTTP liveness probe for orchestrators (Render, K8s, load balancers)."""
    return jsonify({"status": "ok", "service": "estateiq-boston-housing"}), 200


@main_bp.route("/")
def index():
    """Render the primary prediction interface."""
    predictor = getattr(current_app, "predictor", None)
    model_ready = predictor is not None and predictor.is_ready()
    model_info = predictor.get_model_info() if predictor else {}

    return render_template(
        "index.html",
        features=FEATURE_SPECIFICATIONS,
        model_ready=model_ready,
        model_info=model_info,
        active_page="predictor",
    )


@main_bp.route("/insights")
def insights():
    """Render model insights, cross-validation benchmarks, and feature importance."""
    predictor = getattr(current_app, "predictor", None)
    model_ready = predictor is not None and predictor.is_ready()
    model_info = predictor.get_model_info() if predictor else {}

    return render_template(
        "insights.html",
        features=FEATURE_SPECIFICATIONS,
        model_ready=model_ready,
        model_info=model_info,
        active_page="insights",
    )


@main_bp.route("/about")
def about():
    """Render project background, dataset provenance, and ethical policy."""
    predictor = getattr(current_app, "predictor", None)
    model_ready = predictor is not None and predictor.is_ready()
    model_info = predictor.get_model_info() if predictor else {}

    return render_template(
        "about.html",
        features=FEATURE_SPECIFICATIONS,
        model_ready=model_ready,
        model_info=model_info,
        active_page="about",
    )
