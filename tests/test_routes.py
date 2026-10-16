"""Tests for HTTP routes and API endpoints."""

import json
from app.services.predictor import PredictorService


def test_index_route(client):
    """Verify that the home page renders with status 200."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Boston Housing" in response.data
    assert b"Estate" in response.data


def test_about_route(client):
    """Verify that the about / model card page renders with status 200."""
    response = client.get("/about")
    assert response.status_code == 200
    assert b"Ethical Feature Policy" in response.data or b"Exclusion of Variable" in response.data


def test_insights_route(client):
    """Verify that the /insights model benchmark page renders with status 200."""
    response = client.get("/insights")
    assert response.status_code == 200
    assert b"Model Performance" in response.data
    assert b"Feature Importances" in response.data


def test_api_sample_endpoint(client):
    """Verify that /api/sample returns 12 canonical features."""
    response = client.get("/api/sample")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "sample" in data
    assert len(data["sample"]) == 12


def test_api_health_endpoint_healthy(client):
    """Verify /api/health endpoint structure when model is loaded."""
    response = client.get("/api/health")
    data = response.get_json()
    assert response.status_code == 200
    assert data["status"] == "healthy"
    assert data["model_ready"] is True
    assert "model_info" in data
    assert "supported_features" in data["model_info"]
    # Path or secrets should not be exposed
    assert "pipeline_path" not in data["model_info"]
    assert "secret_key" not in data


def test_healthz_endpoint(client):
    """Verify /healthz lightweight liveness probe returns HTTP 200."""
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert "service" in data


def test_predict_endpoint_valid_payload(client, sample_valid_features):
    """Verify /api/predict returns 200 and formatted valuation when model is loaded."""
    response = client.post(
        "/api/predict",
        data=json.dumps(sample_valid_features),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "data" in data
    pred = data["data"]
    assert "prediction_raw" in pred
    assert "prediction_dollar" in pred
    assert "prediction_display" in pred
    assert "unit" in pred
    assert "model_identifier" in pred
    assert pred["prediction_raw"] > 0


def test_predict_endpoint_invalid_json(client):
    """Verify non-JSON request is rejected with 400."""
    response = client.post(
        "/api/predict",
        data="invalid-payload",
        content_type="text/plain",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "error" in data


def test_predict_endpoint_missing_feature(client, sample_valid_features):
    """Verify missing features return 400 with validation details."""
    incomplete = sample_valid_features.copy()
    del incomplete["CRIM"]

    response = client.post(
        "/api/predict",
        data=json.dumps(incomplete),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "details" in data
    assert "CRIM" in data["details"]


def test_predict_endpoint_invalid_chas(client, sample_valid_features):
    """Verify invalid CHAS value returns 400."""
    bad_chas = sample_valid_features.copy()
    bad_chas["CHAS"] = 5

    response = client.post(
        "/api/predict",
        data=json.dumps(bad_chas),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "CHAS" in data["details"]


def test_predict_endpoint_unloaded_model_returns_503(app, client, sample_valid_features):
    """Verify that when the predictor is unready, /api/predict returns 503."""
    original_predictor = app.predictor
    try:
        # Swap with unready predictor double
        unready = PredictorService(pipeline_path="missing.joblib", keras_path="missing.keras")
        app.predictor = unready

        response = client.post(
            "/api/predict",
            data=json.dumps(sample_valid_features),
            content_type="application/json",
        )
        assert response.status_code == 503
        data = response.get_json()
        assert data["success"] is False
        assert "unavailable" in data["error"].lower()

        # Health endpoint should also return 503
        health_resp = client.get("/api/health")
        assert health_resp.status_code == 503
        assert health_resp.get_json()["model_ready"] is False
    finally:
        app.predictor = original_predictor


def test_404_error_page_html(client):
    """Verify unknown web routes render 404 HTML template."""
    response = client.get("/nonexistent-page")
    assert response.status_code == 404
    assert b"404" in response.data


def test_404_error_api_json(client):
    """Verify unknown API routes return 404 JSON."""
    response = client.get("/api/nonexistent-endpoint")
    assert response.status_code == 404
    data = response.get_json()
    assert data["success"] is False
    assert "error" in data
