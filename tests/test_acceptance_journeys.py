"""Functional acceptance tests covering User Journeys A through G (Phase 4).

Validates end-to-end user workflows, edge cases, error handling, and reliability:
- Journey A: First visit & initial dashboard state
- Journey B: Successful prediction & unit conversions ($1,000s -> USD)
- Journey C: Clear previous prediction preserving form values
- Journey D: Reset form to initial state
- Journey E: Invalid inputs, bounds, malformed payloads, and unexpected keys
- Journey F: Model unavailable & graceful 503 error isolation
- Journey G: Request race conditions & rapid concurrent submissions
"""

import json
from concurrent.futures import ThreadPoolExecutor
from app.services.predictor import PredictorService


# ==============================================================================
# Journey A: First Visit
# ==============================================================================
def test_journey_a_first_visit(client):
    """Journey A: Verify initial visit loads cleanly, reports model readiness, and renders standby state."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.data.decode("utf-8")

    # Brand and Hero
    assert "EstateIQ" in html
    assert "Predict your property's value" in html

    # Model status badge reflects backend state
    assert "Model Ready" in html or "Model Unavailable" in html

    # Initial standby empty state
    assert "state-standby" in html
    assert "Awaiting Input" in html

    # Static assets are linked correctly
    assert 'href="/static/css/styles.css"' in html
    assert 'src="/static/js/app.js"' in html


# ==============================================================================
# Journey B: Successful Prediction
# ==============================================================================
def test_journey_b_successful_prediction(client, sample_valid_features):
    """Journey B: Verify valid sample submission generates correct prediction and unit conversion."""
    response = client.post(
        "/api/predict",
        data=json.dumps(sample_valid_features),
        content_type="application/json",
    )
    assert response.status_code == 200
    payload = response.get_json()

    assert payload["success"] is True
    data = payload["data"]

    # Verify 12-feature processing produced positive numerical valuation
    assert "prediction_raw" in data
    assert "prediction_dollar" in data
    raw_val = data["prediction_raw"]
    dollar_val = data["prediction_dollar"]
    assert raw_val > 0.0

    # Verify correct unit conversion: MEDV is in $1,000s
    expected_dollar = round(raw_val * 1000.0, 2)
    assert abs(dollar_val - expected_dollar) <= 0.01

    # Verify display formatting and disclaimer
    assert data["prediction_display"].startswith("$")
    assert "historical benchmark" in data["disclaimer"].lower()
    assert "not current market appraisal" in data["disclaimer"].lower()
    assert "boston_housing_pipeline" in data["model_identifier"]


# ==============================================================================
# Journey C: Clear Previous Prediction
# ==============================================================================
def test_journey_c_clear_previous_prediction(client, sample_valid_features):
    """Journey C: Verify clearing prediction preserves inputs and allows immediate new prediction."""
    # Step 1: Initial prediction
    resp1 = client.post(
        "/api/predict",
        data=json.dumps(sample_valid_features),
        content_type="application/json",
    )
    assert resp1.status_code == 200
    pred1 = resp1.get_json()["data"]["prediction_raw"]

    # Step 2: Form inputs are preserved on client; user modifies RM from 6.5 to 8.0
    modified_features = sample_valid_features.copy()
    modified_features["RM"] = 8.0

    # Step 3: Subsequent prediction with modified features produces updated valuation
    resp2 = client.post(
        "/api/predict",
        data=json.dumps(modified_features),
        content_type="application/json",
    )
    assert resp2.status_code == 200
    pred2 = resp2.get_json()["data"]["prediction_raw"]

    # Larger dwellings (8 rooms vs 6.5 rooms) must yield a distinct, higher predicted valuation
    assert pred2 > pred1


# ==============================================================================
# Journey D: Reset the Form
# ==============================================================================
def test_journey_d_reset_form(client):
    """Journey D: Verify sample endpoint provides clean default reset states."""
    resp = client.get("/api/sample")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    sample = data["sample"]

    # Exactly 12 features, CHAS is 0, RAD is valid level
    assert len(sample) == 12
    assert "B" not in sample
    assert sample["CHAS"] in (0, 1)
    assert sample["RAD"] in (1, 2, 3, 4, 5, 6, 7, 8, 24)


# ==============================================================================
# Journey E: Invalid Inputs & Error Handling
# ==============================================================================
def test_journey_e_missing_required_field(client, sample_valid_features):
    """Journey E: Missing feature must be rejected with 400 and actionable field details."""
    payload = sample_valid_features.copy()
    del payload["CRIM"]

    resp = client.post("/api/predict", data=json.dumps(payload), content_type="application/json")
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["success"] is False
    assert "CRIM" in data["details"]


def test_journey_e_empty_values(client, sample_valid_features):
    """Journey E: Empty string value rejected with 400."""
    payload = sample_valid_features.copy()
    payload["TAX"] = ""

    resp = client.post("/api/predict", data=json.dumps(payload), content_type="application/json")
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["success"] is False
    assert "TAX" in data["details"]


def test_journey_e_non_numeric_strings(client, sample_valid_features):
    """Journey E: Non-numeric strings rejected with 400."""
    payload = sample_valid_features.copy()
    payload["RM"] = "eight_rooms"

    resp = client.post("/api/predict", data=json.dumps(payload), content_type="application/json")
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["success"] is False
    assert "RM" in data["details"]


def test_journey_e_nan_and_infinity(client, sample_valid_features):
    """Journey E: NaN and infinity values rejected with 400."""
    # Raw JSON with NaN/Infinity string representations
    resp_nan = client.post(
        "/api/predict",
        data=json.dumps({**sample_valid_features, "DIS": float("nan")}),
        content_type="application/json",
    )
    # Python json serializes nan as NaN, invalid JSON or rejected in validation
    assert resp_nan.status_code in (400, 422)

    resp_inf = client.post(
        "/api/predict",
        data=json.dumps({**sample_valid_features, "DIS": float("inf")}),
        content_type="application/json",
    )
    assert resp_inf.status_code in (400, 422)


def test_journey_e_invalid_chas_values(client, sample_valid_features):
    """Journey E: Non-binary CHAS values rejected with 400."""
    for invalid_val in [2, -1, 99]:
        payload = sample_valid_features.copy()
        payload["CHAS"] = invalid_val
        resp = client.post("/api/predict", data=json.dumps(payload), content_type="application/json")
        assert resp.status_code == 400
        assert "CHAS" in resp.get_json()["details"]


def test_journey_e_invalid_rad_values(client, sample_valid_features):
    """Journey E: RAD values outside {1-8, 24} rejected with 400."""
    for invalid_rad in [9, 15, 25, 0]:
        payload = sample_valid_features.copy()
        payload["RAD"] = invalid_rad
        resp = client.post("/api/predict", data=json.dumps(payload), content_type="application/json")
        assert resp.status_code == 400
        assert "RAD" in resp.get_json()["details"]


def test_journey_e_out_of_bounds(client, sample_valid_features):
    """Journey E: Feature values outside allowable domain rejected with 400."""
    # Negative crime rate
    payload = sample_valid_features.copy()
    payload["CRIM"] = -5.0
    resp = client.post("/api/predict", data=json.dumps(payload), content_type="application/json")
    assert resp.status_code == 400
    assert "CRIM" in resp.get_json()["details"]

    # Impossibly high rooms
    payload2 = sample_valid_features.copy()
    payload2["RM"] = 30.0
    resp2 = client.post("/api/predict", data=json.dumps(payload2), content_type="application/json")
    assert resp2.status_code == 400
    assert "RM" in resp2.get_json()["details"]


def test_journey_e_malformed_json(client):
    """Journey E: Malformed JSON syntax rejected with 400 without crashing."""
    resp = client.post(
        "/api/predict",
        data="{ 'unclosed': 'json', ",
        content_type="application/json",
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["success"] is False
    assert "json" in data["error"].lower() or "bad request" in data["error"].lower()


def test_journey_e_unexpected_fields_including_b(client, sample_valid_features):
    """Journey E: Unexpected properties (especially historical variable 'B') rejected with 400."""
    # Including excluded racial variable B
    payload = sample_valid_features.copy()
    payload["B"] = 396.90
    resp = client.post("/api/predict", data=json.dumps(payload), content_type="application/json")
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["success"] is False
    assert "_unexpected_fields" in data["details"]

    # Random unauthorized property
    payload2 = sample_valid_features.copy()
    payload2["UNAUTHORIZED_KEY"] = "exploit"
    resp2 = client.post("/api/predict", data=json.dumps(payload2), content_type="application/json")
    assert resp2.status_code == 400


# ==============================================================================
# Journey F: Model Unavailable
# ==============================================================================
def test_journey_f_model_unavailable(app, client, sample_valid_features):
    """Journey F: When model is missing, requests fail with 503 and no tracebacks."""
    orig_predictor = app.predictor
    try:
        # Simulate unready service
        unready_svc = PredictorService(pipeline_path="missing.joblib", keras_path="missing.keras")
        app.predictor = unready_svc

        # /api/predict returns 503
        pred_resp = client.post(
            "/api/predict",
            data=json.dumps(sample_valid_features),
            content_type="application/json",
        )
        assert pred_resp.status_code == 503
        pred_data = pred_resp.get_json()
        assert pred_data["success"] is False
        assert "unavailable" in pred_data["error"].lower()
        # Verify no traceback leaked
        assert "traceback" not in pred_resp.data.decode("utf-8").lower()

        # /api/health reports model_ready = False
        health_resp = client.get("/api/health")
        assert health_resp.status_code == 503
        health_data = health_resp.get_json()
        assert health_data["model_ready"] is False
        assert health_data["status"] == "model_unavailable"
    finally:
        app.predictor = orig_predictor


# ==============================================================================
# Journey G: Request Concurrency & Independence
# ==============================================================================
def test_journey_g_concurrency_and_independence(client, sample_valid_features):
    """Journey G: Multiple concurrent prediction requests execute deterministically."""
    sample1 = sample_valid_features.copy()
    sample1["RM"] = 5.0

    sample2 = sample_valid_features.copy()
    sample2["RM"] = 8.0

    def make_request(features):
        return client.post(
            "/api/predict",
            data=json.dumps(features),
            content_type="application/json",
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(make_request, sample1),
            executor.submit(make_request, sample2),
            executor.submit(make_request, sample1),
            executor.submit(make_request, sample2),
        ]
        results = [f.result() for f in futures]

    for r in results:
        assert r.status_code == 200

    vals = [r.get_json()["data"]["prediction_raw"] for r in results]
    # Check deterministic outputs: results for sample1 must match, sample2 must match
    assert vals[0] == vals[2]
    assert vals[1] == vals[3]
    assert vals[1] > vals[0]
