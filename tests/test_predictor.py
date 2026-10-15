"""Tests for PredictorService inference engine."""

import pandas as pd
import pytest
from app.services.predictor import ModelNotAvailableError, PredictorService


def test_predictor_unloaded_raises_error(sample_valid_features):
    """Verify that calling predict on an unloaded service raises ModelNotAvailableError."""
    service = PredictorService(pipeline_path="nonexistent.joblib", keras_path="nonexistent.keras")
    assert service.is_ready() is False

    with pytest.raises(ModelNotAvailableError):
        service.predict(sample_valid_features)


def test_predictor_loaded_from_artifact(app):
    """Verify that the PredictorService correctly loads the trained pipeline artifact."""
    predictor = getattr(app, "predictor", None)
    assert predictor is not None
    assert predictor.is_ready() is True
    assert "Pipeline" in predictor.model_type or "ANN" in predictor.model_type


def test_predictor_successful_prediction_dict(app, sample_valid_features):
    """Verify that predict returns correct schema and positive valuation for dict input."""
    predictor = getattr(app, "predictor", None)
    assert predictor is not None and predictor.is_ready()

    result = predictor.predict(sample_valid_features)
    assert "prediction_raw" in result
    assert "prediction_dollar" in result
    assert "prediction_display" in result
    assert "unit" in result
    assert "model_identifier" in result
    assert "disclaimer" in result

    # Verify valuation calculations and units ($1,000s)
    raw = result["prediction_raw"]
    dollar = result["prediction_dollar"]
    assert raw > 0.0
    assert abs(dollar - (raw * 1000.0)) < 1.0
    assert result["prediction_display"].startswith("$")


def test_predictor_successful_prediction_dataframe(app, sample_valid_features):
    """Verify that predict accepts a pandas DataFrame with canonical schema."""
    predictor = getattr(app, "predictor", None)
    assert predictor is not None and predictor.is_ready()

    df = pd.DataFrame([sample_valid_features])
    result = predictor.predict(df)
    assert "prediction_raw" in result
    assert result["prediction_raw"] > 0.0


def test_predictor_missing_feature_raises_value_error(app, sample_valid_features):
    """Verify that omitting a canonical feature raises ValueError."""
    predictor = getattr(app, "predictor", None)
    assert predictor is not None and predictor.is_ready()

    incomplete = sample_valid_features.copy()
    del incomplete["TAX"]
    with pytest.raises(ValueError, match="Missing required feature"):
        predictor.predict(incomplete)


def test_predictor_nan_value_raises_value_error(app, sample_valid_features):
    """Verify that NaN values in features raise ValueError."""
    predictor = getattr(app, "predictor", None)
    assert predictor is not None and predictor.is_ready()

    bad_payload = sample_valid_features.copy()
    bad_payload["CRIM"] = float("nan")
    with pytest.raises(ValueError):
        predictor.predict(bad_payload)


def test_predictor_feature_count_and_order(app):
    """Verify that PredictorService enforces exact 12-feature count and canonical order."""
    predictor = getattr(app, "predictor", None)
    assert len(predictor.CANONICAL_FEATURES) == 12
    assert "B" not in predictor.CANONICAL_FEATURES
    assert predictor.CANONICAL_FEATURES[0] == "CRIM"
    assert predictor.CANONICAL_FEATURES[-1] == "LSTAT"


def test_predictor_service_metadata():
    """Verify get_model_info returns expected schema and status."""
    service = PredictorService()
    info = service.get_model_info()
    assert "status" in info
    assert info["status"] == "unavailable"
    assert "supported_features" in info
    assert len(info["supported_features"]) == 12
