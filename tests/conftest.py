"""Pytest configuration and shared fixtures for EstateIQ."""

import pytest
from app import create_app


@pytest.fixture
def app():
    """Create and configure a Flask application instance for testing."""
    application = create_app(config_name="testing")
    yield application


@pytest.fixture
def client(app):
    """Provide a test client for simulating HTTP requests."""
    return app.test_client()


@pytest.fixture
def sample_valid_features():
    """Provide a valid 12-feature dictionary based on typical dataset values."""
    return {
        "CRIM": 0.02,
        "ZN": 18.0,
        "INDUS": 2.5,
        "CHAS": 0,
        "NOX": 0.45,
        "RM": 6.5,
        "AGE": 65.0,
        "DIS": 4.5,
        "RAD": 1,
        "TAX": 300.0,
        "PTRATIO": 15.0,
        "LSTAT": 5.0,
    }
