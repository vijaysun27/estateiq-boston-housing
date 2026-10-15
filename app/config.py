"""Application configuration classes for EstateIQ."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class BaseConfig:
    """Base configuration shared across environments."""

    SECRET_KEY = os.getenv("SECRET_KEY", "estateiq-default-development-secret-key-42")
    JSON_SORT_KEYS = False
    # Security: Limit maximum request payload size to 16 KB
    MAX_CONTENT_LENGTH = 16 * 1024

    # Directories
    BASE_DIR = BASE_DIR
    MODELS_DIR = BASE_DIR / "models"
    DATA_DIR = BASE_DIR / "data"

    # Artifact paths
    MODEL_PIPELINE_PATH = os.getenv(
        "MODEL_PIPELINE_PATH",
        str(MODELS_DIR / "boston_housing_pipeline.joblib"),
    )
    KERAS_MODEL_PATH = os.getenv(
        "KERAS_MODEL_PATH",
        str(MODELS_DIR / "boston_housing_model.keras"),
    )
    METADATA_PATH = os.getenv(
        "METADATA_PATH",
        str(MODELS_DIR / "model_metadata.json"),
    )

    # 12 User-facing Canonical Predictor Features (Historical B variable excluded from user input)
    CANONICAL_FEATURES = [
        "CRIM",
        "ZN",
        "INDUS",
        "CHAS",
        "NOX",
        "RM",
        "AGE",
        "DIS",
        "RAD",
        "TAX",
        "PTRATIO",
        "LSTAT",
    ]

    TARGET_NAME = "MEDV"
    TARGET_UNIT = "$1,000s (Historical 1970s benchmark)"


class DevelopmentConfig(BaseConfig):
    """Development environment settings."""

    DEBUG = True
    TESTING = False


class TestingConfig(BaseConfig):
    """Testing environment settings."""

    DEBUG = False
    TESTING = True
    SECRET_KEY = "estateiq-test-secret-key"


class ProductionConfig(BaseConfig):
    """Production environment settings."""

    DEBUG = False
    TESTING = False


config_by_name = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
