"""Application configuration classes for EstateIQ."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _resolve_artifact_path(env_var_name: str, relative_default: str) -> str:
    """Resolve an artifact path to an absolute path, prioritizing env vars and anchoring relative paths to BASE_DIR."""
    val = os.getenv(env_var_name)
    if val:
        p = Path(val)
        if not p.is_absolute():
            p = BASE_DIR / p
        return str(p)
    return str(BASE_DIR / relative_default)


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

    # Artifact paths (guaranteed absolute and robust across any container working directory)
    MODEL_PIPELINE_PATH = _resolve_artifact_path(
        "MODEL_PIPELINE_PATH",
        "models/boston_housing_pipeline.joblib",
    )
    KERAS_MODEL_PATH = _resolve_artifact_path(
        "KERAS_MODEL_PATH",
        "models/boston_housing_model.keras",
    )
    METADATA_PATH = _resolve_artifact_path(
        "METADATA_PATH",
        "models/model_metadata.json",
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
