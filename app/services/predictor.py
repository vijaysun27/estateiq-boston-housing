"""Model inference service for EstateIQ Boston Housing Price Predictor.

Manages artifact lifecycle, feature ordering, and predictions for both Scikit-Learn
pipelines and pre-trained Keras ANN models, decoupled from HTTP/Flask request semantics.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ModelNotAvailableError(Exception):
    """Raised when the trained model artifact cannot be found or loaded."""
    pass


class PredictorService:
    """Encapsulates model loading, feature schema mapping, and inference.

    Remains completely decoupled from Flask request objects and handles both
    single-dictionary and batch-DataFrame inputs.
    """

    # Canonical 12 predictor features (Historical B variable excluded from user input)
    CANONICAL_FEATURES: List[str] = [
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

    # Neutral reference value for the historical B variable in legacy 13-feature models
    NEUTRAL_B_VALUE: float = 356.674

    def __init__(
        self,
        pipeline_path: Optional[str] = None,
        keras_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
    ):
        self.pipeline_path = pipeline_path
        self.keras_path = keras_path
        self.scaler_path = scaler_path
        self.metadata_path = metadata_path

        self.pipeline = None
        self.keras_model = None
        self.scaler = None
        self.metadata: Dict[str, Any] = {}
        self.model_type: Optional[str] = None
        self.model_identifier: Optional[str] = None
        self._is_loaded = False

    def load_model(self) -> None:
        """Load trained model artifacts into memory once per process."""
        self._is_loaded = False

        # Attempt 1: Check for Scikit-Learn Pipeline (Phase 2 primary artifact)
        pipeline_file = Path(self.pipeline_path) if self.pipeline_path else None
        if pipeline_file and not pipeline_file.exists() and not pipeline_file.is_absolute():
            try:
                from app.config import BASE_DIR
                if (BASE_DIR / pipeline_file).exists():
                    pipeline_file = BASE_DIR / pipeline_file
            except Exception:
                pass

        if pipeline_file and pipeline_file.exists():
            try:
                import joblib

                self.pipeline = joblib.load(str(pipeline_file))
                regressor_name = (
                    self.pipeline.named_steps["regressor"].__class__.__name__
                    if hasattr(self.pipeline, "named_steps") and "regressor" in self.pipeline.named_steps
                    else "Estimator"
                )
                self.model_type = f"Scikit-Learn Pipeline ({regressor_name})"
                self.model_identifier = pipeline_file.stem
                self._is_loaded = True
                logger.info("Loaded scikit-learn pipeline from %s", pipeline_file)

                # Attempt to load metadata JSON
                meta_file = Path(
                    self.metadata_path
                    or (pipeline_file.parent / "model_metadata.json")
                )
                if not meta_file.exists() and not meta_file.is_absolute():
                    try:
                        from app.config import BASE_DIR
                        if (BASE_DIR / meta_file).exists():
                            meta_file = BASE_DIR / meta_file
                    except Exception:
                        pass

                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            self.metadata = json.load(f)
                        logger.info("Loaded model metadata from %s", meta_file)
                    except Exception as me:
                        logger.warning("Could not read metadata from %s: %s", meta_file, me)
                return
            except Exception as e:
                logger.exception("Failed to load pipeline from %s: %s", pipeline_file, e)

        # Attempt 2: Check for Keras ANN model (trained in ANN For Regression.ipynb)
        if self.keras_path and Path(self.keras_path).exists():
            try:
                import tensorflow as tf

                self.keras_model = tf.keras.models.load_model(self.keras_path)

                # Check for accompanying StandardScaler artifact
                scaler_file = self.scaler_path or str(Path(self.keras_path).parent / "scaler.joblib")
                if Path(scaler_file).exists():
                    import joblib

                    self.scaler = joblib.load(scaler_file)
                    logger.info("Loaded pre-fitted StandardScaler from %s", scaler_file)
                else:
                    logger.warning("No pre-fitted scaler found at %s.", scaler_file)

                self.model_type = "TensorFlow Keras Sequential ANN"
                self.model_identifier = "boston_housing_ann_keras"
                self._is_loaded = True
                logger.info("Loaded Keras ANN model from %s", self.keras_path)
                return
            except Exception as e:
                logger.warning("Failed to load Keras model from %s: %s", self.keras_path, e)

        logger.info("No model artifacts found on startup. Predictor initialized in unready state.")

    def is_ready(self) -> bool:
        """Check whether a model is loaded and ready to serve predictions."""
        return self._is_loaded and (self.pipeline is not None or self.keras_model is not None)

    def get_model_info(self) -> Dict[str, Any]:
        """Return non-sensitive model metadata, readiness, and metrics."""
        info: Dict[str, Any] = {
            "status": "ready" if self.is_ready() else "unavailable",
            "model_type": self.model_type,
            "model_identifier": self.model_identifier,
            "supported_features": self.CANONICAL_FEATURES,
            "target": "MEDV ($1,000s historical benchmark)",
        }
        if self.metadata:
            if "held_out_test_metrics" in self.metadata:
                info["test_metrics"] = self.metadata["held_out_test_metrics"]
            if "dataset_provenance" in self.metadata:
                info["dataset_provenance"] = self.metadata["dataset_provenance"]
            if "selected_algorithm" in self.metadata:
                info["selected_algorithm"] = self.metadata["selected_algorithm"]
            if "feature_importances" in self.metadata:
                info["feature_importances"] = self.metadata["feature_importances"]
        return info

    def predict(
        self,
        features: Union[Dict[str, Any], pd.DataFrame],
    ) -> Dict[str, Any]:
        """Generate house value prediction for the given 12 canonical features.

        Args:
            features: Dictionary containing all 12 required canonical features or
                      a pandas DataFrame containing the canonical feature columns.

        Returns:
            Dictionary containing prediction_raw, prediction_dollar, prediction_display,
            and relevant model metadata.

        Raises:
            ModelNotAvailableError: If no model artifact is loaded.
            ValueError: If required features are missing, non-numeric, or malformed.
        """
        if not self.is_ready():
            raise ModelNotAvailableError(
                "Model is currently unavailable. No trained artifact loaded."
            )

        # Normalize input to a DataFrame with canonical schema and exact column ordering
        if isinstance(features, pd.DataFrame):
            missing = [f for f in self.CANONICAL_FEATURES if f not in features.columns]
            if missing:
                raise ValueError(f"Missing required feature column(s): {missing}")
            df_input = features[self.CANONICAL_FEATURES].copy()
        elif isinstance(features, dict):
            # Exclude internal keys like '_warnings' if present
            clean_dict = {k: v for k, v in features.items() if not k.startswith("_")}
            missing = [f for f in self.CANONICAL_FEATURES if f not in clean_dict]
            if missing:
                raise ValueError(f"Missing required feature(s): {missing}")
            try:
                numeric_row = {f: float(clean_dict[f]) for f in self.CANONICAL_FEATURES}
            except (ValueError, TypeError) as conv_err:
                raise ValueError(f"Non-numeric feature value encountered: {conv_err}") from conv_err

            df_input = pd.DataFrame([numeric_row])[self.CANONICAL_FEATURES]
        else:
            raise ValueError(f"Unsupported features type: {type(features)}. Expected dict or DataFrame.")

        # Check for NaN or Inf in dataframe values
        if df_input.isnull().any().any():
            raise ValueError("Input features contain missing (NaN) values.")
        if np.isinf(df_input.values).any():
            raise ValueError("Input features contain infinite values.")

        # Case A: Scikit-learn Pipeline (Phase 2 primary artifact)
        if self.pipeline is not None:
            raw_predictions = self.pipeline.predict(df_input)
            raw_pred = float(raw_predictions[0])

        # Case B: TensorFlow Keras ANN Model (13-feature input vector with neutral B)
        elif self.keras_model is not None:
            # Map 12 features to 13-feature legacy vector:
            # [CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B_neutral, LSTAT]
            row_dict = df_input.iloc[0].to_dict()
            feature_vector_13 = np.array([[
                row_dict["CRIM"],
                row_dict["ZN"],
                row_dict["INDUS"],
                row_dict["CHAS"],
                row_dict["NOX"],
                row_dict["RM"],
                row_dict["AGE"],
                row_dict["DIS"],
                row_dict["RAD"],
                row_dict["TAX"],
                row_dict["PTRATIO"],
                self.NEUTRAL_B_VALUE,
                row_dict["LSTAT"],
            ]], dtype=np.float32)

            if self.scaler is not None:
                scaled_vector = self.scaler.transform(feature_vector_13)
            else:
                scaled_vector = feature_vector_13

            keras_output = self.keras_model.predict(scaled_vector, verbose=0)
            raw_pred = float(np.squeeze(keras_output))

        else:
            raise ModelNotAvailableError("Model inference engine could not be executed.")

        # Real estate median price cannot be negative
        raw_pred = max(0.0, raw_pred)
        rounded_raw = round(raw_pred, 3)

        # In Boston Housing dataset, MEDV is in thousands of dollars ($1,000s)
        dollar_value = round(rounded_raw * 1000.0, 2)

        response: Dict[str, Any] = {
            "prediction_raw": rounded_raw,
            "prediction_dollar": dollar_value,
            "prediction_display": f"${dollar_value:,.2f}",
            "unit": "USD (1970s benchmark equivalent)",
            "model_identifier": self.model_identifier,
            "model_type": self.model_type,
            "disclaimer": "Historical benchmark estimate for educational demonstration. Not current market appraisal.",
        }

        if self.metadata and "held_out_test_metrics" in self.metadata:
            response["metrics"] = self.metadata["held_out_test_metrics"]

        return response
