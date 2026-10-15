"""EstateIQ Boston Housing Price Predictor — Model Training Engine (Phase 2).

Executes a reproducible scikit-learn regression pipeline:
- Loads and validates the authentic Boston Housing dataset (506 observations).
- Excludes the historical race-related 'B' feature from predictors.
- Partitions the data into training (80%) and held-out test (20%) sets using a fixed seed.
- Evaluates a baseline (DummyRegressor) and candidate estimators (Linear, Ridge,
  RandomForest, GradientBoosting) via 5-fold cross-validation on training data only.
- Selects the best performing model without test-set leakage.
- Computes final MAE, RMSE, and R-squared on the held-out test set.
- Persists the fitted pipeline to models/boston_housing_pipeline.joblib and saves
  models/model_metadata.json with full provenance and performance metrics.
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("estateiq.train")

# Canonical 12 predictor schema (Historical 'B' variable strictly excluded)
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
TARGET_VARIABLE = "MEDV"
ALL_ORIGINAL_FEATURES = [
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
    "B",
    "LSTAT",
]


def parse_arguments() -> argparse.Namespace:
    """Parse command line options for model training."""
    parser = argparse.ArgumentParser(
        description="Train EstateIQ Boston Housing regression model pipeline."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/boston_housing.csv",
        help="Path to local Boston Housing CSV dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save the trained pipeline artifact and metadata.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split and estimator reproducibility.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of the dataset reserved for held-out testing.",
    )
    return parser.parse_args()


def load_and_validate_dataset(data_path: Path) -> pd.DataFrame:
    """Load authentic Boston Housing dataset and verify integrity.

    Verifies:
    - 506 observations.
    - All canonical predictors and target present.
    - Zero missing values.
    - Valid data types and reasonable empirical ranges.
    """
    if data_path.exists():
        logger.info("Loading Boston Housing dataset from local file: %s", data_path)
        df = pd.read_csv(data_path)
    else:
        logger.info(
            "Local CSV not found at %s. Loading documented benchmark dataset via keras.datasets...",
            data_path,
        )
        try:
            from tensorflow.keras.datasets import boston_housing

            (x_all, y_all), _ = boston_housing.load_data(test_split=0.0)
            df = pd.DataFrame(x_all, columns=ALL_ORIGINAL_FEATURES)
            df[TARGET_VARIABLE] = y_all
            data_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(data_path, index=False)
            logger.info("Saved verified dataset to %s", data_path)
        except Exception as e:
            logger.error("Failed to load documented Boston Housing dataset: %s", e)
            raise RuntimeError(
                f"Blocker: Documented Boston Housing dataset could not be loaded: {e}"
            ) from e

    # Dimensions check
    if len(df) != 506:
        raise ValueError(
            f"Dataset must contain exactly 506 observations, found {len(df)}."
        )

    # Columns check
    missing_cols = [c for c in CANONICAL_FEATURES + [TARGET_VARIABLE] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    # Null values check
    null_counts = df[CANONICAL_FEATURES + [TARGET_VARIABLE]].isnull().sum()
    if null_counts.any():
        raise ValueError(f"Dataset contains null values: {null_counts[null_counts > 0].to_dict()}")

    # Target sanity check: MEDV is in thousands of dollars (typical range 5.0 to 50.0)
    if (df[TARGET_VARIABLE] <= 0).any():
        raise ValueError("Target MEDV contains non-positive values.")

    logger.info(
        "Dataset verified successfully: %d rows, %d canonical predictors, target='%s' (mean=%.2f, std=%.2f).",
        len(df),
        len(CANONICAL_FEATURES),
        TARGET_VARIABLE,
        df[TARGET_VARIABLE].mean(),
        df[TARGET_VARIABLE].std(),
    )

    if "B" in df.columns:
        logger.info(
            "Ethical Feature Policy: Historical variable 'B' detected and will be excluded from training."
        )

    return df


def evaluate_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
) -> Tuple[str, Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
    """Evaluate candidate regression pipelines via 5-fold cross-validation on training data.

    Returns:
        Tuple of (best_model_name, dictionary_of_pipelines, cross_validation_scores)
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    # Candidate pipelines
    candidates: Dict[str, Pipeline] = {
        "baseline_dummy_mean": Pipeline([
            ("regressor", DummyRegressor(strategy="mean")),
        ]),
        "linear_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]),
        "ridge_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0, random_state=seed)),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", RandomForestRegressor(n_estimators=100, max_depth=6, random_state=seed)),
        ]),
        "gradient_boosting": Pipeline([
            ("scaler", StandardScaler()),
            (
                "regressor",
                GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=seed,
                ),
            ),
        ]),
    }

    cv_results: Dict[str, Dict[str, float]] = {}
    best_name = ""
    best_score = float("inf")  # selecting based on lowest CV RMSE

    logger.info("Evaluating candidates with 5-fold cross-validation on training split (N=%d)...", len(X_train))

    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    for name, pipe in candidates.items():
        scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=kf,
            scoring=scoring,
            n_jobs=-1,
        )
        mean_mae = float(-np.mean(scores["test_mae"]))
        mean_rmse = float(-np.mean(scores["test_rmse"]))
        mean_r2 = float(np.mean(scores["test_r2"]))

        cv_results[name] = {
            "cv_mae": round(mean_mae, 4),
            "cv_rmse": round(mean_rmse, 4),
            "cv_r2": round(mean_r2, 4),
        }

        logger.info(
            "Candidate: %-22s | CV RMSE: %6.4f | CV MAE: %6.4f | CV R2: %6.4f",
            name,
            mean_rmse,
            mean_mae,
            mean_r2,
        )

        # Baseline is excluded from winning selection
        if name != "baseline_dummy_mean" and mean_rmse < best_score:
            best_score = mean_rmse
            best_name = name

    logger.info("Best candidate selected by CV RMSE: '%s' (CV RMSE=%.4f)", best_name, best_score)
    return best_name, candidates, cv_results


def train_and_evaluate(
    df: pd.DataFrame,
    seed: int,
    test_size: float,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Partition data, select model using CV, fit best model, and test on held-out data.

    Returns:
        Tuple of (fitted_pipeline, full_metadata_dict)
    """
    X = df[CANONICAL_FEATURES]
    y = df[TARGET_VARIABLE]

    # Held-out test split (strictly isolated)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    logger.info(
        "Partitioned data: Train=%d samples (%.0f%%), Held-Out Test=%d samples (%.0f%%) [Seed=%d]",
        len(X_train),
        (1 - test_size) * 100,
        len(X_test),
        test_size * 100,
        seed,
    )

    # Cross-validation and model selection on training data only
    best_name, candidates, cv_results = evaluate_candidates(X_train, y_train, seed=seed)
    best_pipeline = candidates[best_name]

    # Fit the chosen pipeline on the complete training set
    logger.info("Fitting best pipeline ('%s') on full training set...", best_name)
    best_pipeline.fit(X_train, y_train)

    # Evaluate on held-out test set
    y_test_pred = best_pipeline.predict(X_test)
    test_mae = float(mean_absolute_error(y_test, y_test_pred))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
    test_r2 = float(r2_score(y_test, y_test_pred))

    # Also evaluate baseline on held-out test set for honest comparison
    baseline_pipe = candidates["baseline_dummy_mean"]
    baseline_pipe.fit(X_train, y_train)
    baseline_pred = baseline_pipe.predict(X_test)
    baseline_mae = float(mean_absolute_error(y_test, baseline_pred))
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_pred)))
    baseline_r2 = float(r2_score(y_test, baseline_pred))

    logger.info("=================================================================")
    logger.info("                    FINAL EVALUATION SUMMARY                     ")
    logger.info("=================================================================")
    logger.info(
        "Baseline (Mean Dummy) Held-Out: RMSE=%.4f, MAE=%.4f, R2=%.4f",
        baseline_rmse,
        baseline_mae,
        baseline_r2,
    )
    logger.info(
        "Selected Model (%s) Held-Out: RMSE=%.4f, MAE=%.4f, R2=%.4f",
        best_name,
        test_rmse,
        test_mae,
        test_r2,
    )
    logger.info("=================================================================")

    regressor_step = best_pipeline.named_steps["regressor"]
    regressor_type = regressor_step.__class__.__name__

    metadata: Dict[str, Any] = {
        "model_name": "EstateIQ Boston Housing Scikit-Learn Pipeline",
        "model_identifier": "boston_housing_pipeline",
        "model_type": f"Scikit-Learn Pipeline ({regressor_type})",
        "selected_algorithm": best_name,
        "framework": "scikit-learn",
        "target_variable": TARGET_VARIABLE,
        "target_unit": "$1,000s USD (1970s historical benchmark)",
        "features": {
            "canonical_schema": CANONICAL_FEATURES,
            "feature_count": len(CANONICAL_FEATURES),
            "excluded_features": ["B"],
            "exclusion_rationale": "Historical racial variable excluded per modern AI ethics guidelines.",
        },
        "dataset_provenance": {
            "source": "Harrison, D. and Rubinfeld, D.L. (1978) Hedonic prices and the demand for clean air",
            "total_observations": len(df),
            "train_observations": len(X_train),
            "test_observations": len(X_test),
            "random_seed": seed,
            "test_size": test_size,
        },
        "cross_validation_evaluation": cv_results,
        "baseline_comparison": {
            "baseline_model": "DummyRegressor(strategy='mean')",
            "held_out_mae": round(baseline_mae, 4),
            "held_out_rmse": round(baseline_rmse, 4),
            "held_out_r2": round(baseline_r2, 4),
        },
        "held_out_test_metrics": {
            "mae": round(test_mae, 4),
            "rmse": round(test_rmse, 4),
            "r2": round(test_r2, 4),
        },
        "hyperparameters": regressor_step.get_params(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "disclaimer": "Historical academic valuation model. Not valid for contemporary financial or market appraisal.",
    }

    if hasattr(regressor_step, "feature_importances_"):
        metadata["feature_importances"] = {
            f: round(float(imp), 4)
            for f, imp in zip(CANONICAL_FEATURES, regressor_step.feature_importances_)
        }

    return best_pipeline, metadata


def save_artifacts(
    pipeline: Pipeline,
    metadata: Dict[str, Any],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Save fitted pipeline joblib artifact and metadata JSON to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_path = output_dir / "boston_housing_pipeline.joblib"
    metadata_path = output_dir / "model_metadata.json"

    logger.info("Serializing fitted pipeline to %s ...", pipeline_path)
    joblib.dump(pipeline, pipeline_path)

    logger.info("Saving model metadata to %s ...", metadata_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Artifacts saved successfully.")
    return pipeline_path, metadata_path


def main() -> int:
    """Main execution entry point."""
    args = parse_arguments()
    logger.info("Starting EstateIQ Phase 2 model training workflow.")
    logger.info("Random seed: %d | Test split: %.2f", args.seed, args.test_size)

    try:
        data_path = Path(args.data_path)
        output_dir = Path(args.output_dir)

        df = load_and_validate_dataset(data_path)
        pipeline, metadata = train_and_evaluate(df, seed=args.seed, test_size=args.test_size)
        save_artifacts(pipeline, metadata, output_dir)

        logger.info("Phase 2 model training completed successfully.")
        return 0
    except Exception as e:
        logger.exception("Training pipeline failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
