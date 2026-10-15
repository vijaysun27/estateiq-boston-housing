"""Unit tests for training pipeline and dataset preparation (Phase 2)."""

import numpy as np
import pandas as pd
import pytest
from scripts.train_model import (
    CANONICAL_FEATURES,
    TARGET_VARIABLE,
    evaluate_candidates,
    load_and_validate_dataset,
    train_and_evaluate,
)


@pytest.fixture
def mini_synthetic_boston_df():
    """Controlled in-memory DataFrame resembling Boston Housing structure."""
    np.random.seed(42)
    n = 506
    data = {
        "CRIM": np.random.uniform(0.01, 10.0, n),
        "ZN": np.random.choice([0.0, 12.5, 25.0], n),
        "INDUS": np.random.uniform(1.0, 25.0, n),
        "CHAS": np.random.choice([0, 1], n),
        "NOX": np.random.uniform(0.4, 0.8, n),
        "RM": np.random.uniform(4.0, 8.5, n),
        "AGE": np.random.uniform(10.0, 95.0, n),
        "DIS": np.random.uniform(1.5, 9.0, n),
        "RAD": np.random.choice([1, 2, 3, 4, 5, 24], n),
        "TAX": np.random.uniform(200.0, 650.0, n),
        "PTRATIO": np.random.uniform(13.0, 21.0, n),
        "B": np.random.uniform(300.0, 396.9, n),
        "LSTAT": np.random.uniform(2.0, 30.0, n),
        "MEDV": np.random.uniform(10.0, 48.0, n),
    }
    return pd.DataFrame(data)


def test_dataset_verification_valid_dataset(tmp_path, mini_synthetic_boston_df):
    """Verify load_and_validate_dataset passes for a valid 506-row dataset."""
    csv_file = tmp_path / "valid_boston.csv"
    mini_synthetic_boston_df.to_csv(csv_file, index=False)

    df = load_and_validate_dataset(csv_file)
    assert len(df) == 506
    assert TARGET_VARIABLE in df.columns
    for feat in CANONICAL_FEATURES:
        assert feat in df.columns


def test_dataset_verification_wrong_row_count(tmp_path, mini_synthetic_boston_df):
    """Verify load_and_validate_dataset raises ValueError if row count != 506."""
    csv_file = tmp_path / "short_boston.csv"
    mini_synthetic_boston_df.iloc[:200].to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="506 observations"):
        load_and_validate_dataset(csv_file)


def test_dataset_verification_missing_column(tmp_path, mini_synthetic_boston_df):
    """Verify load_and_validate_dataset raises ValueError if a canonical column is missing."""
    csv_file = tmp_path / "missing_col.csv"
    bad_df = mini_synthetic_boston_df.drop(columns=["CRIM"])
    bad_df.to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_and_validate_dataset(csv_file)


def test_evaluate_candidates_and_selection(mini_synthetic_boston_df):
    """Verify that candidate evaluation selects a non-baseline pipeline and records CV metrics."""
    X_train = mini_synthetic_boston_df[CANONICAL_FEATURES].iloc[:404]
    y_train = mini_synthetic_boston_df[TARGET_VARIABLE].iloc[:404]

    best_name, candidates, cv_results = evaluate_candidates(X_train, y_train, seed=42)

    assert best_name != "baseline_dummy_mean"
    assert "baseline_dummy_mean" in cv_results
    assert best_name in candidates
    assert "cv_rmse" in cv_results[best_name]
    assert "cv_mae" in cv_results[best_name]
    assert "cv_r2" in cv_results[best_name]


def test_train_and_evaluate_output(mini_synthetic_boston_df):
    """Verify that train_and_evaluate produces a fitted pipeline and complete metadata."""
    pipeline, metadata = train_and_evaluate(
        mini_synthetic_boston_df, seed=42, test_size=0.2
    )

    assert hasattr(pipeline, "predict")
    assert metadata["features"]["feature_count"] == 12
    assert "B" in metadata["features"]["excluded_features"]
    assert "held_out_test_metrics" in metadata
    assert "mae" in metadata["held_out_test_metrics"]
    assert "rmse" in metadata["held_out_test_metrics"]
    assert "r2" in metadata["held_out_test_metrics"]
