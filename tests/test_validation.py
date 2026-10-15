"""Unit tests for input feature schema and validation utility."""

from app.utils.validation import validate_input_features


def test_valid_payload_passes(sample_valid_features):
    """Verify that a compliant 12-feature dictionary is accepted and converted."""
    is_valid, cleaned = validate_input_features(sample_valid_features)
    assert is_valid is True
    assert isinstance(cleaned, dict)
    assert len([k for k in cleaned if not k.startswith("_")]) == 12
    assert cleaned["CHAS"] == 0
    assert cleaned["RAD"] == 1


def test_missing_feature_rejected(sample_valid_features):
    """Ensure omitting any canonical feature returns an error."""
    incomplete = sample_valid_features.copy()
    del incomplete["RM"]

    is_valid, result = validate_input_features(incomplete)
    assert is_valid is False
    assert "field_errors" in result
    assert "RM" in result["field_errors"]


def test_non_numeric_feature_rejected(sample_valid_features):
    """Ensure malformed string values are rejected."""
    malformed = sample_valid_features.copy()
    malformed["TAX"] = "not_a_number"

    is_valid, result = validate_input_features(malformed)
    assert is_valid is False
    assert "TAX" in result["field_errors"]


def test_chas_strictly_binary(sample_valid_features):
    """Verify CHAS only accepts 0 or 1."""
    invalid_chas = sample_valid_features.copy()
    invalid_chas["CHAS"] = 2

    is_valid, result = validate_input_features(invalid_chas)
    assert is_valid is False
    assert "CHAS" in result["field_errors"]

    valid_chas = sample_valid_features.copy()
    valid_chas["CHAS"] = 1
    is_valid, result = validate_input_features(valid_chas)
    assert is_valid is True
    assert result["CHAS"] == 1


def test_rad_valid_levels(sample_valid_features):
    """Verify RAD is validated against empirical dataset categories (1-8, 24)."""
    # 24 is valid
    rad_24 = sample_valid_features.copy()
    rad_24["RAD"] = 24
    is_valid, result = validate_input_features(rad_24)
    assert is_valid is True
    assert result["RAD"] == 24

    # 9 is not a valid Boston dataset highway index
    invalid_rad = sample_valid_features.copy()
    invalid_rad["RAD"] = 9
    is_valid, result = validate_input_features(invalid_rad)
    assert is_valid is False
    assert "RAD" in result["field_errors"]


def test_nan_and_inf_rejected(sample_valid_features):
    """Verify NaN and infinity are rejected."""
    bad_payload = sample_valid_features.copy()
    bad_payload["CRIM"] = float("nan")

    is_valid, result = validate_input_features(bad_payload)
    assert is_valid is False
    assert "CRIM" in result["field_errors"]

    bad_payload["CRIM"] = float("inf")
    is_valid, result = validate_input_features(bad_payload)
    assert is_valid is False
    assert "CRIM" in result["field_errors"]


def test_bounds_checking_below_minimum(sample_valid_features):
    """Verify that negative or below-minimum values are rejected."""
    negative_crim = sample_valid_features.copy()
    negative_crim["CRIM"] = -0.5

    is_valid, result = validate_input_features(negative_crim)
    assert is_valid is False
    assert "CRIM" in result["field_errors"]


def test_bounds_checking_above_maximum(sample_valid_features):
    """Verify that values exceeding allowable bounds are rejected."""
    excess_rm = sample_valid_features.copy()
    excess_rm["RM"] = 25.0

    is_valid, result = validate_input_features(excess_rm)
    assert is_valid is False
    assert "RM" in result["field_errors"]


def test_unexpected_fields_rejected(sample_valid_features):
    """Verify that unrecognized extra fields trigger validation failure."""
    extra = sample_valid_features.copy()
    extra["B"] = 396.90  # Excluded racial variable
    extra["UNKNOWN_FEATURE"] = 999.0

    is_valid, result = validate_input_features(extra, allow_unexpected=False)
    assert is_valid is False
    assert "_unexpected_fields" in result["field_errors"]


def test_non_dict_payload_rejected():
    """Verify non-dictionary input is safely rejected."""
    is_valid, result = validate_input_features(["not", "a", "dict"])
    assert is_valid is False
    assert "error" in result
