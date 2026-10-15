"""Input validation and feature schema definitions for EstateIQ.

Defines the canonical 12-feature schema, empirical boundary rules, and payload sanitization
for Boston Housing valuation inference.
"""

import math
from typing import Any, Dict, List, Set, Tuple

# Valid discrete highway index values observed in the authentic Boston Housing dataset
VALID_RAD_VALUES: Set[int] = {1, 2, 3, 4, 5, 6, 7, 8, 24}

# Schema definitions, empirical training ranges, and bounds
FEATURE_SPECIFICATIONS: Dict[str, Dict[str, Any]] = {
    "CRIM": {
        "name": "Per capita crime rate by town",
        "unit": "ratio",
        "min": 0.0,
        "max": 100.0,
        "empirical_min": 0.00632,
        "empirical_max": 88.976,
        "type": float,
        "default": 0.02,
        "description": "Per capita crime rate by town.",
    },
    "ZN": {
        "name": "Proportion of residential land zoned for lots over 25,000 sq.ft.",
        "unit": "%",
        "min": 0.0,
        "max": 100.0,
        "empirical_min": 0.0,
        "empirical_max": 100.0,
        "type": float,
        "default": 18.0,
        "description": "Zoned residential land over 25k sq ft.",
    },
    "INDUS": {
        "name": "Proportion of non-retail business acres per town",
        "unit": "%",
        "min": 0.0,
        "max": 40.0,
        "empirical_min": 0.46,
        "empirical_max": 27.74,
        "type": float,
        "default": 2.5,
        "description": "Non-retail business acreage proportion.",
    },
    "CHAS": {
        "name": "Charles River dummy variable",
        "unit": "binary (0=No, 1=Yes)",
        "allowed_values": [0, 1],
        "type": int,
        "default": 0,
        "description": "1 if tract bounds Charles River; 0 otherwise.",
    },
    "NOX": {
        "name": "Nitric oxides concentration",
        "unit": "parts per 10 million",
        "min": 0.3,
        "max": 1.0,
        "empirical_min": 0.385,
        "empirical_max": 0.871,
        "type": float,
        "default": 0.45,
        "description": "Nitric oxides concentration (parts per 10 million).",
    },
    "RM": {
        "name": "Average number of rooms per dwelling",
        "unit": "rooms",
        "min": 1.0,
        "max": 15.0,
        "empirical_min": 3.561,
        "empirical_max": 8.780,
        "type": float,
        "default": 6.5,
        "description": "Average number of rooms per dwelling.",
    },
    "AGE": {
        "name": "Proportion of owner-occupied units built prior to 1940",
        "unit": "%",
        "min": 0.0,
        "max": 100.0,
        "empirical_min": 2.9,
        "empirical_max": 100.0,
        "type": float,
        "default": 65.0,
        "description": "Owner-occupied units built before 1940 (%).",
    },
    "DIS": {
        "name": "Weighted distances to five Boston employment centres",
        "unit": "distance index",
        "min": 0.5,
        "max": 20.0,
        "empirical_min": 1.1296,
        "empirical_max": 12.1265,
        "type": float,
        "default": 4.5,
        "description": "Weighted distances to five employment centers.",
    },
    "RAD": {
        "name": "Index of accessibility to radial highways",
        "unit": "highway index",
        "allowed_values": sorted(list(VALID_RAD_VALUES)),
        "type": int,
        "default": 1,
        "description": "Highway accessibility index (valid levels: 1-8, 24).",
    },
    "TAX": {
        "name": "Full-value property-tax rate per $10,000",
        "unit": "rate ($)",
        "min": 100.0,
        "max": 1000.0,
        "empirical_min": 187.0,
        "empirical_max": 711.0,
        "type": float,
        "default": 300.0,
        "description": "Full-value property-tax rate per $10,000.",
    },
    "PTRATIO": {
        "name": "Pupil-teacher ratio by town",
        "unit": "ratio",
        "min": 10.0,
        "max": 30.0,
        "empirical_min": 12.6,
        "empirical_max": 22.0,
        "type": float,
        "default": 15.0,
        "description": "Pupil-teacher ratio by town.",
    },
    "LSTAT": {
        "name": "% lower status of the population",
        "unit": "%",
        "min": 0.1,
        "max": 50.0,
        "empirical_min": 1.73,
        "empirical_max": 37.97,
        "type": float,
        "default": 5.0,
        "description": "% lower economic/educational status of population.",
    },
}

CANONICAL_FEATURE_NAMES: List[str] = list(FEATURE_SPECIFICATIONS.keys())


def validate_input_features(
    raw_data: Any,
    allow_unexpected: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """Validate, sanitize, and convert input features for model inference.

    Args:
        raw_data: Dictionary containing key-value pairs for the 12 canonical features.
        allow_unexpected: If False, extra unrecognized fields will cause validation failure.

    Returns:
        Tuple of (is_valid: bool, result: dict).
        - If valid: (True, cleaned_features_dict)
        - If invalid: (False, error_report_dict with 'error' and 'field_errors')
    """
    if not isinstance(raw_data, dict):
        return False, {
            "error": "Invalid payload format. Expected a JSON object with feature key-value pairs.",
            "field_errors": {"_payload": "Expected a JSON object mapping feature names to numbers."},
        }

    errors: Dict[str, str] = {}
    warnings: Dict[str, str] = {}
    cleaned: Dict[str, float] = {}

    # Check for unexpected fields
    unexpected = [k for k in raw_data.keys() if k not in CANONICAL_FEATURE_NAMES]
    if unexpected and not allow_unexpected:
        errors["_unexpected_fields"] = (
            f"Unexpected field(s) provided: {unexpected}. Only the 12 canonical features are accepted."
        )

    # Check for missing required features
    for feature in CANONICAL_FEATURE_NAMES:
        if feature not in raw_data:
            errors[feature] = f"Missing required feature: '{feature}'."
            continue

        val = raw_data[feature]

        # Convert and check numeric validity
        try:
            numeric_val = float(val)
        except (ValueError, TypeError):
            errors[feature] = f"Value for '{feature}' must be a valid number, got '{val}'."
            continue

        if math.isnan(numeric_val) or math.isinf(numeric_val):
            errors[feature] = f"Value for '{feature}' must be a finite number."
            continue

        spec = FEATURE_SPECIFICATIONS[feature]

        # Strict categorical validation for CHAS (Charles River dummy variable: 0 or 1)
        if feature == "CHAS":
            if numeric_val not in (0.0, 1.0):
                errors[feature] = "CHAS must be exactly 0 (not bounding river) or 1 (bounding river)."
                continue
            cleaned[feature] = int(numeric_val)
            continue

        # Strict accessibility index validation for RAD (empirical dataset categories: 1-8, 24)
        if feature == "RAD":
            int_rad = int(numeric_val)
            if numeric_val != int_rad or int_rad not in VALID_RAD_VALUES:
                errors[feature] = (
                    f"RAD must be a valid integer accessibility index in {sorted(list(VALID_RAD_VALUES))}."
                )
                continue
            cleaned[feature] = int_rad
            continue

        # Range bounds validation
        min_bound = spec.get("min")
        max_bound = spec.get("max")
        if min_bound is not None and numeric_val < min_bound:
            errors[feature] = f"Value {numeric_val} is below allowable minimum of {min_bound}."
            continue
        if max_bound is not None and numeric_val > max_bound:
            errors[feature] = f"Value {numeric_val} exceeds allowable maximum of {max_bound}."
            continue

        # Soft empirical warning (if outside 1970 historical range, recorded as advisory)
        emp_min = spec.get("empirical_min")
        emp_max = spec.get("empirical_max")
        if emp_min is not None and emp_max is not None:
            if numeric_val < emp_min or numeric_val > emp_max:
                warnings[feature] = (
                    f"Value {numeric_val} is outside historical 1970 empirical range [{emp_min}, {emp_max}]."
                )

        cleaned[feature] = numeric_val

    if errors:
        return False, {"error": "Validation failed", "field_errors": errors}

    result = cleaned
    if warnings:
        result["_warnings"] = warnings  # type: ignore

    return True, result
