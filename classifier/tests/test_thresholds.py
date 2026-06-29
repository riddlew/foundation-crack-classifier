import pytest

from foundation_crack_classifier.thresholds import ThresholdConfig, apply_thresholds


def test_low_confidence_becomes_unclear():
    result = apply_thresholds(
        {"level1": 0.40, "level2": 0.25, "level3": 0.20, "unclear": 0.15},
        ThresholdConfig(minimum_confidence=0.60),
    )

    assert result.final_label == "unclear"
    assert result.severity_level == "Unclear"


def test_level3_probability_overrides_top_level1_prediction():
    result = apply_thresholds(
        {"level1": 0.54, "level2": 0.05, "level3": 0.31, "unclear": 0.10},
        ThresholdConfig(urgent_threshold=0.30),
    )

    assert result.final_label == "level3"
    assert result.urgency == "contact_immediately"
    assert result.confidence == 0.31


def test_level2_probability_routes_to_contact_soon():
    result = apply_thresholds(
        {"level1": 0.50, "level2": 0.34, "level3": 0.08, "unclear": 0.08},
        ThresholdConfig(concerning_threshold=0.30),
    )

    assert result.final_label == "level2"
    assert result.urgency == "contact_soon"
    assert result.confidence == 0.34


def test_weak_level1_prediction_becomes_unclear():
    result = apply_thresholds(
        {"level1": 0.62, "level2": 0.20, "level3": 0.09, "unclear": 0.09},
        ThresholdConfig(minimum_level1_confidence=0.75),
    )

    assert result.final_label == "unclear"


def test_all_zero_probabilities_raise_value_error():
    with pytest.raises(
        ValueError, match="Probability values must sum to a positive number."
    ):
        apply_thresholds(
            {"level1": 0.0, "level2": 0.0, "level3": 0.0, "unclear": 0.0}
        )


def test_non_positive_raw_total_with_negative_probability_raises_value_error():
    with pytest.raises(ValueError, match="Probability values cannot be negative."):
        apply_thresholds(
            {"level1": 1.0, "level2": -2.0, "level3": 0.0, "unclear": 0.0}
        )


def test_any_negative_probability_raises_value_error():
    with pytest.raises(ValueError, match="Probability values cannot be negative."):
        apply_thresholds(
            {"level1": 0.80, "level2": -0.10, "level3": 0.20, "unclear": 0.10}
        )


def test_extra_probability_key_raises_value_error():
    with pytest.raises(ValueError, match="Probability labels must exactly match LABELS."):
        apply_thresholds(
            {
                "level1": 0.40,
                "level2": 0.30,
                "level3": 0.20,
                "unclear": 0.10,
                "other": 0.01,
            }
        )


def test_missing_probability_key_raises_value_error():
    with pytest.raises(ValueError, match="Probability labels must exactly match LABELS."):
        apply_thresholds({"level1": 0.40, "level2": 0.30, "level3": 0.20})


def test_boolean_probability_value_raises_value_error():
    with pytest.raises(ValueError, match="Probability values must be real numbers."):
        apply_thresholds(
            {"level1": True, "level2": 0.30, "level3": 0.20, "unclear": 0.10}
        )


def test_numeric_string_probability_value_raises_value_error():
    with pytest.raises(ValueError, match="Probability values must be real numbers."):
        apply_thresholds(
            {"level1": "0.40", "level2": 0.30, "level3": 0.20, "unclear": 0.10}
        )


def test_nan_probability_value_raises_value_error():
    with pytest.raises(ValueError, match="Probability values must be finite."):
        apply_thresholds(
            {
                "level1": float("nan"),
                "level2": 0.30,
                "level3": 0.20,
                "unclear": 0.10,
            }
        )


def test_inf_probability_value_raises_value_error():
    with pytest.raises(ValueError, match="Probability values must be finite."):
        apply_thresholds(
            {
                "level1": float("inf"),
                "level2": 0.30,
                "level3": 0.20,
                "unclear": 0.10,
            }
        )


def test_probability_value_greater_than_one_raises_value_error():
    with pytest.raises(ValueError, match="Probability values cannot exceed 1.0."):
        apply_thresholds(
            {"level1": 1.01, "level2": 0.30, "level3": 0.20, "unclear": 0.10}
        )


def test_exact_urgent_threshold_routes_to_level3():
    result = apply_thresholds(
        {"level1": 0.55, "level2": 0.05, "level3": 0.30, "unclear": 0.10},
        ThresholdConfig(urgent_threshold=0.30),
    )

    assert result.final_label == "level3"
    assert result.confidence == 0.30


def test_exact_concerning_threshold_routes_to_level2():
    result = apply_thresholds(
        {"level1": 0.52, "level2": 0.30, "level3": 0.08, "unclear": 0.10},
        ThresholdConfig(concerning_threshold=0.30),
    )

    assert result.final_label == "level2"
    assert result.confidence == 0.30


def test_level3_probability_just_below_threshold_does_not_round_up_to_level3():
    result = apply_thresholds(
        {
            "level1": 0.5500004,
            "level2": 0.05,
            "level3": 0.2999996,
            "unclear": 0.10,
        },
        ThresholdConfig(urgent_threshold=0.30),
    )

    assert result.final_label == "unclear"


def test_weak_level1_confidence_just_below_threshold_does_not_round_up_to_level1():
    result = apply_thresholds(
        {
            "level1": 0.7499996,
            "level2": 0.1000004,
            "level3": 0.05,
            "unclear": 0.10,
        },
        ThresholdConfig(minimum_level1_confidence=0.75),
    )

    assert result.final_label == "unclear"
