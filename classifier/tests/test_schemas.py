import json

import pytest

from foundation_crack_classifier.labels import LABELS, LABEL_DETAILS
from foundation_crack_classifier.schemas import InferenceResult


def valid_probabilities():
    return {
        "level1": 0.05,
        "level2": 0.82,
        "level3": 0.10,
        "unclear": 0.03,
    }


def canonical_result_kwargs():
    details = LABEL_DETAILS["level2"]
    return {
        "severity_level": details.severity_level,
        "urgency": details.urgency,
        "final_label": "level2",
        "confidence": 0.82,
        "raw_probabilities": valid_probabilities(),
        "why_this_result": "The model found the strongest match with Level 2.",
        "customer_summary": details.customer_summary,
        "disclaimer": details.disclaimer,
        "recommended_action": details.recommended_action,
    }


def test_inference_result_serializes_to_customer_safe_json():
    probabilities = valid_probabilities()
    result = InferenceResult.from_label(
        final_label="level2",
        confidence=0.82,
        raw_probabilities=probabilities,
        why_this_result="The model found the strongest match with Level 2.",
    )
    probabilities["level3"] = 0.99

    payload = result.to_dict()
    encoded = json.dumps(payload)
    details = LABEL_DETAILS["level2"]

    assert set(payload) == {
        "severity_level",
        "urgency",
        "final_label",
        "confidence",
        "raw_probabilities",
        "why_this_result",
        "customer_summary",
        "disclaimer",
        "recommended_action",
    }
    assert payload["severity_level"] == details.severity_level
    assert payload["urgency"] == details.urgency
    assert payload["final_label"] == "level2"
    assert payload["raw_probabilities"]["level3"] == 0.10
    assert payload["customer_summary"] == details.customer_summary
    assert payload["disclaimer"] == details.disclaimer
    assert payload["recommended_action"] == details.recommended_action
    assert "structural diagnosis" in encoded


def test_raw_probabilities_are_immutable_after_construction():
    result = InferenceResult.from_label(
        final_label="level2",
        confidence=0.82,
        raw_probabilities=valid_probabilities(),
        why_this_result="The model found the strongest match with Level 2.",
    )

    with pytest.raises(TypeError):
        result.raw_probabilities["level3"] = 99

    payload = result.to_dict()
    payload["raw_probabilities"]["level3"] = 99

    assert result.raw_probabilities["level3"] == 0.10
    assert result.to_dict()["raw_probabilities"]["level3"] == 0.10
    assert isinstance(result.to_dict()["raw_probabilities"], dict)


@pytest.mark.parametrize(
    "field_name",
    [
        "severity_level",
        "urgency",
        "customer_summary",
        "disclaimer",
        "recommended_action",
    ],
)
def test_direct_construction_with_mismatched_customer_copy_raises_value_error(
    field_name,
):
    kwargs = canonical_result_kwargs()
    kwargs[field_name] = "Non-canonical customer copy."

    with pytest.raises(ValueError, match=field_name):
        InferenceResult(**kwargs)


def test_invalid_final_label_raises_value_error():
    with pytest.raises(ValueError):
        InferenceResult.from_label(
            final_label="level4",
            confidence=0.82,
            raw_probabilities=valid_probabilities(),
            why_this_result="The model found the strongest match with Level 4.",
        )


def test_invalid_confidence_raises_value_error():
    with pytest.raises(ValueError):
        InferenceResult.from_label(
            final_label="level2",
            confidence=1.2,
            raw_probabilities=valid_probabilities(),
            why_this_result="The model found the strongest match with Level 2.",
        )


def test_missing_probability_key_raises_value_error():
    probabilities = valid_probabilities()
    probabilities.pop("unclear")

    with pytest.raises(ValueError):
        InferenceResult.from_label(
            final_label="level2",
            confidence=0.82,
            raw_probabilities=probabilities,
            why_this_result="The model found the strongest match with Level 2.",
        )


def test_out_of_range_probability_raises_value_error():
    probabilities = valid_probabilities()
    probabilities["level3"] = -0.1

    with pytest.raises(ValueError):
        InferenceResult.from_label(
            final_label="level2",
            confidence=0.82,
            raw_probabilities=probabilities,
            why_this_result="The model found the strongest match with Level 2.",
        )


def test_all_labels_can_build_customer_safe_results_from_canonical_copy():
    for label in LABELS:
        result = InferenceResult.from_label(
            final_label=label,
            confidence=0.25,
            raw_probabilities={label_name: 0.25 for label_name in LABELS},
            why_this_result="The model returned an even distribution.",
        )
        details = LABEL_DETAILS[label]

        assert result.severity_level == details.severity_level
        assert result.urgency == details.urgency
        assert result.customer_summary == details.customer_summary
        assert result.disclaimer == details.disclaimer
        assert result.recommended_action == details.recommended_action
