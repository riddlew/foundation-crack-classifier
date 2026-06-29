from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from foundation_crack_classifier.labels import LABEL_DETAILS, LABELS
from foundation_crack_classifier.schemas import InferenceResult


@dataclass(frozen=True)
class ThresholdConfig:
    minimum_confidence: float = 0.60
    urgent_threshold: float = 0.30
    concerning_threshold: float = 0.35
    minimum_level1_confidence: float = 0.75


def apply_thresholds(
    probabilities: dict[str, float],
    config: ThresholdConfig | None = None,
) -> InferenceResult:
    config = config or ThresholdConfig()
    normalized = _normalized_probabilities(probabilities)
    predicted_label, confidence = max(normalized.items(), key=lambda item: item[1])

    if _meets_threshold(normalized["level3"], config.urgent_threshold):
        final_label = "level3"
        reason = "The Level 3 probability crossed the urgent-review threshold."
    elif _meets_threshold(normalized["level2"], config.concerning_threshold):
        final_label = "level2"
        reason = "The Level 2 probability crossed the contact-soon threshold."
    elif confidence < config.minimum_confidence:
        final_label = "unclear"
        reason = "The model confidence was too low for a reliable severity call."
    elif predicted_label == "level1" and confidence < config.minimum_level1_confidence:
        final_label = "unclear"
        reason = (
            "The image looked lower severity, but confidence was not strong "
            "enough to reassure."
        )
    else:
        final_label = predicted_label
        reason = (
            "The model found the strongest match with "
            f"{LABEL_DETAILS[final_label].severity_level}."
        )

    rounded_probabilities = {
        label: round(float(normalized[label]), 6) for label in LABELS
    }

    return InferenceResult.from_label(
        final_label=final_label,
        confidence=rounded_probabilities[final_label],
        raw_probabilities=rounded_probabilities,
        why_this_result=reason,
    )


def _meets_threshold(value: float, threshold: float) -> bool:
    return value >= threshold or math.isclose(
        value, threshold, rel_tol=0.0, abs_tol=1e-12
    )


def _normalized_probabilities(probabilities: dict[str, float]) -> dict[str, float]:
    expected_labels = set(LABELS)
    actual_labels = set(probabilities)
    if actual_labels != expected_labels:
        missing = sorted(expected_labels - actual_labels)
        unexpected = sorted(actual_labels - expected_labels)
        raise ValueError(
            "Probability labels must exactly match LABELS. "
            f"Missing: {missing}. Unexpected: {unexpected}."
        )

    for label in LABELS:
        value = probabilities[label]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("Probability values must be real numbers.")
        if not math.isfinite(value):
            raise ValueError("Probability values must be finite.")
        if value < 0:
            raise ValueError("Probability values cannot be negative.")
        if value > 1.0:
            raise ValueError("Probability values cannot exceed 1.0.")

    values = {label: float(probabilities[label]) for label in LABELS}
    raw_total = sum(values.values())
    if raw_total <= 0:
        raise ValueError("Probability values must sum to a positive number.")

    return {label: values[label] / raw_total for label in LABELS}
