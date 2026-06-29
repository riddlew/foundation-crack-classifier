from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType

from foundation_crack_classifier.labels import LABELS, LABEL_DETAILS


def _is_probability(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and 0 <= value <= 1


def _validate_canonical_field(field_name: str, actual: str, expected: str) -> None:
    if actual != expected:
        raise ValueError(
            f"{field_name} must match canonical LABEL_DETAILS copy for final_label"
        )


@dataclass(frozen=True)
class InferenceResult:
    severity_level: str
    urgency: str
    final_label: str
    confidence: float
    raw_probabilities: Mapping[str, float]
    why_this_result: str
    customer_summary: str
    disclaimer: str
    recommended_action: str

    @classmethod
    def from_label(
        cls,
        final_label: str,
        confidence: float,
        raw_probabilities: Mapping[str, float],
        why_this_result: str,
    ) -> InferenceResult:
        if final_label not in LABEL_DETAILS:
            raise ValueError(f"Unknown final_label: {final_label}")

        details = LABEL_DETAILS[final_label]
        return cls(
            severity_level=details.severity_level,
            urgency=details.urgency,
            final_label=final_label,
            confidence=confidence,
            raw_probabilities=raw_probabilities,
            why_this_result=why_this_result,
            customer_summary=details.customer_summary,
            disclaimer=details.disclaimer,
            recommended_action=details.recommended_action,
        )

    def __post_init__(self) -> None:
        if self.final_label not in LABELS:
            raise ValueError(f"final_label must be one of {LABELS}")

        details = LABEL_DETAILS[self.final_label]
        _validate_canonical_field(
            "severity_level", self.severity_level, details.severity_level
        )
        _validate_canonical_field("urgency", self.urgency, details.urgency)
        _validate_canonical_field(
            "customer_summary", self.customer_summary, details.customer_summary
        )
        _validate_canonical_field("disclaimer", self.disclaimer, details.disclaimer)
        _validate_canonical_field(
            "recommended_action", self.recommended_action, details.recommended_action
        )

        if not _is_probability(self.confidence):
            raise ValueError("confidence must be numeric and between 0 and 1")

        if set(self.raw_probabilities) != set(LABELS):
            raise ValueError("raw_probabilities keys must exactly match LABELS")

        for label, probability in self.raw_probabilities.items():
            if not _is_probability(probability):
                raise ValueError(
                    f"raw_probabilities[{label!r}] must be numeric and between 0 and 1"
                )

        object.__setattr__(
            self, "raw_probabilities", MappingProxyType(dict(self.raw_probabilities))
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "severity_level": self.severity_level,
            "urgency": self.urgency,
            "final_label": self.final_label,
            "confidence": self.confidence,
            "raw_probabilities": dict(self.raw_probabilities),
            "why_this_result": self.why_this_result,
            "customer_summary": self.customer_summary,
            "disclaimer": self.disclaimer,
            "recommended_action": self.recommended_action,
        }
