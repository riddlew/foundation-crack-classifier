from __future__ import annotations

from dataclasses import dataclass


LABELS = ["level1", "level2", "level3", "unclear"]
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}


@dataclass(frozen=True)
class LabelDetails:
    severity_level: str
    urgency: str
    customer_summary: str
    recommended_action: str
    disclaimer: str


DISCLAIMER = (
    "This AI result is based on a single photo and is not a final structural "
    "diagnosis. A qualified foundation professional should inspect the area."
)


LABEL_DETAILS = {
    "level1": LabelDetails(
        severity_level="Level 1",
        urgency="inspection_recommended",
        customer_summary=(
            "The photo appears most consistent with a lower-severity crack, "
            "but a photo cannot confirm whether the foundation is structurally sound."
        ),
        recommended_action="Consider scheduling an inspection.",
        disclaimer=DISCLAIMER,
    ),
    "level2": LabelDetails(
        severity_level="Level 2",
        urgency="contact_soon",
        customer_summary=(
            "Visible signs in the photo may indicate a more concerning "
            "foundation crack."
        ),
        recommended_action="Contact a qualified foundation repair professional soon.",
        disclaimer=DISCLAIMER,
    ),
    "level3": LabelDetails(
        severity_level="Level 3",
        urgency="contact_immediately",
        customer_summary=(
            "Visible signs may indicate elevated collapse risk or severe wall distress."
        ),
        recommended_action=(
            "Avoid the area if it appears unsafe and contact a qualified "
            "professional immediately."
        ),
        disclaimer=DISCLAIMER,
    ),
    "unclear": LabelDetails(
        severity_level="Unclear",
        urgency="unable_to_assess",
        customer_summary=(
            "The photo is not clear enough for a reliable AI triage result."
        ),
        recommended_action="A professional inspection is recommended.",
        disclaimer=DISCLAIMER,
    ),
}
