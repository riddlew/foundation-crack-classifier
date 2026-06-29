from foundation_crack_classifier.labels import (
    INDEX_TO_LABEL,
    LABELS,
    LABEL_DETAILS,
    LABEL_TO_INDEX,
)


def test_labels_are_in_stable_order():
    assert LABELS == ["level1", "level2", "level3", "unclear"]
    assert LABEL_TO_INDEX == {
        "level1": 0,
        "level2": 1,
        "level3": 2,
        "unclear": 3,
    }
    assert INDEX_TO_LABEL == {
        0: "level1",
        1: "level2",
        2: "level3",
        3: "unclear",
    }


def test_each_label_has_customer_copy():
    assert set(LABEL_DETAILS) == set(LABELS)

    for label in LABELS:
        details = LABEL_DETAILS[label]
        assert details.severity_level
        assert details.urgency
        assert details.customer_summary
        assert "not a final structural diagnosis" in details.disclaimer.lower()
