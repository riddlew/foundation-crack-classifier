from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from foundation_crack_api.classifier_service import ImageDecodeError
from foundation_crack_api.main import app, get_classifier_service


class FakeClassifierService:
    def classify_bytes(self, image_bytes: bytes) -> dict[str, object]:
        if image_bytes == b"bad image":
            raise ImageDecodeError("Unable to read image file.")
        return {
            "severity_level": "Level 2",
            "urgency": "contact_soon",
            "final_label": "level2",
            "confidence": 0.71,
            "raw_probabilities": {
                "level1": 0.08,
                "level2": 0.71,
                "level3": 0.13,
                "unclear": 0.08,
            },
            "why_this_result": "The Level 2 probability crossed the contact-soon threshold.",
            "customer_summary": "Visible signs in the photo may indicate a more concerning foundation crack. Contact a qualified foundation repair professional soon.",
            "disclaimer": "This AI result is not a final structural diagnosis.",
            "recommended_action": "Contact a qualified foundation repair professional soon.",
        }


@pytest.fixture
def client() -> Iterator[TestClient]:
    app.dependency_overrides[get_classifier_service] = lambda: FakeClassifierService()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
